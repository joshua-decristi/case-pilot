"""Live eval runner (HTTP to backend) and offline report grading.

Live path never sends ground-truth labels. Paid calls only with ``--live``.

**Budget ownership:** backend is the sole CostLedger mutator. This runner does
NOT reserve/settle (no double-charging). It:
- enforces opt-in ``--live``, ``--max-cases > 0``, ``--budget-usd ≤ 5``, GLM-only
- reads ledger snapshot (if present) for remaining global budget before cases
- tracks per-eval spend from backend-reported cost metadata when available
- pauses when backend reports budget errors or session threshold is hit
- captures usage / provider model metadata when returned (reports gaps otherwise)
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

from casepilot.evaluation.cost import (
    ALLOWED_LIVE_MODEL,
    DEFAULT_BUDGET_USD,
    DEFAULT_LEDGER_PATH,
    CostPause,
    validate_live_model,
)
from casepilot.evaluation.grader import (
    GroundTruth,
    Scorecard,
    grade_results,
    load_ground_truth,
)
from casepilot.evaluation.metadata import RunMetadata, collect_metadata
from casepilot.shared.contracts import (
    BackendRoutes,
    InvestigationResult,
    StartInvestigationRequest,
    StartInvestigationResponse,
)

DEFAULT_OBJECTIVE = "Investigate why this case is stalled and recommend the next action."
DEFAULT_REPORT_DIR = Path("evals/reports")


@dataclass
class LiveEvalConfig:
    live: bool = False
    split: str = "evaluation"
    max_cases: int = 0  # live requires > 0
    budget_usd: float = DEFAULT_BUDGET_USD  # per-eval session threshold ≤ global $5
    model: str = ALLOWED_LIVE_MODEL
    backend_base_url: str = "http://127.0.0.1:8000"
    timeout_seconds: float = 120.0
    objective: str = DEFAULT_OBJECTIVE
    labels_path: Path | None = None
    report_dir: Path = DEFAULT_REPORT_DIR
    # Shared ledger path — read-only for eval; backend mutates.
    ledger_path: Path | None = None
    enforce_glm_only: bool = True


@dataclass
class EvalCaseResult:
    case_id: str
    ok: bool
    result: dict[str, Any] | None = None
    error: str | None = None
    latency_s: float | None = None
    usage: dict[str, Any] | None = None
    cost_usd: float | None = None
    cost_estimated: bool = True
    provider_model: str | None = None
    run_id: str | None = None
    gaps: list[str] = field(default_factory=list)

    def investigation_result(self) -> InvestigationResult | None:
        if self.result is None:
            return None
        try:
            return InvestigationResult.model_validate(self.result)
        except Exception:  # noqa: BLE001
            return None


@dataclass
class EvalReport:
    metadata: RunMetadata
    scorecard: Scorecard
    case_results: list[EvalCaseResult]
    ledger_snapshot: dict[str, Any] | None
    live: bool
    session_spend_usd: float = 0.0
    gaps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": self.metadata.to_dict(),
            "live": self.live,
            "session_spend_usd": self.session_spend_usd,
            "gaps": list(self.gaps),
            "ledger": self.ledger_snapshot,
            "scorecard": self.scorecard.to_dict(),
            "cases": [asdict(c) for c in self.case_results],
        }


def _extract_run_meta(
    payload: dict[str, Any],
) -> tuple[
    dict[str, Any] | None,
    float | None,
    str | None,
    str | None,
    list[str],
]:
    """Pull usage / cost / provider model / run_id if backend surfaces them.

    Gaps are reported only — this package does not modify the backend.
    """
    gaps: list[str] = []
    usage: dict[str, Any] | None = None
    cost: float | None = None
    provider_model: str | None = None
    run_id: str | None = None

    def _walk(obj: dict[str, Any], prefix: str = "") -> None:
        nonlocal usage, cost, provider_model, run_id
        if run_id is None and isinstance(obj.get("run_id"), str):
            run_id = obj["run_id"]
        for key in ("usage", "token_usage", "model_usage"):
            if usage is None and isinstance(obj.get(key), dict):
                usage = obj[key]
        for key in ("cost_usd", "total_cost_usd"):
            if cost is None and isinstance(obj.get(key), (int, float)):
                cost = float(obj[key])
        if cost is None and isinstance(obj.get("cost"), (int, float)):
            cost = float(obj["cost"])
        for key in ("provider_model", "model", "model_id"):
            if provider_model is None and isinstance(obj.get(key), str):
                # Avoid treating diagnosis-like fields as model; only top-ish keys
                if key == "model" and prefix == "result.":
                    continue
                provider_model = obj[key]
        if isinstance(obj.get("provider"), dict):
            prov = obj["provider"]
            if provider_model is None and isinstance(prov.get("model"), str):
                provider_model = prov["model"]
            if usage is None and isinstance(prov.get("usage"), dict):
                usage = prov["usage"]
            if cost is None and isinstance(prov.get("cost_usd"), (int, float)):
                cost = float(prov["cost_usd"])

    _walk(payload)
    if isinstance(payload.get("result"), dict):
        _walk(payload["result"], prefix="result.")
    if isinstance(payload.get("meta"), dict):
        _walk(payload["meta"], prefix="meta.")
    if isinstance(payload.get("trace_summary"), dict):
        _walk(payload["trace_summary"], prefix="trace_summary.")

    # usage nested token fields often carry cost
    if usage and cost is None:
        for key in ("cost", "cost_usd", "total_cost"):
            if isinstance(usage.get(key), (int, float)):
                cost = float(usage[key])
                break

    if usage is None:
        gaps.append(
            "backend response has no usage/token metadata "
            "(eval reports gap; does not modify backend)"
        )
    if cost is None:
        gaps.append("backend response has no cost_usd; session threshold uses only known costs")
    if provider_model is None:
        gaps.append("backend response has no provider_model/model metadata")
    return usage, cost, provider_model, run_id, gaps


def _post_investigation(
    client: httpx.Client,
    case_id: str,
    objective: str,
) -> tuple[InvestigationResult | None, dict[str, Any], list[str], str | None]:
    """POST start investigation. Sends only case_id + objective (no labels)."""
    body = StartInvestigationRequest(case_id=case_id, objective=objective)
    payload = body.model_dump(mode="json")
    assert "expected" not in payload and "label" not in payload
    try:
        resp = client.post(BackendRoutes.INVESTIGATIONS, json=payload)
    except httpx.RequestError as exc:
        return None, {}, [], f"backend unreachable: {exc}"
    gaps: list[str] = []
    try:
        data = resp.json()
    except ValueError:
        return None, {}, gaps, f"non-JSON response HTTP {resp.status_code}"
    if not isinstance(data, dict):
        return None, {}, gaps, f"unexpected JSON type HTTP {resp.status_code}"
    if resp.status_code >= 400:
        detail = data.get("detail", data)
        detail_s = detail if isinstance(detail, str) else json.dumps(detail)
        # Backend cost guard surfaces as 402/429/503-style messages
        lower = detail_s.lower()
        if resp.status_code in {402, 429} or "budget" in lower or "cost pause" in lower:
            return None, data, gaps, f"backend cost guard HTTP {resp.status_code}: {detail_s}"
        return None, data, gaps, f"HTTP {resp.status_code}: {detail_s}"
    try:
        parsed = StartInvestigationResponse.model_validate(data)
        return parsed.result, data, gaps, None
    except Exception as exc:  # noqa: BLE001
        if "result" in data:
            try:
                result = InvestigationResult.model_validate(data["result"])
                return result, data, gaps, None
            except Exception as exc2:  # noqa: BLE001
                return None, data, gaps, f"invalid result payload: {exc2}"
        return None, data, gaps, f"invalid investigation response: {exc}"


def _read_ledger_snapshot(path: Path) -> dict[str, Any] | None:
    """Observe existing ledger without creating/mutating it."""
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except OSError, json.JSONDecodeError:
        return None
    return data if isinstance(data, dict) else None


def run_live(
    config: LiveEvalConfig,
    *,
    http_client: httpx.Client | None = None,
    gt: GroundTruth | None = None,
) -> EvalReport:
    """Run live evaluations against the backend. Requires config.live is True.

    Does **not** mutate CostLedger. Backend owns reservation/accounting.
    """
    if not config.live:
        raise RuntimeError("run_live refused: config.live is False (pass --live)")
    if config.max_cases <= 0:
        raise RuntimeError("live mode requires explicit max_cases > 0")
    if config.budget_usd <= 0:
        raise CostPause("budget_usd must be positive")
    if config.budget_usd > DEFAULT_BUDGET_USD + 1e-12:
        raise CostPause(
            f"budget_usd {config.budget_usd} exceeds initial envelope "
            f"${DEFAULT_BUDGET_USD:.2f}; obtain explicit permission to raise"
        )
    if config.enforce_glm_only:
        validate_live_model(config.model)

    ground = gt or load_ground_truth(config.labels_path)
    case_ids = list(ground.case_ids_for_split(config.split))[: config.max_cases]

    ledger_path = Path(config.ledger_path or DEFAULT_LEDGER_PATH)
    ledger_snap = _read_ledger_snapshot(ledger_path)

    meta = collect_metadata(
        labels_path=Path(ground.source_path),
        model=config.model,
        split=config.split,
    )
    gaps: list[str] = [
        "Backend is sole CostLedger mutator; eval does not reserve/settle "
        "(avoids double-charging).",
        "Live spend is application-enforced; not a provider hard cap.",
    ]
    if ledger_snap is None:
        gaps.append(
            f"no ledger at {ledger_path} yet — cannot pre-check global remaining; "
            "rely on backend cost guard + session threshold from reported costs"
        )
    else:
        remaining = max(
            0.0,
            float(ledger_snap.get("budget_usd") or DEFAULT_BUDGET_USD)
            - float(ledger_snap.get("spent_usd") or 0)
            - float(ledger_snap.get("reserved_usd") or 0),
        )
        if remaining <= 0:
            raise CostPause(f"global ledger remaining ${remaining:.4f} at {ledger_path}; paused")

    case_results: list[EvalCaseResult] = []
    result_map: dict[str, InvestigationResult | None] = {}
    errors: dict[str, str] = {}
    session_spend = 0.0
    paused = False
    pause_reason = ""

    owns = http_client is None
    client = http_client or httpx.Client(
        base_url=config.backend_base_url.rstrip("/"),
        timeout=httpx.Timeout(config.timeout_seconds),
        headers={"Accept": "application/json"},
    )
    try:
        for cid in case_ids:
            if paused:
                case_results.append(
                    EvalCaseResult(
                        case_id=cid, ok=False, error=f"skipped after pause: {pause_reason}"
                    )
                )
                result_map[cid] = None
                errors[cid] = f"skipped after pause: {pause_reason}"
                continue

            # Re-check global ledger remaining (backend may have settled other runs).
            snap = _read_ledger_snapshot(ledger_path)
            if snap is not None:
                rem = max(
                    0.0,
                    float(snap.get("budget_usd") or DEFAULT_BUDGET_USD)
                    - float(snap.get("spent_usd") or 0)
                    - float(snap.get("reserved_usd") or 0),
                )
                if rem <= 0:
                    paused = True
                    pause_reason = f"global ledger remaining ${rem:.4f}"
                    gaps.append(f"paused before {cid}: {pause_reason}")
                    case_results.append(EvalCaseResult(case_id=cid, ok=False, error=pause_reason))
                    result_map[cid] = None
                    errors[cid] = pause_reason
                    continue

            # Per-eval session threshold from known backend-reported costs.
            if session_spend >= config.budget_usd - 1e-12:
                paused = True
                pause_reason = (
                    f"per-eval session spend ${session_spend:.4f} "
                    f">= threshold ${config.budget_usd:.4f}"
                )
                gaps.append(f"paused before {cid}: {pause_reason}")
                case_results.append(EvalCaseResult(case_id=cid, ok=False, error=pause_reason))
                result_map[cid] = None
                errors[cid] = pause_reason
                continue

            t0 = datetime.now(UTC)
            inv, raw, call_gaps, err = _post_investigation(client, cid, config.objective)
            latency = (datetime.now(UTC) - t0).total_seconds()
            usage, cost, provider_model, run_id, meta_gaps = _extract_run_meta(raw)
            all_gaps = list(call_gaps) + list(meta_gaps)
            gaps.extend(f"{cid}: {g}" for g in all_gaps)

            if provider_model and provider_model != config.model:
                gaps.append(
                    f"{cid}: provider_model {provider_model!r} != configured {config.model!r}"
                )

            if err:
                case_results.append(
                    EvalCaseResult(
                        case_id=cid,
                        ok=False,
                        error=err,
                        latency_s=latency,
                        usage=usage,
                        cost_usd=cost,
                        cost_estimated=cost is None,
                        provider_model=provider_model,
                        run_id=run_id,
                        gaps=all_gaps,
                    )
                )
                result_map[cid] = None
                errors[cid] = err
                if "cost guard" in err.lower() or "budget" in err.lower():
                    paused = True
                    pause_reason = err
                    gaps.append(f"paused after backend cost guard on {cid}")
                continue

            if cost is not None:
                session_spend += cost
            else:
                # Without backend cost, cannot enforce dollar threshold tightly —
                # report gap; still rely on backend ledger + max_cases.
                gaps.append(f"{cid}: cost unknown from backend — session_spend not increased")

            case_results.append(
                EvalCaseResult(
                    case_id=cid,
                    ok=True,
                    result=inv.model_dump(mode="json") if inv else None,
                    latency_s=latency,
                    usage=usage,
                    cost_usd=cost,
                    cost_estimated=cost is None,
                    provider_model=provider_model,
                    run_id=run_id,
                    gaps=all_gaps,
                )
            )
            result_map[cid] = inv

            if session_spend >= config.budget_usd - 1e-12:
                paused = True
                pause_reason = (
                    f"per-eval session spend ${session_spend:.4f} "
                    f"reached threshold ${config.budget_usd:.4f}"
                )
                gaps.append(pause_reason)
    finally:
        if owns:
            client.close()

    final_snap = _read_ledger_snapshot(ledger_path)
    scorecard = grade_results(ground, result_map, errors=errors)
    report = EvalReport(
        metadata=meta,
        scorecard=scorecard,
        case_results=case_results,
        ledger_snapshot=final_snap,
        live=True,
        session_spend_usd=session_spend,
        gaps=list(dict.fromkeys(gaps)),
    )
    _write_report(config.report_dir, report)
    return report


def run_offline_report(
    results_path: Path,
    *,
    labels_path: Path | None = None,
    report_dir: Path = DEFAULT_REPORT_DIR,
    split: str | None = None,
) -> EvalReport:
    """Grade a previously saved results JSON without any model/backend calls."""
    data = json.loads(results_path.read_text(encoding="utf-8"))
    ground = load_ground_truth(labels_path)
    result_map: dict[str, InvestigationResult | None] = {}
    errors: dict[str, str] = {}
    case_results: list[EvalCaseResult] = []

    if isinstance(data, dict) and isinstance(data.get("cases"), list):
        for row in data["cases"]:
            cid = str(row["case_id"])
            if split and split != "all":
                label = ground.cases.get(cid)
                if label is None or (label.split and label.split != split):
                    continue
            err = row.get("error")
            raw_result = row.get("result")
            inv: InvestigationResult | None = None
            if raw_result is not None:
                inv = InvestigationResult.model_validate(raw_result)
            result_map[cid] = inv
            if err:
                errors[cid] = str(err)
            case_results.append(
                EvalCaseResult(
                    case_id=cid,
                    ok=inv is not None and err is None,
                    result=raw_result,
                    error=str(err) if err else None,
                    usage=row.get("usage") if isinstance(row.get("usage"), dict) else None,
                    cost_usd=float(row["cost_usd"])
                    if isinstance(row.get("cost_usd"), (int, float))
                    else None,
                    provider_model=row.get("provider_model")
                    if isinstance(row.get("provider_model"), str)
                    else None,
                )
            )
    elif isinstance(data, dict):
        for cid, raw in data.items():
            if cid in {"metadata", "scorecard", "gaps", "live", "ledger", "session_spend_usd"}:
                continue
            if isinstance(raw, dict) and "diagnosis" in raw:
                inv = InvestigationResult.model_validate(raw)
                result_map[cid] = inv
                case_results.append(
                    EvalCaseResult(
                        case_id=cid,
                        ok=True,
                        result=inv.model_dump(mode="json"),
                    )
                )
            elif isinstance(raw, dict) and "result" in raw:
                inv = (
                    InvestigationResult.model_validate(raw["result"])
                    if raw.get("result") is not None
                    else None
                )
                err = raw.get("error")
                result_map[cid] = inv
                if err:
                    errors[cid] = str(err)
                case_results.append(
                    EvalCaseResult(
                        case_id=cid,
                        ok=inv is not None,
                        result=raw.get("result"),
                        error=str(err) if err else None,
                    )
                )
    else:
        raise ValueError("unrecognized offline results JSON shape")

    meta = collect_metadata(
        labels_path=Path(ground.source_path),
        model=None,
        split=split or "offline",
    )
    meta.notes.append(f"offline grade from {results_path}")
    scorecard = grade_results(ground, result_map, errors=errors)
    report = EvalReport(
        metadata=meta,
        scorecard=scorecard,
        case_results=case_results,
        ledger_snapshot=None,
        live=False,
        gaps=["offline report: no paid calls"],
    )
    _write_report(report_dir, report, prefix="offline")
    return report


def _write_report(report_dir: Path, report: EvalReport, *, prefix: str = "eval") -> Path:
    report_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = report_dir / f"{prefix}_{ts}.json"
    blob = json.dumps(report.to_dict(), indent=2)
    path.write_text(blob, encoding="utf-8")
    (report_dir / f"{prefix}_latest.json").write_text(blob, encoding="utf-8")
    return path
