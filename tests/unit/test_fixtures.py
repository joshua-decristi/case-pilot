"""Integrity checks for enterprise fixtures and evaluator-only ground truth."""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
FIXTURE_PATH = ROOT / "fixtures" / "enterprise" / "cases.json"
GROUND_TRUTH_PATH = ROOT / "evals" / "ground_truth" / "labels.json"

CASE_REQUIRED = {"id", "status", "symptom", "application", "required_documents"}
DOC_REQUIRED = {"id", "case_id", "document_type", "status", "fields", "content"}
EVENT_REQUIRED = {"id", "case_id", "event_type", "status", "details", "created_at"}
GT_CASE_REQUIRED = {
    "expected_categories",
    "acceptable_actions",
    "required_evidence",
    "require_uncertainty",
}
ALLOWED_CATEGORIES = {
    "missing_document",
    "data_mismatch",
    "processing_failure",
    "indeterminate",
    "no_issue",
}
ALLOWED_ACTIONS = {
    "request_document",
    "reconcile_data",
    "investigate_processing",
    "escalate",
    "no_action",
}
LEAK_KEY_RE = re.compile(
    r"(expected_categor|acceptable_action|require_uncertainty|ground[_\s-]?truth|"
    r"root[_\s-]?cause|required_evidence|development|evaluation)",
    re.I,
)
LABEL_TOKENS = (
    "missing_document",
    "data_mismatch",
    "processing_failure",
    "indeterminate",
    "no_issue",
    "request_document",
    "reconcile_data",
    "investigate_processing",
    "escalate",
    "no_action",
)
TIMESTAMP_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
INJECTION_HINT = re.compile(r"ignore prior|ignore previous|overrides all other", re.I)


@pytest.fixture(scope="module")
def fixture_data() -> dict[str, Any]:
    assert FIXTURE_PATH.is_file(), f"missing fixture: {FIXTURE_PATH}"
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ground_truth() -> dict[str, Any]:
    assert GROUND_TRUTH_PATH.is_file(), f"missing ground truth: {GROUND_TRUTH_PATH}"
    return json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))


def test_fixture_record_schema_links_and_timestamps(fixture_data: dict[str, Any]) -> None:
    """Enterprise cases/docs/events validate, cross-link, and use fixed timestamps."""
    assert set(fixture_data) >= {"cases", "documents", "events"}, (
        f"fixture top-level keys missing domains: {sorted(fixture_data)}"
    )
    assert isinstance(fixture_data["cases"], list)
    assert isinstance(fixture_data["documents"], list)
    assert isinstance(fixture_data["events"], list)
    assert len(fixture_data["cases"]) == 20, f"expected 20 cases, got {len(fixture_data['cases'])}"
    assert fixture_data.get("generated_at") == "2024-06-01T00:00:00Z", (
        "fixture generated_at must be the fixed deterministic literal"
    )

    for case in fixture_data["cases"]:
        cid = case.get("id", "<missing-id>")
        assert CASE_REQUIRED <= set(case), f"{cid} missing case fields: {CASE_REQUIRED - set(case)}"
        assert isinstance(case["id"], str) and case["id"], f"case id must be non-empty str: {cid}"
        assert isinstance(case["status"], str) and case["status"], f"{cid}: empty status"
        assert isinstance(case["symptom"], str) and case["symptom"], f"{cid}: empty symptom"
        assert isinstance(case["application"], dict) and case["application"], (
            f"{cid}: application must be non-empty dict"
        )
        assert isinstance(case["required_documents"], list) and case["required_documents"], (
            f"{cid}: required_documents must be non-empty list"
        )
        assert all(isinstance(x, str) for x in case["required_documents"]), (
            f"{cid}: required_documents entries must be strings"
        )

    for doc in fixture_data["documents"]:
        did = doc.get("id", "<missing-id>")
        assert DOC_REQUIRED <= set(doc), f"{did} missing doc fields: {DOC_REQUIRED - set(doc)}"
        assert isinstance(doc["fields"], dict), f"{did}: fields must be dict"
        assert isinstance(doc["content"], str), f"{did}: content must be str"

    stamps = [e["created_at"] for e in fixture_data["events"]]
    assert stamps, "events must be non-empty for stable timestamp coverage"
    for event in fixture_data["events"]:
        eid = event.get("id", "<missing-id>")
        assert EVENT_REQUIRED <= set(event), (
            f"{eid} missing event fields: {EVENT_REQUIRED - set(event)}"
        )
        assert isinstance(event["details"], dict), f"{eid}: details must be dict"
        assert TIMESTAMP_RE.match(event["created_at"]), (
            f"{eid}: created_at must be fixed UTC literal, got {event['created_at']!r}"
        )

    case_ids = [c["id"] for c in fixture_data["cases"]]
    doc_ids = [d["id"] for d in fixture_data["documents"]]
    event_ids = [e["id"] for e in fixture_data["events"]]
    assert len(case_ids) == len(set(case_ids)), "duplicate case ids"
    assert len(doc_ids) == len(set(doc_ids)), "duplicate document ids"
    assert len(event_ids) == len(set(event_ids)), "duplicate event ids"
    all_ids = case_ids + doc_ids + event_ids
    assert len(all_ids) == len(set(all_ids)), "ids must be unique across case/doc/event domains"

    case_set = set(case_ids)
    for doc in fixture_data["documents"]:
        assert doc["case_id"] in case_set, (
            f"document {doc['id']} references unknown case_id {doc['case_id']}"
        )
    for event in fixture_data["events"]:
        assert event["case_id"] in case_set, (
            f"event {event['id']} references unknown case_id {event['case_id']}"
        )

    docs_by_case: Counter[str] = Counter(d["case_id"] for d in fixture_data["documents"])
    events_by_case: Counter[str] = Counter(e["case_id"] for e in fixture_data["events"])
    for case in fixture_data["cases"]:
        cid = case["id"]
        assert docs_by_case[cid] >= 1 and events_by_case[cid] >= 1, (
            f"{cid} must span case + documents + events (docs={docs_by_case[cid]}, "
            f"events={events_by_case[cid]})"
        )


def test_ground_truth_family_split_and_label_invariants(
    fixture_data: dict[str, Any], ground_truth: dict[str, Any]
) -> None:
    """Labels cover every case; splits are disjoint/complete with full family coverage."""
    assert "split" in ground_truth and "cases" in ground_truth, (
        "ground truth must define split and cases"
    )
    split = ground_truth["split"]
    dev = split["development"]
    eva = split["evaluation"]
    assert isinstance(dev, list) and isinstance(eva, list), "split sides must be lists"
    assert set(dev) & set(eva) == set(), (
        f"development/evaluation splits must be disjoint; overlap={sorted(set(dev) & set(eva))}"
    )
    case_ids = {c["id"] for c in fixture_data["cases"]}
    assert set(dev) | set(eva) == case_ids, (
        "split union must equal fixture case ids; "
        f"missing={sorted(case_ids - (set(dev) | set(eva)))} "
        f"extra={sorted((set(dev) | set(eva)) - case_ids)}"
    )
    assert set(ground_truth["cases"]) == case_ids, (
        "ground_truth.cases keys must match fixture case ids"
    )

    record_ids = (
        case_ids
        | {d["id"] for d in fixture_data["documents"]}
        | {e["id"] for e in fixture_data["events"]}
    )
    primary: list[str] = []
    for cid, label in ground_truth["cases"].items():
        assert GT_CASE_REQUIRED <= set(label), (
            f"{cid} missing label fields: {GT_CASE_REQUIRED - set(label)}"
        )
        assert set(label["expected_categories"]) <= ALLOWED_CATEGORIES, (
            f"{cid}: unknown categories {set(label['expected_categories']) - ALLOWED_CATEGORIES}"
        )
        assert label["expected_categories"], f"{cid}: expected_categories must be non-empty"
        assert set(label["acceptable_actions"]) <= ALLOWED_ACTIONS, (
            f"{cid}: unknown actions {set(label['acceptable_actions']) - ALLOWED_ACTIONS}"
        )
        assert label["acceptable_actions"], f"{cid}: acceptable_actions must be non-empty"
        assert isinstance(label["require_uncertainty"], bool), (
            f"{cid}: require_uncertainty must be bool"
        )
        is_ambiguous = "indeterminate" in label["expected_categories"]
        assert label["require_uncertainty"] is is_ambiguous, (
            f"{cid}: require_uncertainty must match indeterminate membership "
            f"(got {label['require_uncertainty']}, ambiguous={is_ambiguous})"
        )
        alts = label["required_evidence"]["alternatives"]
        assert isinstance(alts, list) and alts, f"{cid}: required_evidence.alternatives required"
        for alt in alts:
            assert isinstance(alt, list) and alt, (
                f"{cid}: each evidence alternative must be non-empty"
            )
            assert set(alt) <= record_ids, (
                f"{cid} evidence refs unknown ids: {set(alt) - record_ids}"
            )
        primary.append(label["expected_categories"][0])

    counts = Counter(primary)
    expected_dist = {
        "missing_document": 5,
        "data_mismatch": 5,
        "processing_failure": 5,
        "indeterminate": 3,
        "no_issue": 2,
    }
    for cat, n in expected_dist.items():
        assert counts[cat] == n, f"primary category {cat}: expected {n}, got {counts[cat]}"

    controls = [
        cid
        for cid, label in ground_truth["cases"].items()
        if label["expected_categories"] == ["no_issue"]
    ]
    assert len(controls) == 2, f"expected 2 no_issue controls, got {controls}"
    for cid in controls:
        assert ground_truth["cases"][cid]["acceptable_actions"] == ["no_action"], (
            f"{cid}: no_issue control must accept only no_action"
        )

    def families(case_id_list: list[str]) -> set[str]:
        return {ground_truth["cases"][cid]["expected_categories"][0] for cid in case_id_list}

    dev_f = families(dev)
    eva_f = families(eva)
    assert ALLOWED_CATEGORIES <= dev_f, (
        f"development split missing families: {ALLOWED_CATEGORIES - dev_f}"
    )
    assert ALLOWED_CATEGORIES <= eva_f, (
        f"evaluation split missing families: {ALLOWED_CATEGORIES - eva_f}"
    )


def test_prompt_injection_document_present(fixture_data: dict[str, Any]) -> None:
    injected = [
        d
        for d in fixture_data["documents"]
        if INJECTION_HINT.search(d["content"]) or INJECTION_HINT.search(json.dumps(d["fields"]))
    ]
    assert len(injected) >= 1, "fixture must include at least one prompt-injection document"
    # Injection lives only in visible content, not as a labeled instruction field
    for doc in injected:
        blob = json.dumps(doc).lower()
        assert "expected_categories" not in blob, f"{doc['id']}: injection doc leaks label key"
        assert "ground_truth" not in blob, f"{doc['id']}: injection doc leaks ground_truth key"


def test_contradictory_signals_exist(fixture_data: dict[str, Any]) -> None:
    # CASE-016: vendor complete vs missing labs
    docs_016 = [d for d in fixture_data["documents"] if d["case_id"] == "CASE-016"]
    events_016 = [e for e in fixture_data["events"] if e["case_id"] == "CASE-016"]
    assert any(
        d["document_type"] == "lab_results" and d["status"] == "missing" for d in docs_016
    ), "CASE-016 must include missing lab_results document"
    assert any(
        e["event_type"] == "exam_vendor_status" and e["status"] == "complete" for e in events_016
    ), "CASE-016 must include complete exam_vendor_status event"

    # CASE-018: success and error for same engine step
    events_018 = [e for e in fixture_data["events"] if e["case_id"] == "CASE-018"]
    statuses = {
        (e["event_type"], e["status"]) for e in events_018 if e["event_type"] == "rules_engine"
    }
    assert ("rules_engine", "passed") in statuses, "CASE-018 missing rules_engine passed signal"
    assert ("rules_engine", "error") in statuses, "CASE-018 missing rules_engine error signal"


def _walk(obj: Any):
    if isinstance(obj, dict):
        for key, value in obj.items():
            yield "key", str(key)
            yield from _walk(value)
    elif isinstance(obj, list):
        for item in obj:
            yield from _walk(item)
    elif isinstance(obj, str):
        yield "str", obj
    else:
        yield "other", obj


def test_ground_truth_does_not_leak_into_fixture(fixture_data: dict[str, Any]) -> None:
    # Structural answer-key fields must not appear as fixture keys.
    for kind, value in _walk(fixture_data):
        if kind == "key":
            assert LEAK_KEY_RE.search(value) is None, f"fixture key leaks evaluator field: {value}"
            assert value not in LABEL_TOKENS, f"fixture key is label token: {value}"
    # Non-content scalar fields also must not carry label tokens.
    for case in fixture_data["cases"]:
        for field in ("id", "status", "symptom"):
            assert case[field] not in LABEL_TOKENS, (
                f"case {case.get('id')} field {field} is label token {case[field]!r}"
            )
        for req in case["required_documents"]:
            assert req not in LABEL_TOKENS, (
                f"case {case['id']} required_documents entry is label token {req!r}"
            )
    for doc in fixture_data["documents"]:
        for field in ("id", "case_id", "document_type", "status"):
            assert doc[field] not in LABEL_TOKENS, (
                f"doc {doc.get('id')} field {field} is label token {doc[field]!r}"
            )
    for event in fixture_data["events"]:
        for field in ("id", "case_id", "event_type", "status", "created_at"):
            assert event[field] not in LABEL_TOKENS, (
                f"event {event.get('id')} field {field} is label token {event[field]!r}"
            )
