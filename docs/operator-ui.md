# Local operator UI

[Overview](../README.md) · [Quickstart](quickstart.md) · [Architecture](architecture.md)

## Scope

Local-first visual shell over the existing investigation + human-approval workflow. **Not** production-hosted; packaging/deploy are out of scope.

**Startup** (backend + simulator already running via `make dev`):

```bash
npm install --prefix frontend
npm run dev --prefix frontend
```

Open **http://127.0.0.1:5173** (or `http://127.0.0.1:$FRONTEND_PORT` when set). The Vite dev server proxies same-origin `/api` to the backend (`BACKEND_BASE_URL`, default `127.0.0.1:8000`).

**Credentials (local wiring only):**

| Secret                         | Browser? | How                                                                                                                                          |
| ------------------------------ | -------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| `OPERATOR_TOKEN` (root `.env`) | **No**   | Vite proxy attaches it **server-side** to approve/reject requests using **non-`VITE_`** config. Never embed in frontend assets or commit it. |
| `SIMULATOR_SERVICE_TOKEN`      | **No**   | Backend→simulator only. UI must not call the simulator or receive this credential.                                                           |
| `OPENROUTER_API_KEY`           | **No**   | Backend live runs only.                                                                                                                      |

This preserves the existing operator-token gate. It is **not** a login system, multi-user auth, or a claim of production browser security.

**Paid model calls:** Investigate hits the real backend investigation path. Do **not** click Investigate / run live evals without **explicit** permission and a configured key + ledger headroom. UI chrome (list/detail/history of already-persisted data) does not need OpenRouter.

## Screens and workflow (exact scope)

1. **Case list** — seeded cases + status; flags **pending review** vs **approved-but-unfinished / execution failed** separately. No charts, metrics, or global recovery dashboard.
2. **Case detail** — brief summary; **Investigate** with the **default objective only** (no editable objective); latest persisted run; small **history** of earlier runs; unfinished proposals remain visible across runs.
3. **Findings (one page)** — diagnosis/explanation, cited evidence (facts + source/record/field refs), uncertainty, recommended next action, and exact proposed task when present. A result **without** a proposal stays useful (findings only; no approval controls). Evidence lives with findings — **no** document/event/task record browser.
4. **Proposal review** — show the **exact persisted** task payload (title, body, case) + status.
   - **Pending:** **Approve & execute** (immediate, no confirm dialog) or **Reject** (optional inline reason). Disable conflicting controls while a request runs.
   - **Approved unfinished / execution_failed:** show error details; **Retry execution** via the existing approval endpoint + stored payload. **No reject after approval.**
   - **Executed / rejected:** terminal outcome only; show rejection reason or execution error only as applicable — do not mislabel rejection as execution failure.
5. **Loading / errors / history** — synchronous REST; show **Investigating…** until complete (no streaming, tool feed, or fake progress). Block duplicate Investigate while in flight **in the mounted UI only** (not cross-tab/job infra). Inline request failures; keep earlier results if a new run fails. History reopens **persisted** runs/proposals; page reload refetches persisted state. Distinguish network/request errors from confirmed execution failure; refetch before claiming failure. **No** automatic execution or automatic retries.

## Manual verification checklist

Manual only — **no new automated UI tests** in this workstream. Against a running backend + simulator. Historical status after authorized manual pass (**5** live Investigate calls; ledger delta **`$0.00864441`**):

| Item                                                                                                                                              | Status                                                                                                                    |
| ------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| Open case list and select a case without the CLI                                                                                                  | **Verified**                                                                                                              |
| See summary; start default investigation                                                                                                          | **Verified** (5 live calls)                                                                                               |
| Evidence, uncertainty, recommended action, exact proposed task on one page                                                                        | **Verified**                                                                                                              |
| Approve → persisted **executed** outcome                                                                                                          | **Verified**                                                                                                              |
| Reject pending proposal **with** a reason                                                                                                         | **Verified**                                                                                                              |
| Reject pending proposal **without** a reason                                                                                                      | **Verified**                                                                                                              |
| Reopen earlier investigations; earlier **pending** proposal preserved across a later run                                                          | **Verified**                                                                                                              |
| Reload pages; recover persisted history and **current** proposal statuses                                                                         | **Verified**                                                                                                              |
| Local startup; CSRF / no service or operator credentials in frontend assets                                                                       | **Verified**                                                                                                              |
| Honest **Investigating…** (controls disabled); in-mounted-UI duplicate Investigate guard (two clicks → one request via pending `fetch` intercept) | **Verified** (ephemeral browser; **no paid calls**)                                                                       |
| Manual network failure on Investigate → inline error; prior findings preserved; reload clean                                                      | **Verified** (forced failure; **no paid calls**)                                                                          |
| Failed/interrupted execution → explicit **Retry** via existing backend behavior                                                                   | **Unavailable / not verified** — no `approved`-unfinished or `execution_failed` state observed; do **not** claim coverage |

**Automated gates (separate from manual UI checklist):** pytest + Ruff lint/format + mypy + frontend build; historical reviews passed. Use the [current offline commands](quickstart.md#quality), not a fixed test count.

**Overall:** operator-UI spec is **not fully validated** solely because execution recovery/retry was **unavailable** to observe. These observations are historical, not a fresh verification or the current cumulative ledger total. Further paid Investigate/eval calls still require **explicit** authorization.

## Grounded follow-up chat

Eligible completed or indeterminate runs support read-only follow-up Q&A using server-owned evidence snapshots. Turns are persisted; send/retry recovery does not execute tasks. See [grounding](architecture.md#grounding-and-follow-up-chat).
