import type {
  CaseListItem,
  CaseSummary,
  ConversationHistoryResponse,
  ConversationTurnResponse,
  InvestigationResponse,
  Proposal,
  ProposalDecisionResponse,
  RetryTurnRequest,
  RunRecord,
  RunSummary,
  SendMessageRequest,
} from "./types";

/** CLI default objective — not exposed as an editable UI field. */
const DEFAULT_OBJECTIVE =
  "Investigate why this case is stalled and recommend the next action.";

const API_PREFIX = "/api";

/** Non-simple header: forces CORS preflight; required by Vite proxy on decisions. */
const OPERATOR_CSRF_HEADER = "X-Casepilot-Operator";
const OPERATOR_CSRF_VALUE = "1";

export class ApiError extends Error {
  readonly status: number | undefined;
  readonly body: unknown;

  constructor(message: string, status?: number, body?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.body = body;
  }
}

function detailMessage(body: unknown, fallback: string): string {
  if (body && typeof body === "object" && "detail" in body) {
    const d = (body as { detail: unknown }).detail;
    if (typeof d === "string") return d;
    try {
      return JSON.stringify(d);
    } catch {
      /* ignore */
    }
  }
  return fallback;
}

/** Single-shot fetch. No automatic retries. Distinguishes network vs HTTP errors. */
export async function request<T>(
  path: string,
  options?: RequestInit,
): Promise<T> {
  const url = path.startsWith("http")
    ? path
    : `${API_PREFIX}${path.startsWith("/") ? path : `/${path}`}`;

  let res: Response;
  try {
    res = await fetch(url, {
      ...options,
      headers: {
        Accept: "application/json",
        ...(options?.body ? { "Content-Type": "application/json" } : {}),
        ...options?.headers,
      },
    });
  } catch (err) {
    const msg = err instanceof Error ? err.message : "network error";
    throw new ApiError(`Request failed: ${msg}`);
  }

  const text = await res.text();
  let body: unknown = undefined;
  if (text) {
    try {
      body = JSON.parse(text) as unknown;
    } catch {
      body = text;
    }
  }

  if (!res.ok) {
    throw new ApiError(
      detailMessage(body, `HTTP ${res.status}`),
      res.status,
      body,
    );
  }

  return body as T;
}

export function getCases(): Promise<CaseListItem[]> {
  return request<CaseListItem[]>("/v1/cases");
}

export function getCase(id: string): Promise<CaseSummary> {
  return request<CaseSummary>(`/v1/cases/${encodeURIComponent(id)}`);
}

export function getHistory(id: string): Promise<RunSummary[]> {
  return request<RunSummary[]>(
    `/v1/cases/${encodeURIComponent(id)}/runs`,
  );
}

export function getRun(id: string): Promise<RunRecord> {
  return request<RunRecord>(`/v1/investigations/${encodeURIComponent(id)}`);
}

export function getProposal(id: string): Promise<Proposal> {
  return request<Proposal>(`/v1/proposals/${encodeURIComponent(id)}`);
}

export function getCaseProposals(id: string): Promise<Proposal[]> {
  return request<Proposal[]>(
    `/v1/cases/${encodeURIComponent(id)}/proposals`,
  );
}

export function investigate(id: string): Promise<InvestigationResponse> {
  return request<InvestigationResponse>("/v1/investigations", {
    method: "POST",
    body: JSON.stringify({ case_id: id, objective: DEFAULT_OBJECTIVE }),
  });
}

export function approve(id: string): Promise<ProposalDecisionResponse> {
  return request<ProposalDecisionResponse>(
    `/v1/proposals/${encodeURIComponent(id)}/approve`,
    {
      method: "POST",
      body: JSON.stringify({}),
      headers: { [OPERATOR_CSRF_HEADER]: OPERATOR_CSRF_VALUE },
    },
  );
}

export function reject(
  id: string,
  reason?: string,
): Promise<ProposalDecisionResponse> {
  return request<ProposalDecisionResponse>(
    `/v1/proposals/${encodeURIComponent(id)}/reject`,
    {
      method: "POST",
      body: JSON.stringify(reason ? { reason } : {}),
      headers: { [OPERATOR_CSRF_HEADER]: OPERATOR_CSRF_VALUE },
    },
  );
}

// ---------------------------------------------------------------------------
// Run-scoped conversation (BackendRoutes.CONVERSATION*)
// ---------------------------------------------------------------------------

/** GET /v1/investigations/{run_id}/conversation */
export function getConversation(
  runId: string,
): Promise<ConversationHistoryResponse> {
  return request<ConversationHistoryResponse>(
    `/v1/investigations/${encodeURIComponent(runId)}/conversation`,
  );
}

/** POST /v1/investigations/{run_id}/conversation/messages */
export function sendMessage(
  runId: string,
  body: SendMessageRequest,
): Promise<ConversationTurnResponse> {
  return request<ConversationTurnResponse>(
    `/v1/investigations/${encodeURIComponent(runId)}/conversation/messages`,
    { method: "POST", body: JSON.stringify(body) },
  );
}

/** POST /v1/investigations/{run_id}/conversation/turns/{turn_id}/retry */
export function retryTurn(
  runId: string,
  turnId: string,
  body: RetryTurnRequest,
): Promise<ConversationTurnResponse> {
  return request<ConversationTurnResponse>(
    `/v1/investigations/${encodeURIComponent(runId)}/conversation/turns/${encodeURIComponent(turnId)}/retry`,
    { method: "POST", body: JSON.stringify(body) },
  );
}

/** Stable client idempotency key for one explicit send/retry action. */
export function newIdempotencyKey(): string {
  return crypto.randomUUID();
}
