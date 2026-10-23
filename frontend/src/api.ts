import type { CaseListItem, CaseSummary } from "./types";

const API_PREFIX = "/api";

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
