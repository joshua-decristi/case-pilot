import { useRef, useState, type FormEvent } from "react";
import { approve, reject } from "../api";
import type { Proposal, ProposalStatus } from "../types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export type ProposalPanelProps = {
  proposal: Proposal;
  onRefresh: () => Promise<void>;
  /** Parent lock (e.g. investigation in flight). Combined with local in-flight. */
  disabled?: boolean;
};

type ActionError =
  | { kind: "request"; message: string }
  | { kind: "refresh"; message: string };

function statusBadgeVariant(
  status: ProposalStatus,
): "default" | "secondary" | "outline" | "warning" | "danger" | "success" {
  switch (status) {
    case "pending":
      return "warning";
    case "approved":
      return "warning";
    case "execution_failed":
      return "danger";
    case "executed":
      return "success";
    case "rejected":
      return "secondary";
    default:
      return "outline";
  }
}

function statusLabel(status: ProposalStatus): string {
  switch (status) {
    case "pending":
      return "Pending review";
    case "approved":
      return "Approved — unfinished";
    case "execution_failed":
      return "Execution failed";
    case "executed":
      return "Executed";
    case "rejected":
      return "Rejected";
    default:
      return status;
  }
}

/**
 * Review/execute one persisted proposal.
 * pending → approve&execute or reject; approved/execution_failed → retry; terminal → read-only.
 */
export function ProposalPanel({
  proposal,
  onRefresh,
  disabled = false,
}: ProposalPanelProps) {
  const inFlight = useRef(false);
  const [busy, setBusy] = useState(false);
  const [reason, setReason] = useState("");
  const [actionError, setActionError] = useState<ActionError | null>(null);

  const { status, task_payload: task } = proposal;
  const isPending = status === "pending";
  const isRetryable = status === "approved" || status === "execution_failed";
  const controlsLocked = disabled || busy;

  async function runDecision(fn: () => Promise<{ proposal: Proposal }>) {
    if (disabled || inFlight.current) return;
    inFlight.current = true;
    setBusy(true);
    setActionError(null);

    let requestFailed: string | null = null;
    try {
      await fn();
    } catch (err) {
      requestFailed =
        err instanceof Error
          ? err.message
          : "Request failed; outcome ambiguous until refresh.";
    }

    try {
      await onRefresh();
    } catch {
      // Refresh failure → unknown persisted state; do not auto-retry the action.
      setActionError({
        kind: "refresh",
        message: requestFailed
          ? `Request may have failed (${requestFailed}). Persisted state unknown — refresh failed.`
          : "Action may have completed, but refresh failed; persisted state unknown.",
      });
      inFlight.current = false;
      setBusy(false);
      return;
    }

    if (requestFailed) {
      // Network/request error is ambiguous; do not label as confirmed execution_failed.
      setActionError({
        kind: "request",
        message: `Request/network error (outcome ambiguous): ${requestFailed}. Showing refreshed persisted state.`,
      });
    }

    inFlight.current = false;
    setBusy(false);
  }

  function onApprove() {
    void runDecision(() => approve(proposal.proposal_id));
  }

  function onReject(e: FormEvent) {
    e.preventDefault();
    const trimmed = reason.trim().slice(0, 1000);
    void runDecision(() =>
      reject(proposal.proposal_id, trimmed.length > 0 ? trimmed : undefined),
    );
  }

  return (
    <Card aria-label="Proposal review" className="border-slate-200 bg-white">
      <CardHeader className="gap-2 p-4 pb-3 sm:flex-row sm:items-start sm:justify-between">
        <div className="space-y-1">
          <CardTitle className="text-base font-semibold text-slate-900">
            Proposal review
          </CardTitle>
          <CardDescription className="text-xs text-slate-500">
            ID <code className="font-mono">{proposal.proposal_id}</code>
            {" · "}
            Case{" "}
            <code className="font-mono">{task.case_id || proposal.case_id}</code>
          </CardDescription>
        </div>
        <Badge variant={statusBadgeVariant(status)} className="shrink-0">
          {statusLabel(status)}
        </Badge>
      </CardHeader>

      <CardContent className="space-y-4 p-4 pt-0">
        {/* Exact persisted task — plain text, no nested surface */}
        <div className="space-y-2 border-t border-slate-100 pt-4">
          <p className="text-xs font-medium uppercase tracking-wide text-slate-500">
            Task
          </p>
          <h4 className="text-base font-semibold text-slate-900">{task.title}</h4>
          <pre className="whitespace-pre-wrap break-words font-sans text-sm leading-relaxed text-slate-700">
            {task.body}
          </pre>
          <p className="text-xs text-slate-500">
            Case{" "}
            <code className="font-mono">{task.case_id || proposal.case_id}</code>
          </p>
        </div>

        {/* Outcomes/errors: left rule only — no nested boxes */}
        {status === "rejected" && proposal.execution_error ? (
          <p
            role="status"
            className="border-l-2 border-slate-300 pl-3 text-sm leading-relaxed text-slate-700"
          >
            <span className="font-medium text-slate-900">Rejection reason: </span>
            {proposal.execution_error}
          </p>
        ) : null}

        {status === "execution_failed" ? (
          <p
            role="alert"
            className="border-l-2 border-red-500 pl-3 text-sm leading-relaxed text-red-800"
          >
            Confirmed execution failure
            {proposal.execution_error ? `: ${proposal.execution_error}` : "."}
          </p>
        ) : null}

        {status === "approved" ? (
          <p
            role="status"
            className="border-l-2 border-amber-500 pl-3 text-sm leading-relaxed text-amber-900"
          >
            Approved — execution unfinished. Retry to continue.
          </p>
        ) : null}

        {status === "executed" ? (
          <div
            role="status"
            className="space-y-2 border-l-2 border-emerald-500 pl-3 text-sm leading-relaxed text-emerald-900"
          >
            <p className="font-medium text-emerald-900">Executed</p>
            {proposal.task_id ? (
              <dl className="space-y-1 text-emerald-900">
                <div>
                  <dt className="inline text-xs font-medium uppercase tracking-wide text-emerald-800/80">
                    Task ID{" "}
                  </dt>
                  <dd className="inline font-mono text-sm">{proposal.task_id}</dd>
                </div>
                <div>
                  <dt className="inline text-xs font-medium uppercase tracking-wide text-emerald-800/80">
                    Title{" "}
                  </dt>
                  <dd className="inline text-sm">{task.title}</dd>
                </div>
                <div>
                  <dt className="text-xs font-medium uppercase tracking-wide text-emerald-800/80">
                    Body
                  </dt>
                  <dd className="mt-0.5 whitespace-pre-wrap break-words font-sans text-sm">
                    {task.body}
                  </dd>
                </div>
                <div>
                  <dt className="inline text-xs font-medium uppercase tracking-wide text-emerald-800/80">
                    Case{" "}
                  </dt>
                  <dd className="inline font-mono text-sm">
                    {task.case_id || proposal.case_id}
                  </dd>
                </div>
              </dl>
            ) : (
              <p className="text-sm text-emerald-900">
                Executed, but historical task identity is unavailable (no stored
                task ID).
              </p>
            )}
          </div>
        ) : null}

        {actionError ? (
          <p
            role="alert"
            data-error-kind={actionError.kind}
            className={
              actionError.kind === "refresh"
                ? "border-l-2 border-amber-500 pl-3 text-sm leading-relaxed text-amber-950"
                : "border-l-2 border-red-500 pl-3 text-sm leading-relaxed text-red-800"
            }
          >
            {actionError.message}
          </p>
        ) : null}

        {isPending ? (
          <div className="space-y-3 border-t border-slate-100 pt-4">
            <div className="flex flex-wrap gap-2">
              <Button
                type="button"
                variant="default"
                onClick={onApprove}
                disabled={controlsLocked}
              >
                Approve &amp; execute
              </Button>
            </div>
            <form onSubmit={onReject} className="space-y-2">
              <label className="block space-y-1.5">
                <span className="text-xs font-medium text-slate-600">
                  Rejection reason (optional)
                </span>
                <input
                  type="text"
                  value={reason}
                  maxLength={1000}
                  disabled={controlsLocked}
                  onChange={(e) => setReason(e.target.value)}
                  className="w-full rounded-md border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 placeholder:text-slate-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-400 disabled:cursor-not-allowed disabled:opacity-50"
                  placeholder="Optional reason (max 1000)"
                />
              </label>
              <div className="flex flex-wrap gap-2">
                <Button
                  type="submit"
                  variant="outline"
                  disabled={controlsLocked}
                  className="border-red-200 text-red-700 hover:bg-red-50"
                >
                  Reject
                </Button>
              </div>
            </form>
          </div>
        ) : null}

        {isRetryable ? (
          <div className="flex flex-wrap gap-2 border-t border-slate-100 pt-4">
            <Button
              type="button"
              variant="default"
              onClick={onApprove}
              disabled={controlsLocked}
            >
              Retry execution
            </Button>
          </div>
        ) : null}
      </CardContent>
    </Card>
  );
}

export default ProposalPanel;
