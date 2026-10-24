import { useCallback, useEffect, useRef, useState, type FormEvent } from "react";
import {
  ApiError,
  getConversation,
  newIdempotencyKey,
  retryTurn,
  sendMessage,
} from "@/api";
import type {
  ConversationAttempt,
  ConversationTurn,
  FollowUpAnswer,
  RenderedSourceFact,
  TurnStatus,
} from "@/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";

export type ChatPanelProps = {
  runId: string;
  /** Parent lock (e.g. investigation in flight). */
  disabled?: boolean;
};

function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

function isAmbiguousTransport(e: unknown): boolean {
  // No HTTP status → network/abort. Do not treat as model failure.
  return e instanceof ApiError ? e.status === undefined : true;
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined) return String(value);
  if (typeof value === "string") return value;
  try {
    return JSON.stringify(value);
  } catch {
    return String(value);
  }
}

/** Latest attempt is canonical current attempt (attempts ordered oldest→newest). */
function currentAttempt(turn: ConversationTurn): ConversationAttempt | null {
  const list = turn.attempts ?? [];
  return list.length ? list[list.length - 1]! : null;
}

function statusBadge(
  status: TurnStatus,
): "warning" | "success" | "danger" | "secondary" {
  if (status === "pending") return "warning";
  if (status === "succeeded") return "success";
  if (status === "failed") return "danger";
  return "secondary";
}

function SourceFactView({ fact }: { fact: RenderedSourceFact }) {
  const provenance =
    fact.provenance === "follow_up" ? "New retrieval" : "Original investigation";
  return (
    <li className="space-y-0.5 border-t border-slate-100 py-2 first:border-t-0 first:pt-0">
      <p className="text-sm leading-relaxed text-slate-800 break-words">
        <span className="font-medium text-slate-900">{fact.field_path}: </span>
        {formatValue(fact.value)}
      </p>
      <p className="font-mono text-[11px] leading-snug text-slate-500 break-all">
        {fact.source_system} / {fact.record_id}
        {fact.retrieval_ref ? ` (${fact.retrieval_ref})` : ""}
      </p>
      <p className="text-[11px] text-slate-500">
        {provenance} · {fact.retrieved_at}
      </p>
    </li>
  );
}

function AnswerView({ answer }: { answer: FollowUpAnswer }) {
  const byLocal = new Map(answer.source_facts.map((f) => [f.local_id, f]));
  return (
    <div className="space-y-3">
      {answer.notes?.length
        ? answer.notes.map((note, i) => (
            <p
              key={`note-${i}`}
              className="text-sm leading-relaxed text-slate-700 whitespace-pre-wrap break-words"
            >
              {note}
            </p>
          ))
        : null}

      {answer.source_facts?.length ? (
        <div className="space-y-1">
          <p className="text-[11px] font-medium uppercase tracking-wide text-slate-500">
            Source facts
          </p>
          <ul>
            {answer.source_facts.map((f) => (
              <SourceFactView key={f.local_id || f.snapshot_id} fact={f} />
            ))}
          </ul>
        </div>
      ) : null}

      {answer.interpretations?.length ? (
        <div className="space-y-2">
          <p className="text-[11px] font-medium uppercase tracking-wide text-slate-500">
            Interpretations
          </p>
          {answer.interpretations.map((interp, i) => (
            <div
              key={`${interp.kind}-${i}`}
              className="space-y-1 border-l-2 border-slate-300 pl-3"
            >
              <p className="text-[11px] font-medium uppercase tracking-wide text-slate-500">
                {interp.kind}
              </p>
              <p className="text-sm leading-relaxed text-slate-800 whitespace-pre-wrap break-words">
                {interp.text}
              </p>
              {interp.cites?.length ? (
                <p className="text-[11px] text-slate-500">
                  Cites:{" "}
                  {interp.cites
                    .map((id) => {
                      const f = byLocal.get(id);
                      return f ? f.field_path : id;
                    })
                    .join(", ")}
                </p>
              ) : null}
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function TurnView({
  turn,
  onRetry,
  retryDisabled,
}: {
  turn: ConversationTurn;
  onRetry: (turn: ConversationTurn, attempt: ConversationAttempt) => void;
  retryDisabled: boolean;
}) {
  const attempt = currentAttempt(turn);
  const answer =
    turn.status === "succeeded"
      ? attempt?.answer ??
        [...(turn.attempts ?? [])].reverse().find((a) => a.status === "succeeded")
          ?.answer ??
        null
      : null;
  const failError =
    turn.status === "failed"
      ? attempt?.error?.trim() || "This attempt failed."
      : null;

  return (
    <article
      className="space-y-2 border-t border-slate-100 py-4 first:border-t-0 first:pt-0"
      aria-label={`Turn ${turn.status}`}
    >
      <div className="flex flex-wrap items-start justify-between gap-2">
        <div className="min-w-0 space-y-1">
          <p className="text-[11px] font-medium uppercase tracking-wide text-slate-500">
            Operator
          </p>
          <p className="text-sm leading-relaxed text-slate-900 whitespace-pre-wrap break-words">
            {turn.question}
          </p>
        </div>
        <Badge variant={statusBadge(turn.status)} className="shrink-0 capitalize">
          {turn.status}
        </Badge>
      </div>

      {turn.status === "pending" ? (
        <p className="text-sm text-slate-500" role="status">
          Waiting for answer…
        </p>
      ) : null}

      {turn.status === "failed" && attempt ? (
        <div className="space-y-2">
          <p
            role="alert"
            className="border-l-2 border-red-500 pl-3 text-sm leading-relaxed text-red-800"
          >
            {failError}
          </p>
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={retryDisabled}
            onClick={() => onRetry(turn, attempt)}
          >
            Retry
          </Button>
        </div>
      ) : null}

      {turn.status === "succeeded" && answer ? (
        <div className="space-y-1">
          <p className="text-[11px] font-medium uppercase tracking-wide text-slate-500">
            Agent
          </p>
          <AnswerView answer={answer} />
        </div>
      ) : null}

      <p className="text-[11px] text-slate-400">
        {turn.created_at}
        {turn.updated_at && turn.updated_at !== turn.created_at
          ? ` · updated ${turn.updated_at}`
          : ""}
      </p>
    </article>
  );
}

/**
 * Non-streaming, run-scoped follow-up chat.
 * Eligible runs only (parent gates COMPLETED/INDETERMINATE). One conversation per run.
 */
export function ChatPanel({ runId, disabled = false }: ChatPanelProps) {
  const [turns, setTurns] = useState<ConversationTurn[]>([]);
  const [loading, setLoading] = useState(true);
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const mounted = useRef(true);
  const inFlight = useRef(false);
  const loadGen = useRef(0);
  // Sticky keys: one explicit action stays idempotent across ambiguous transport re-clicks.
  const pendingSend = useRef<{ question: string; key: string } | null>(null);
  const pendingRetry = useRef<{
    turnId: string;
    attemptId: string;
    key: string;
  } | null>(null);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const applyHistory = useCallback((list: ConversationTurn[]) => {
    setTurns(list);
  }, []);

  const loadHistory = useCallback(async (): Promise<"ok" | "failed" | "aborted"> => {
    const gen = ++loadGen.current;
    setLoading(true);
    try {
      const hist = await getConversation(runId);
      if (gen !== loadGen.current) return "aborted";
      applyHistory(hist.turns ?? []);
      return "ok";
    } catch {
      if (gen !== loadGen.current) return "aborted";
      return "failed";
    } finally {
      if (gen === loadGen.current && mounted.current) setLoading(false);
    }
  }, [runId, applyHistory]);

  useEffect(() => {
    pendingSend.current = null;
    pendingRetry.current = null;
    setQuestion("");
    setError(null);
    setTurns([]);
    void (async () => {
      const outcome = await loadHistory();
      if (!mounted.current) return;
      if (outcome === "failed") {
        setError("Failed to load conversation for this run.");
      }
    })();
  }, [runId, loadHistory]);

  const hasPending = turns.some((t) => t.status === "pending");
  const controlsLocked = disabled || busy || hasPending || loading;

  /**
   * After send/retry: always reconcile with GET conversation.
   * Lost response ≠ failed model attempt.
   */
  async function mutateAndReconcile(
    action: () => Promise<{ turn: ConversationTurn }>,
    clearStickyKey: () => void,
  ) {
    if (disabled || inFlight.current) return;
    inFlight.current = true;
    setBusy(true);
    setError(null);

    let actionErr: unknown = null;
    let returned: ConversationTurn | null = null;
    try {
      const resp = await action();
      returned = resp.turn;
    } catch (e) {
      actionErr = e;
    }

    let refreshFailed = false;
    try {
      const hist = await getConversation(runId);
      if (mounted.current) applyHistory(hist.turns ?? []);
      if (!actionErr) clearStickyKey();
      else if (!isAmbiguousTransport(actionErr)) clearStickyKey();
    } catch {
      refreshFailed = true;
      if (returned && mounted.current) {
        setTurns((prev) => {
          const idx = prev.findIndex((t) => t.turn_id === returned!.turn_id);
          if (idx >= 0) {
            const next = prev.slice();
            next[idx] = returned!;
            return next;
          }
          return [...prev, returned!];
        });
      }
    }

    if (!mounted.current) return;

    if (actionErr) {
      const msg = errMsg(actionErr);
      if (refreshFailed) {
        setError(
          isAmbiguousTransport(actionErr)
            ? `Request outcome ambiguous (${msg}); refresh also failed. Reload before retrying.`
            : `Request failed (${msg}); refresh failed — showing last known state.`,
        );
      } else if (isAmbiguousTransport(actionErr)) {
        setError(
          `Request outcome ambiguous (${msg}). Showing persisted server state — retry only if the turn is still failed.`,
        );
      } else {
        setError(msg);
      }
    } else if (refreshFailed) {
      setError("Message accepted but history refresh failed; showing returned turn.");
      clearStickyKey();
    }

    inFlight.current = false;
    if (mounted.current) setBusy(false);
  }

  function onSend(e: FormEvent) {
    e.preventDefault();
    const q = question.trim();
    if (!q || controlsLocked) return;
    if (!pendingSend.current || pendingSend.current.question !== q) {
      pendingSend.current = { question: q, key: newIdempotencyKey() };
    }
    const key = pendingSend.current.key;
    void mutateAndReconcile(
      () => sendMessage(runId, { question: q, idempotency_key: key }),
      () => {
        pendingSend.current = null;
        setQuestion("");
      },
    );
  }

  function onRetry(turn: ConversationTurn, attempt: ConversationAttempt) {
    if (controlsLocked || turn.status !== "failed") return;
    const same =
      pendingRetry.current &&
      pendingRetry.current.turnId === turn.turn_id &&
      pendingRetry.current.attemptId === attempt.attempt_id;
    if (!same) {
      pendingRetry.current = {
        turnId: turn.turn_id,
        attemptId: attempt.attempt_id,
        key: newIdempotencyKey(),
      };
    }
    const key = pendingRetry.current!.key;
    void mutateAndReconcile(
      () =>
        retryTurn(runId, turn.turn_id, {
          idempotency_key: key,
          expected_previous_attempt_id: attempt.attempt_id,
        }),
      () => {
        pendingRetry.current = null;
      },
    );
  }

  return (
    <Card aria-label="Follow-up conversation" className="border-slate-200 bg-white">
      <CardHeader className="gap-2 p-4 pb-3">
        <div className="space-y-1">
          <CardTitle className="text-base font-semibold text-slate-900">
            Follow-up
          </CardTitle>
          <CardDescription className="text-xs text-slate-500">
            Run-scoped conversation · read-only investigation chat · no task creation
            here
          </CardDescription>
        </div>
      </CardHeader>

      <CardContent className="space-y-4 p-4 pt-0">
        <p
          role="note"
          className="border-l-2 border-slate-300 pl-3 text-xs leading-relaxed text-slate-600"
        >
          Older messages may fall outside the model context window. Prior chat claims
          are not evidence by themselves.
        </p>

        {error ? (
          <p
            role="alert"
            className="border-l-2 border-amber-500 pl-3 text-sm leading-relaxed text-amber-950"
          >
            {error}
          </p>
        ) : null}

        {loading && !turns.length ? (
          <p className="text-sm text-slate-500">Loading conversation…</p>
        ) : null}

        {!loading && !turns.length ? (
          <p className="text-sm text-slate-500">No messages yet for this run.</p>
        ) : null}

        {turns.length ? (
          <div className="min-w-0">
            {turns.map((t) => (
              <TurnView
                key={t.turn_id}
                turn={t}
                onRetry={onRetry}
                retryDisabled={controlsLocked}
              />
            ))}
          </div>
        ) : null}

        <form onSubmit={onSend} className="space-y-2 border-t border-slate-100 pt-4">
          <label className="block space-y-1.5">
            <span className="text-xs font-medium text-slate-600">Question</span>
            <textarea
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              disabled={controlsLocked}
              rows={3}
              maxLength={4000}
              placeholder="Ask about this investigation’s findings or evidence…"
              className="w-full resize-y rounded-md border border-slate-200 bg-white px-3 py-2 text-sm text-slate-900 placeholder:text-slate-400 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-slate-400 disabled:cursor-not-allowed disabled:opacity-50"
            />
          </label>
          <div className="flex flex-wrap items-center gap-2">
            <Button type="submit" disabled={controlsLocked || !question.trim()}>
              {busy || hasPending ? "Sending…" : "Send"}
            </Button>
            {hasPending ? (
              <span className="text-xs text-slate-500" role="status">
                A turn is in flight — composer locked.
              </span>
            ) : null}
          </div>
        </form>
      </CardContent>
    </Card>
  );
}

export default ChatPanel;
