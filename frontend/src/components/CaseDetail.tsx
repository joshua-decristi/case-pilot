import { useEffect, useState } from "react";
import { ApiError, getCase } from "@/api";
import type { CaseSummary } from "@/types";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";

export type CaseDetailProps = {
  caseId: string;
  onBack: () => void;
};

/** Brief case summary placeholder — investigation/history/decisions come later. */
export function CaseDetail({ caseId, onBack }: CaseDetailProps) {
  const [summary, setSummary] = useState<CaseSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    getCase(caseId)
      .then((data) => {
        if (!cancelled) setSummary(data);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        const msg =
          err instanceof ApiError
            ? err.status
              ? `Request error (${err.status}): ${err.message}`
              : `Network error: ${err.message}`
            : "Failed to load case";
        setError(msg);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [caseId]);

  return (
    <main className="mx-auto max-w-6xl px-4 py-6 text-[14px] leading-[1.6] text-slate-900">
      <div className="mb-4">
        <Button
          variant="ghost"
          size="sm"
          onClick={onBack}
          className="-ml-2 text-slate-600"
        >
          ← Back to cases
        </Button>
      </div>

      {loading ? (
        <p className="text-sm text-slate-500" aria-live="polite">
          Loading case…
        </p>
      ) : null}

      {error ? (
        <p
          role="alert"
          className="mb-4 rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800"
        >
          {error}
        </p>
      ) : null}

      {summary ? (
        <header className="mb-6">
          <p className="text-[11px] font-medium uppercase tracking-wide text-slate-500">
            Case
          </p>
          <div className="mt-1 flex flex-wrap items-center gap-2.5">
            <h1 className="text-2xl font-semibold tracking-tight text-slate-900">
              {summary.case_id}
            </h1>
            <Badge variant="outline" className="capitalize">
              {summary.status}
            </Badge>
          </div>
          {summary.symptom ? (
            <p className="mt-1.5 text-[14px] leading-relaxed text-slate-600">
              {summary.symptom}
            </p>
          ) : null}
        </header>
      ) : null}
    </main>
  );
}

export default CaseDetail;
