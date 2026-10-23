import { useEffect, useState } from "react";
import { ApiError, getCases } from "@/api";
import type { CaseListItem } from "@/types";
import { Badge } from "@/components/ui/badge";
import { Card, CardContent } from "@/components/ui/card";

export interface CaseListProps {
  onOpen: (caseId: string) => void;
}

export function CaseList({ onOpen }: CaseListProps) {
  const [cases, setCases] = useState<CaseListItem[] | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setError(null);
    getCases()
      .then((data) => {
        if (!cancelled) setCases(data);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        const msg =
          err instanceof ApiError
            ? err.status
              ? `Request error (${err.status}): ${err.message}`
              : `Network error: ${err.message}`
            : "Failed to load cases";
        setError(msg);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) {
    return <p className="text-sm text-slate-500">Loading cases…</p>;
  }

  if (error) {
    return (
      <p className="rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">
        {error}
      </p>
    );
  }

  if (!cases?.length) {
    return <p className="text-sm text-slate-500">No cases found.</p>;
  }

  return (
    <section className="flex flex-col gap-4">
      <div className="space-y-1">
        <h2 className="text-sm font-semibold text-slate-900">Cases</h2>
        <p className="text-sm text-slate-500">
          Select a case to investigate. Status describes the insurance workflow.
        </p>
      </div>

      <ul className="flex flex-col gap-2.5">
        {cases.map((c) => (
          <li key={c.case_id}>
            <button
              type="button"
              className="group w-full cursor-pointer text-left"
              onClick={() => onOpen(c.case_id)}
            >
              <Card className="rounded-xl border-slate-200 bg-white shadow-none transition-colors group-hover:border-slate-300 group-hover:bg-slate-50/80">
                <CardContent className="flex flex-col gap-2 p-3.5 sm:flex-row sm:items-center sm:justify-between sm:gap-4">
                  <div className="min-w-0 flex-1 space-y-1">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-mono text-sm font-semibold text-slate-900">
                        {c.case_id}
                      </span>
                      <Badge
                        variant="outline"
                        className="rounded-full border-slate-200 bg-slate-50 px-2 py-0 text-[11px] font-medium capitalize text-slate-600"
                      >
                        {c.status}
                      </Badge>
                      {c.pending_review ? (
                        <Badge
                          variant="warning"
                          className="rounded-full px-2 py-0 text-[11px]"
                        >
                          Pending review
                        </Badge>
                      ) : null}
                      {c.needs_execution ? (
                        <Badge
                          variant="danger"
                          className="rounded-full px-2 py-0 text-[11px]"
                        >
                          Needs execution
                        </Badge>
                      ) : null}
                    </div>
                    <p className="text-sm leading-snug text-slate-500">
                      {c.symptom}
                    </p>
                  </div>
                  <span className="shrink-0 text-sm font-medium text-slate-500 group-hover:text-slate-800">
                    Open case →
                  </span>
                </CardContent>
              </Card>
            </button>
          </li>
        ))}
      </ul>
    </section>
  );
}
