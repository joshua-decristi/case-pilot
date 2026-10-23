import { useCallback, useEffect, useRef, useState } from 'react';
import {
  getCase,
  getCaseProposals,
  getHistory,
  getRun,
  investigate,
} from '@/api';
import type {
  CaseSummary,
  InvestigationResponse,
  Proposal,
  RunRecord,
  RunSummary,
} from '@/types';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { ProposalPanel } from './ProposalPanel';
import { RunHistory } from './RunHistory';
import { RunResult } from './RunResult';

export type CaseDetailProps = {
  caseId: string;
  onBack: () => void;
};

const ACTIONABLE = new Set(['pending', 'approved', 'execution_failed']);

/** ok = committed; failed = network/HTTP error; aborted = superseded (not an error). */
type LoadRunOutcome = 'ok' | 'failed' | 'aborted';

function errMsg(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}

/** Dedup: selected-run proposals ∪ all pending/approved/execution_failed case proposals. */
function visibleProposals(
  caseProposals: Proposal[],
  selectedRunId: string | null,
): Proposal[] {
  const byId = new Map<string, Proposal>();
  for (const p of caseProposals) {
    if (ACTIONABLE.has(p.status)) byId.set(p.proposal_id, p);
  }
  if (selectedRunId) {
    for (const p of caseProposals) {
      if (p.run_id === selectedRunId) byId.set(p.proposal_id, p);
    }
  }
  return [...byId.values()];
}

/** Build a displayable RunRecord from a successful investigate response. */
function runFromInvestigateResponse(resp: InvestigationResponse): RunRecord {
  return {
    run_id: resp.run_id,
    case_id: resp.result.case_id,
    objective: '',
    status: resp.result.status,
    result: resp.result,
    error: null,
    created_at: new Date().toISOString(),
    completed_at: new Date().toISOString(),
  };
}

export function CaseDetail({ caseId, onBack }: CaseDetailProps) {
  const [summary, setSummary] = useState<CaseSummary | null>(null);
  const [history, setHistory] = useState<RunSummary[]>([]);
  const [caseProposals, setCaseProposals] = useState<Proposal[]>([]);
  // selectedRunId always matches displayed selectedRun (committed only on success).
  const [selectedRunId, setSelectedRunId] = useState<string | null>(null);
  const [selectedRun, setSelectedRun] = useState<RunRecord | null>(null);

  const [pageLoading, setPageLoading] = useState(true);
  const [runLoading, setRunLoading] = useState(false);
  const [investigating, setInvestigating] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const selectGen = useRef(0);
  const investigatingRef = useRef(false);
  const mounted = useRef(true);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
    };
  }, []);

  const refreshProposals = useCallback(async () => {
    const list = await getCaseProposals(caseId);
    if (mounted.current) setCaseProposals(list);
  }, [caseId]);

  /**
   * Fetch persisted run. Commits selection only on success so highlight matches findings.
   * Superseded requests return 'aborted' — callers must not treat that as load failure
   * (StrictMode double-effect and rapid history clicks both bump selectGen).
   */
  const loadRun = useCallback(async (runId: string): Promise<LoadRunOutcome> => {
    const gen = ++selectGen.current;
    setRunLoading(true);
    try {
      const run = await getRun(runId);
      if (gen !== selectGen.current) return 'aborted';
      setSelectedRunId(run.run_id);
      setSelectedRun(run);
      return 'ok';
    } catch {
      if (gen !== selectGen.current) return 'aborted';
      // Preserve earlier selectedRunId/selectedRun — caller sets explicit error.
      return 'failed';
    } finally {
      if (gen === selectGen.current) setRunLoading(false);
    }
  }, []);

  const commitRun = useCallback((run: RunRecord) => {
    selectGen.current += 1;
    setSelectedRunId(run.run_id);
    setSelectedRun(run);
    setRunLoading(false);
  }, []);

  // Bootstrap with effect-local cancel so StrictMode cleanup doesn't false-error.
  useEffect(() => {
    let cancelled = false;

    (async () => {
      setPageLoading(true);
      setError(null);
      try {
        const [sum, hist, props] = await Promise.all([
          getCase(caseId),
          getHistory(caseId),
          getCaseProposals(caseId),
        ]);
        if (cancelled) return;
        setSummary(sum);
        setHistory(hist);
        setCaseProposals(props);
        const newest = hist[0];
        if (newest) {
          const outcome = await loadRun(newest.run_id);
          if (cancelled) return;
          if (outcome === 'failed') {
            setError(
              `Failed to load run ${newest.run_id}. No earlier results to show.`,
            );
          }
        } else {
          setSelectedRunId(null);
          setSelectedRun(null);
        }
      } catch (e) {
        if (cancelled) return;
        setError(errMsg(e));
      } finally {
        if (!cancelled) setPageLoading(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [caseId, loadRun]);

  const onSelectHistory = useCallback(
    async (runId: string) => {
      setError(null);
      const outcome = await loadRun(runId);
      if (outcome === 'failed' && mounted.current) {
        setError(
          `Failed to load run ${runId}. Earlier results preserved.`,
        );
      }
    },
    [loadRun],
  );

  const onInvestigate = useCallback(async () => {
    if (investigatingRef.current) return;
    investigatingRef.current = true;
    setInvestigating(true);
    setError(null);
    try {
      const resp = await investigate(caseId);
      if (!mounted.current) return;

      // Investigation itself succeeded (paid call done). Refresh is separate.
      const refreshNotes: string[] = [];

      try {
        const hist = await getHistory(caseId);
        if (mounted.current) setHistory(hist);
      } catch (e) {
        refreshNotes.push(`history: ${errMsg(e)}`);
      }

      try {
        const props = await getCaseProposals(caseId);
        if (mounted.current) setCaseProposals(props);
      } catch (e) {
        refreshNotes.push(`proposals: ${errMsg(e)}`);
      }

      const outcome = await loadRun(resp.run_id);
      if (outcome === 'failed' && mounted.current) {
        // Use returned run_id/result so finished investigation remains visible.
        commitRun(runFromInvestigateResponse(resp));
        refreshNotes.push(`run ${resp.run_id}: fetch failed; showing response`);
      }

      if (refreshNotes.length && mounted.current) {
        setError(
          `Investigation finished but refresh failed: ${refreshNotes.join('; ')}`,
        );
      }
    } catch (e) {
      if (!mounted.current) return;
      // Real investigation failure — preserve prior selected run/results.
      setError(`Investigation failed: ${errMsg(e)}`);
    } finally {
      investigatingRef.current = false;
      if (mounted.current) setInvestigating(false);
    }
  }, [caseId, loadRun, commitRun]);

  const onRefresh = useCallback(async () => {
    await refreshProposals();
  }, [refreshProposals]);

  const proposals = visibleProposals(caseProposals, selectedRunId);

  const liveStatus = investigating
    ? 'Investigating…'
    : runLoading
      ? 'Loading investigation…'
      : pageLoading && !summary
        ? 'Loading case…'
        : null;

  if (pageLoading && !summary) {
    return (
      <main className="mx-auto max-w-6xl px-4 py-6 text-[14px] leading-[1.6] text-slate-900">
        <p className="text-sm text-slate-500" aria-live="polite">
          Loading case…
        </p>
      </main>
    );
  }

  // Show findings when we have a run; skip empty "no selection" while investigating.
  const showResult = selectedRun != null;

  return (
    <main className="mx-auto max-w-6xl px-4 py-6 text-[14px] leading-[1.6] text-slate-900">
      <div className="mb-4">
        <Button variant="ghost" size="sm" onClick={onBack} className="-ml-2 text-slate-600">
          ← Back to cases
        </Button>
      </div>

      <header className="mb-6 flex flex-col gap-4 sm:flex-row sm:items-start sm:justify-between">
        <div className="min-w-0">
          <p className="text-[11px] font-medium uppercase tracking-wide text-slate-500">
            Case
          </p>
          <div className="mt-1 flex flex-wrap items-center gap-2.5">
            <h1 className="text-2xl font-semibold tracking-tight text-slate-900">
              {summary?.case_id ?? caseId}
            </h1>
            {summary ? (
              <Badge variant="outline" className="capitalize">
                {summary.status}
              </Badge>
            ) : null}
          </div>
          {summary?.symptom ? (
            <p className="mt-1.5 text-[14px] leading-relaxed text-slate-600">
              {summary.symptom}
            </p>
          ) : null}
        </div>

        <div className="shrink-0 sm:pt-1 sm:self-start">
          <Button
            size="lg"
            className="w-full bg-blue-600 text-white hover:bg-blue-700 sm:w-auto"
            onClick={() => void onInvestigate()}
            disabled={investigating}
          >
            {investigating ? 'Investigating…' : 'Investigate'}
          </Button>
        </div>
      </header>

      {error ? (
        <p
          role="alert"
          className="mb-4 rounded-xl border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800"
        >
          {error}
        </p>
      ) : null}

      {/* Single live region — no duplicate loading copy elsewhere */}
      <div aria-live="polite" className="sr-only">
        {liveStatus}
      </div>
      {liveStatus && !(pageLoading && !summary) ? (
        <p className="mb-4 text-sm text-slate-500">{liveStatus}</p>
      ) : null}

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-[minmax(0,1fr)_260px] lg:gap-0">
        {/* History: above findings on mobile, right sidebar on desktop */}
        <aside className="order-1 min-w-0 border-b border-slate-200 pb-5 lg:order-2 lg:border-b-0 lg:border-l lg:pb-0 lg:pl-6">
          <RunHistory
            runs={history}
            selectedRunId={selectedRunId}
            onSelect={(id) => void onSelectHistory(id)}
            disabled={investigating || runLoading}
          />
        </aside>

        <div className="order-2 flex min-w-0 flex-col gap-6 lg:order-1 lg:pr-6">
          {showResult ? (
            <section className="min-w-0">
              <RunResult run={selectedRun} />
            </section>
          ) : !investigating && !runLoading ? (
            <section className="min-w-0 py-6">
              <p className="text-sm text-slate-500">
                No investigation yet. Run Investigate to start.
              </p>
            </section>
          ) : null}

          {proposals.length ? (
            <section className="flex min-w-0 flex-col gap-3">
              <div>
                <p className="text-[11px] font-medium uppercase tracking-wide text-slate-500">
                  Proposals
                </p>
                <h2 className="text-base font-semibold text-slate-900">
                  Task proposals
                </h2>
              </div>
              {proposals.map((p) => (
                <ProposalPanel
                  key={p.proposal_id}
                  proposal={p}
                  onRefresh={onRefresh}
                  disabled={investigating}
                />
              ))}
            </section>
          ) : null}
        </div>
      </div>
    </main>
  );
}

export default CaseDetail;
