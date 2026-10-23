import type { ReactNode } from 'react';
import type { InvestigationResult, RunRecord, RunStatus } from '../types';
import { Badge } from '@/components/ui/badge';

type Props = {
  run: RunRecord | null;
  loading?: boolean;
};

function humanLabel(value: string): string {
  return value.replaceAll('_', ' ');
}

function statusVariant(
  status: RunStatus | string,
): 'success' | 'warning' | 'danger' | 'secondary' {
  if (status === 'completed') return 'success';
  if (status === 'indeterminate') return 'warning';
  if (status === 'failed') return 'danger';
  return 'secondary';
}

function Eyebrow({ children }: { children: ReactNode }) {
  return (
    <p className="text-[11px] font-medium uppercase tracking-wide text-slate-500">
      {children}
    </p>
  );
}

function SectionRule() {
  return <hr className="border-0 border-t border-slate-100" />;
}

function Surface({ children }: { children: ReactNode }) {
  return (
    <div className="min-w-0 rounded-xl border border-slate-200 bg-white px-5 py-5">
      <div className="flex min-w-0 flex-col gap-5">{children}</div>
    </div>
  );
}

function EvidenceList({ result }: { result: InvestigationResult }) {
  if (!result.evidence?.length) return null;
  return (
    <section className="min-w-0 space-y-3">
      <div>
        <Eyebrow>Supporting detail</Eyebrow>
        <h3 className="mt-0.5 text-base font-semibold text-slate-900">
          Evidence
        </h3>
      </div>
      <ol className="min-w-0">
        {result.evidence.map((e, i) => (
          <li
            key={`${e.retrieval_ref}-${i}`}
            className="flex gap-3 border-t border-slate-100 py-3 first:border-t-0 first:pt-0 last:pb-0"
          >
            <span className="mt-0.5 w-5 shrink-0 text-[11px] font-medium tabular-nums text-slate-400">
              {i + 1}.
            </span>
            <div className="min-w-0 flex-1 space-y-1">
              <p className="text-sm leading-relaxed text-slate-800 break-words">
                {e.fact}
              </p>
              <p className="font-mono text-[11px] leading-snug text-slate-500 break-all">
                {e.source_system} / {e.record_id} / {e.field}
                {e.retrieval_ref ? ` (${e.retrieval_ref})` : ''}
              </p>
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}

export function RunResult({ run, loading }: Props) {
  if (loading) {
    return <p className="text-sm text-slate-500">Loading investigation…</p>;
  }
  if (!run) {
    return <p className="text-sm text-slate-500">No investigation selected.</p>;
  }

  const { result, error, status } = run;

  if (!result) {
    return (
      <Surface>
        <section className="min-w-0 space-y-2">
          <div className="flex flex-wrap items-center gap-2">
            <Eyebrow>Investigation</Eyebrow>
            <Badge variant={statusVariant(status)}>{humanLabel(status)}</Badge>
          </div>
          <h2 className="text-base font-semibold text-slate-900">
            Investigation
          </h2>
          {error ? (
            <p role="alert" className="text-sm text-red-700 break-words">
              {error}
            </p>
          ) : (
            <p className="text-sm text-slate-600">
              No result recorded for this run.
            </p>
          )}
        </section>
      </Surface>
    );
  }

  const showUncertainty =
    Boolean(result.uncertainty) || Boolean(result.unresolved_questions?.length);
  const showEvidence = Boolean(result.evidence?.length);
  const showTask = Boolean(result.proposed_task);

  return (
    <Surface>
      {/* Findings */}
      <section className="min-w-0 space-y-3">
        <div className="flex flex-wrap items-center gap-2">
          <Eyebrow>Findings</Eyebrow>
          <Badge variant={statusVariant(result.status)}>
            {humanLabel(result.status)}
          </Badge>
          {run.status !== result.status ? (
            <Badge variant="outline">run: {humanLabel(run.status)}</Badge>
          ) : null}
          <Badge variant="secondary">{humanLabel(result.diagnosis)}</Badge>
        </div>
        <h2 className="text-base font-semibold text-slate-900">Diagnosis</h2>
        {error ? (
          <p role="alert" className="text-sm text-red-700 break-words">
            Run error: {error}
          </p>
        ) : null}
        <p className="text-lg font-semibold leading-snug text-slate-900 break-words">
          {result.diagnosis_summary}
        </p>
        <p className="text-sm leading-[1.7] text-slate-700 break-words">
          {result.explanation}
        </p>
      </section>

      <SectionRule />

      {/* Next action — left rule only, no box fill */}
      <section className="min-w-0 border-l-2 border-sky-400 pl-3">
        <Eyebrow>Recommended next action</Eyebrow>
        <p className="mt-1 text-sm font-semibold capitalize text-slate-900 break-words">
          {humanLabel(result.next_action)}
        </p>
        <p className="mt-1 text-sm leading-[1.7] text-slate-700 break-words">
          {result.next_action_explanation}
        </p>
      </section>

      {showEvidence ? (
        <>
          <SectionRule />
          <EvidenceList result={result} />
        </>
      ) : null}

      {showUncertainty ? (
        <>
          <SectionRule />
          <section className="min-w-0 space-y-3 border-l-2 border-amber-400 pl-3">
            {result.uncertainty ? (
              <div>
                <div className="flex items-center gap-1.5">
                  <span
                    aria-hidden
                    className="text-amber-600 text-sm leading-none"
                  >
                    ⚠
                  </span>
                  <Eyebrow>Uncertainty</Eyebrow>
                </div>
                <p className="mt-1 text-sm leading-[1.7] text-slate-700 break-words">
                  {result.uncertainty}
                </p>
              </div>
            ) : null}
            {result.unresolved_questions?.length ? (
              <div>
                <div className="flex items-center gap-1.5">
                  <span
                    aria-hidden
                    className="text-amber-600 text-sm leading-none"
                  >
                    ⚠
                  </span>
                  <Eyebrow>Unresolved questions</Eyebrow>
                </div>
                <ul className="mt-1 list-disc space-y-1 pl-5 text-sm leading-[1.7] text-slate-700">
                  {result.unresolved_questions.map((q) => (
                    <li key={q} className="break-words">
                      {q}
                    </li>
                  ))}
                </ul>
              </div>
            ) : null}
          </section>
        </>
      ) : null}

      {showTask && result.proposed_task ? (
        <>
          <SectionRule />
          <section className="min-w-0 space-y-2">
            <div>
              <Eyebrow>Follow-up</Eyebrow>
              <h3 className="mt-0.5 text-base font-semibold text-slate-900">
                Proposed follow-up task
              </h3>
            </div>
            <dl className="min-w-0 space-y-3 text-sm">
              <div className="min-w-0">
                <dt className="text-[11px] font-medium uppercase tracking-wide text-slate-500">
                  Title
                </dt>
                <dd className="mt-0.5 font-semibold text-slate-900 whitespace-pre-wrap break-words">
                  {result.proposed_task.title}
                </dd>
              </div>
              <div className="min-w-0">
                <dt className="text-[11px] font-medium uppercase tracking-wide text-slate-500">
                  Body
                </dt>
                <dd className="mt-0.5 max-h-64 overflow-y-auto leading-[1.7] text-slate-700 whitespace-pre-wrap break-words">
                  {result.proposed_task.body}
                </dd>
              </div>
              <div className="min-w-0">
                <dt className="text-[11px] font-medium uppercase tracking-wide text-slate-500">
                  Case
                </dt>
                <dd className="mt-0.5 font-mono text-[11px] text-slate-500 break-all">
                  {result.proposed_task.case_id}
                </dd>
              </div>
            </dl>
            {!result.proposal_id ? (
              <p className="text-sm italic text-slate-500">
                No proposal record available for review controls.
              </p>
            ) : null}
          </section>
        </>
      ) : null}
    </Surface>
  );
}

export default RunResult;
