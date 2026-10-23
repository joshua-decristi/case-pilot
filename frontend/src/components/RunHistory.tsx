import type { RunStatus, RunSummary } from '@/types';
import { Badge } from '@/components/ui/badge';
import { cn } from '@/lib/utils';

type Props = {
  runs: RunSummary[];
  selectedRunId: string | null;
  onSelect: (runId: string) => void;
  disabled?: boolean;
};

function formatWhen(iso: string | null | undefined): string {
  if (!iso) return '—';
  try {
    return new Date(iso).toLocaleString(undefined, {
      dateStyle: 'medium',
      timeStyle: 'short',
    });
  } catch {
    return iso;
  }
}

function truncateId(id: string): string {
  if (id.length <= 12) return id;
  return `${id.slice(0, 8)}…`;
}

function statusVariant(
  status: RunStatus,
): 'success' | 'danger' | 'warning' | 'secondary' {
  switch (status) {
    case 'completed':
      return 'success';
    case 'failed':
      return 'danger';
    case 'indeterminate':
      return 'warning';
    default:
      return 'secondary';
  }
}

export function RunHistory({ runs, selectedRunId, onSelect, disabled }: Props) {
  return (
    <section>
      <p className="text-[11px] font-medium uppercase tracking-wide text-slate-500">
        History
      </p>
      <h2 className="mt-0.5 text-sm font-semibold text-slate-900">
        Investigation history
      </h2>

      {!runs.length ? (
        <p className="mt-3 text-sm leading-relaxed text-slate-500">
          No investigations yet.
        </p>
      ) : (
        <ul className="mt-3 flex flex-col gap-1">
          {runs.map((r) => {
            const selected = r.run_id === selectedRunId;
            return (
              <li key={r.run_id}>
                <button
                  type="button"
                  disabled={disabled || selected}
                  aria-current={selected ? 'true' : undefined}
                  title={r.run_id}
                  onClick={() => onSelect(r.run_id)}
                  className={cn(
                    'w-full rounded-md px-2 py-2 text-left transition-colors',
                    'disabled:pointer-events-none',
                    selected
                      ? 'bg-blue-50 text-slate-900'
                      : 'text-slate-800 hover:bg-slate-100',
                    disabled && !selected ? 'opacity-50' : null,
                  )}
                >
                  <div className="flex items-start justify-between gap-2">
                    <span
                      className={cn(
                        'text-[13px] leading-snug',
                        selected && 'font-medium',
                      )}
                    >
                      {formatWhen(r.created_at)}
                    </span>
                    <Badge
                      variant={statusVariant(r.status)}
                      className="shrink-0 capitalize"
                    >
                      {r.status}
                    </Badge>
                  </div>
                  <span
                    className="mt-1 block truncate font-mono text-[11px] text-slate-400"
                    title={r.run_id}
                  >
                    {truncateId(r.run_id)}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </section>
  );
}

export default RunHistory;
