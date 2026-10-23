import { useCallback, useEffect, useState } from "react";
import { CaseList } from "@/components/CaseList";
import { CaseDetail } from "@/components/CaseDetail";

function caseIdFromHash(): string | null {
  const raw = window.location.hash.replace(/^#/, "").replace(/^\//, "");
  if (!raw) return null;
  // Support #/cases/CASE-001, #cases/CASE-001, #CASE-001
  const m = raw.match(/^(?:cases\/)?([^/?#]+)/i);
  if (!m?.[1]) return null;
  try {
    return decodeURIComponent(m[1]);
  } catch {
    return m[1];
  }
}

function setHashCase(caseId: string | null) {
  if (caseId) {
    window.location.hash = `/cases/${encodeURIComponent(caseId)}`;
  } else if (window.location.hash) {
    history.replaceState(
      null,
      "",
      `${window.location.pathname}${window.location.search}`,
    );
  }
}

function BrandMark() {
  return (
    <span
      className="flex h-7 w-7 shrink-0 items-center justify-center rounded-md bg-slate-900 text-[11px] font-bold tracking-tight text-slate-50"
      aria-hidden
    >
      CP
    </span>
  );
}

export default function App() {
  const [caseId, setCaseId] = useState<string | null>(() => caseIdFromHash());

  useEffect(() => {
    const onHash = () => setCaseId(caseIdFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const openCase = useCallback((id: string) => {
    setCaseId(id);
    setHashCase(id);
  }, []);

  const back = useCallback(() => {
    setCaseId(null);
    setHashCase(null);
  }, []);

  return (
    <div className="mx-auto min-h-full w-full max-w-6xl px-4 py-4 sm:px-6 sm:py-6">
      <header className="mb-5 flex items-center gap-2.5 border-b border-slate-200/80 pb-4">
        <BrandMark />
        <div className="min-w-0 leading-tight">
          <div className="text-sm font-semibold tracking-tight text-slate-900">
            CasePilot
          </div>
          <div className="text-xs text-slate-500">Operator workspace</div>
        </div>
      </header>
      <main>
        {caseId ? (
          <CaseDetail key={caseId} caseId={caseId} onBack={back} />
        ) : (
          <CaseList onOpen={openCase} />
        )}
      </main>
    </div>
  );
}
