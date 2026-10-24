/** Frontend mirrors of CasePilot shared contracts (shape-only). */

export type Diagnosis =
  | "missing_document"
  | "data_mismatch"
  | "processing_failure"
  | "indeterminate"
  | "no_issue";

export type NextAction =
  | "create_follow_up_task"
  | "request_missing_document"
  | "correct_data_mismatch"
  | "retry_processing"
  | "escalate_to_human"
  | "no_action_required";

export type RunStatus = "completed" | "indeterminate" | "failed";

export type ProposalStatus =
  | "pending"
  | "rejected"
  | "approved"
  | "executed"
  | "execution_failed";

export interface CaseSummary {
  case_id: string;
  status: string;
  symptom: string;
}

export interface CaseListItem {
  case_id: string;
  status: string;
  symptom: string;
  pending_review: boolean;
  needs_execution: boolean;
}

export interface RunSummary {
  run_id: string;
  case_id: string;
  status: RunStatus;
  created_at: string;
  completed_at: string | null;
}

export interface FollowUpTaskPayload {
  title: string;
  body: string;
  case_id: string;
}

export interface Evidence {
  source_system: string;
  record_id: string;
  field: string;
  fact: string;
  retrieval_ref: string;
}

export interface InvestigationResult {
  run_id: string;
  case_id: string;
  status: RunStatus;
  diagnosis: Diagnosis;
  diagnosis_summary: string;
  explanation: string;
  evidence: Evidence[];
  uncertainty: string | null;
  unresolved_questions: string[];
  next_action: NextAction;
  next_action_explanation: string;
  approval_required: boolean;
  proposed_task: FollowUpTaskPayload | null;
  proposal_id: string | null;
}

export interface RunRecord {
  run_id: string;
  case_id: string;
  objective: string;
  status: RunStatus;
  result: InvestigationResult | null;
  error: string | null;
  created_at: string;
  completed_at: string | null;
}

export interface Proposal {
  proposal_id: string;
  run_id: string;
  case_id: string;
  action: NextAction;
  task_payload: FollowUpTaskPayload;
  status: ProposalStatus;
  payload_digest: string;
  created_at: string;
  decided_at: string | null;
  operator_identity: string | null;
  execution_error: string | null;
  /** Simulator-confirmed; never fabricate for legacy rows. */
  task_id: string | null;
}

export interface InvestigationResponse {
  run_id: string;
  result: InvestigationResult;
}

export interface ProposalDecisionResponse {
  proposal: Proposal;
}
