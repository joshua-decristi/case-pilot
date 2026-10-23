/** Frontend mirrors of CasePilot shared contracts (browse slice). */

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
