export interface Session {
  session_id: string;
  title: string;
  memory_turn_limit: number;
  created_at: string;
  updated_at: string;
}

export interface ConversationTurn {
  turn_number: number;
  user_question: string;
  assistant_answer: string;
  created_at: string;
}

export interface FeatureFlags {
  faq_enabled: boolean;
  classifier_enabled: boolean;
}

export interface QueryResult {
  answer: string;
  source: "faq" | "rag";
  classification: string | null;
  citations: Array<{ chunk_id: string; source: string; title_path: string; text: string }>;
  web_citations: Array<{ title: string; url: string; snippet: string }>;
  web_search_used: boolean;
  session_id: string;
  turn_number: number;
  selected_memory_turns: number;
}

export interface IngestionJob {
  job_id: string;
  original_filename: string;
  status: "PENDING" | "RUNNING" | "SUCCEEDED" | "FAILED" | "DELETED";
  document_id: string | null;
  stored_chunks: number;
  error_message: string | null;
  created_at: string;
  updated_at: string;
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    const body = await response.text();
    throw new Error(body || `请求失败：${response.status}`);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export const api = {
  createSession: (memoryTurnLimit = 12) => request<Session>("/sessions", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ memory_turn_limit: memoryTurnLimit })
  }),
  listSessions: () => request<Session[]>("/sessions"),
  updateSession: (sessionId: string, payload: Partial<Pick<Session, "title" | "memory_turn_limit">>) => request<Session>(`/sessions/${sessionId}`, {
    method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload)
  }),
  deleteSession: (sessionId: string) => request<void>(`/sessions/${sessionId}`, { method: "DELETE" }),
  getTurns: (sessionId: string) => request<ConversationTurn[]>(`/sessions/${sessionId}/turns`),
  query: (question: string, sessionId: string) => request<QueryResult>("/query", {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question, session_id: sessionId })
  }),
  getFeatures: () => request<FeatureFlags>("/runtime/features"),
  updateFeatures: (payload: Partial<FeatureFlags>) => request<FeatureFlags>("/runtime/features", {
    method: "PATCH", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload)
  }),
  listJobs: () => request<IngestionJob[]>("/admin/documents"),
  uploadDocument: (file: File) => {
    const form = new FormData(); form.append("file", file);
    return request<IngestionJob>("/admin/documents", { method: "POST", body: form });
  },
  deleteDocument: (jobId: string) => request<IngestionJob>(`/admin/documents/${jobId}`, { method: "DELETE" }),
  importFaq: (file: File) => {
    const form = new FormData(); form.append("file", file);
    return request<{ created_count: number; updated_count: number; redis_warmed_count: number }>("/admin/qa/import", { method: "POST", body: form });
  }
};
