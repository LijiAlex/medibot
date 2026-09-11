/* The only place that talks to the backend.
 *
 * The token is kept in sessionStorage rather than localStorage: it expires in an hour,
 * and a closed tab should not leave one lying around. */

const BASE = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export type Source = {
  source_document: string;
  section_title: string;
  collection: string;
};

export type Answer = {
  answer: string;
  sources: Source[];
  retrieval_type: "hybrid_rag" | "sql_rag";
  role: string;
  sql: string | null;
  /* Why there is no answer, when there is none: "role", "not_found" or "no_query".
   * null on a real answer. The label on screen reads this rather than guessing from an
   * empty sources list. */
  refusal: "role" | "not_found" | "no_query" | null;
};

export type Session = { token: string; role: string };

const SESSION_KEY = "medibot.session";

export function loadSession(): Session | null {
  if (typeof window === "undefined") return null;
  const raw = window.sessionStorage.getItem(SESSION_KEY);
  return raw ? (JSON.parse(raw) as Session) : null;
}

export function saveSession(session: Session): void {
  window.sessionStorage.setItem(SESSION_KEY, JSON.stringify(session));
}

export function clearSession(): void {
  window.sessionStorage.removeItem(SESSION_KEY);
}

async function readError(response: Response, fallback: string): Promise<string> {
  try {
    const body = await response.json();
    return typeof body.detail === "string" ? body.detail : fallback;
  } catch {
    return fallback;
  }
}

export async function login(username: string, password: string): Promise<Session> {
  const response = await fetch(`${BASE}/login`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ username, password }),
  });
  if (!response.ok) throw new Error(await readError(response, "Sign-in failed."));
  return (await response.json()) as Session;
}

export async function ask(question: string, token: string): Promise<Answer> {
  const response = await fetch(`${BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json", Authorization: `Bearer ${token}` },
    body: JSON.stringify({ question }),
  });
  if (response.status === 401) throw new Error("Your session has ended. Sign in again.");
  if (!response.ok) throw new Error(await readError(response, "MediBot could not answer that."));
  return (await response.json()) as Answer;
}

export async function collectionsFor(role: string): Promise<string[]> {
  const response = await fetch(`${BASE}/collections/${role}`);
  if (!response.ok) return [];
  return ((await response.json()) as { collections: string[] }).collections;
}
