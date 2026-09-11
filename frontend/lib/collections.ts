/* The five collections, their colours, and who may read each.
 *
 * This mirrors COLLECTION_ROLES in the backend's config.py. It is duplicated here only
 * for the sign-in screen, which has to describe every role before anyone has a token.
 * Once signed in, the rail reads the authoritative list from GET /collections/{role}. */

export const COLLECTIONS = ["general", "clinical", "nursing", "billing", "equipment"] as const;
export type Collection = (typeof COLLECTIONS)[number];

export const COLLECTION_COLOUR: Record<string, string> = {
  general: "var(--general)",
  clinical: "var(--clinical)",
  nursing: "var(--nursing)",
  billing: "var(--billing)",
  equipment: "var(--equipment)",
};

/* What each collection holds, in the words a member of staff would use. */
/* The operations database is a source too, but not a collection: it answers with counts
 * rather than passages, and it is gated per role rather than per document. It is named
 * and described here so the rail can list it alongside the collections without implying
 * it belongs to the same colour system. */
export const RECORDS_SOURCE = "claims and ticket records";
/* Was "counts and totals, not documents", which was a promise the system does not keep:
 * a question like "which patients have claims above 200000" returns the rows themselves.
 * The wording now describes what the source is rather than what shape the answer takes. */
export const RECORDS_HOLDS = "claim and ticket rows, queried directly";

export const COLLECTION_HOLDS: Record<Collection, string> = {
  general: "handbook, leave policy, code of conduct",
  clinical: "drug formulary, treatment protocols, diagnostics",
  nursing: "ICU procedures, infection control",
  billing: "insurance codes, claim submission",
  equipment: "operation and maintenance manuals",
};

export type DemoAccount = {
  username: string;
  role: string;
  roleLabel: string;
  canRead: Collection[];
  /* Access has two axes. Documents are filtered per collection inside the vector search;
   * the operations database is gated per role, and only these two may query it. */
  canQuery: boolean;
};

/* The five accounts the assignment names. Passwords follow one rule, stated on screen. */
export const DEMO_ACCOUNTS: DemoAccount[] = [
  { username: "dr.mehta", role: "doctor", roleLabel: "Doctor", canRead: ["general", "clinical", "nursing"], canQuery: false },
  { username: "nurse.priya", role: "nurse", roleLabel: "Nurse", canRead: ["general", "nursing"], canQuery: false },
  { username: "billing.ravi", role: "billing_executive", roleLabel: "Billing executive", canRead: ["general", "billing"], canQuery: true },
  { username: "tech.anand", role: "technician", roleLabel: "Technician", canRead: ["general", "equipment"], canQuery: false },
  { username: "admin.sys", role: "admin", roleLabel: "Admin", canRead: ["general", "clinical", "nursing", "billing", "equipment"], canQuery: true },
];

export const demoPassword = (username: string) => `${username}-demo`;

export const roleLabel = (role: string) =>
  DEMO_ACCOUNTS.find((a) => a.role === role)?.roleLabel ?? role.replace(/_/g, " ");

/* Mirrors SQL_RAG_ROLES in the backend's config.py. */
export const canQueryDatabase = (role: string) =>
  DEMO_ACCOUNTS.some((account) => account.role === role && account.canQuery);
