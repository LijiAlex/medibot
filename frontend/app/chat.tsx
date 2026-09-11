"use client";

import { useEffect, useRef, useState } from "react";
import { ask, clearSession, collectionsFor, type Answer, type Session } from "@/lib/api";
import {
  COLLECTIONS,
  COLLECTION_COLOUR,
  COLLECTION_HOLDS,
  RECORDS_HOLDS,
  RECORDS_SOURCE,
  canQueryDatabase,
  roleLabel,
  type Collection,
} from "@/lib/collections";
import styles from "./chat.module.css";

type Turn = { question: string; answer: Answer };

/* Shown while the answer is on its way. Measured 2026-09-11: a request takes about one
 * and a half to two seconds, so a staged sequence would never reach its later steps. One
 * line, picked at random, for the one beat the wait actually lasts. Each is true of both
 * branches, since the frontend does not yet know which one will answer. */
const WAITING = [
  "Working on it",
  "Checking what your role can see",
  "Searching the sources you can read",
  "Reading the relevant sections",
  "Putting the answer together",
  "Looking this up",
];

const pickWaiting = () => WAITING[Math.floor(Math.random() * WAITING.length)];

/* A SQL citation carries collection "sql", which is the field name rather than something
 * a reader recognises. Show the words the rail already uses for that source. */
const sourceLabel = (collection: string) => (collection === "sql" ? "records" : collection);

/* The model writes citation markers as 【1】 and emphasises doses with markdown bold.
 * Normalise the markers, and render the bold rather than showing its asterisks. This is
 * the only markdown it reliably produces, so a whole parser would be a dependency spent
 * on nothing. */
const tidy = (text: string) =>
  text
    /* The model writes 【1】, and sometimes 【1†L4-L7】 with a line range attached. The
     * first form was handled and the second leaked through as literal text. Keep the
     * number, drop everything after the dagger. */
    .replace(/【(\d+)[^】]*】/g, "[$1]")
    /* It also emits U+202F narrow no-break space and U+00A0 inside names and doses:
     * "HDFC\u202fErgo", "1\u202fg". Invisible on screen, but it breaks find-in-page and
     * survives a copy-paste into a spreadsheet, so normalise it to a plain space. */
    .replace(/[\u202f\u00a0]/g, " ");

/* The SQL branch appends a note about the row cap. It is a caveat about how the query
 * ran, not part of the finding, so it is lifted out of the answer and set quietly. */
const ROW_CAP = "Results limited to 5 rows.";

function split(answer: string): { body: string; caveat: string | null } {
  const trimmed = answer.trimEnd();
  return trimmed.endsWith(ROW_CAP)
    ? { body: trimmed.slice(0, -ROW_CAP.length).trimEnd(), caveat: ROW_CAP }
    : { body: trimmed, caveat: null };
}

function render(text: string) {
  return tidy(text)
    .split(/(\*\*[^*]+\*\*)/g)
    .filter(Boolean)
    .map((piece, index) =>
      piece.startsWith("**") && piece.endsWith("**") ? (
        <strong key={index}>{piece.slice(2, -2)}</strong>
      ) : (
        <span key={index}>{piece}</span>
      ),
    );
}

/* Openers worth trying, chosen per role from questions this corpus actually answers. */
const OPENERS: Record<string, string[]> = {
  doctor: ["What is the standard dose of meropenem?", "What are the critical values I must escalate?"],
  nurse: ["Which IV cannula size for a paediatric patient under 5 kg?", "When do I change a peripheral cannula?"],
  billing_executive: ["What is on the pre-authorisation document checklist?", "How many claims are still pending?"],
  technician: ["What does fault code F-05 mean on the infusion pump?", "What are the daily log requirements for the X-ray unit?"],
  admin: ["Which equipment category has the most open maintenance tickets?", "Which insurers are empanelled with us?"],
};

export default function Chat({ session, onSignedOut }: { session: Session; onSignedOut: () => void }) {
  const [permitted, setPermitted] = useState<string[]>([]);
  const [turns, setTurns] = useState<Turn[]>([]);
  const [question, setQuestion] = useState("");
  /* The question being answered right now. Held apart from the finished turns so it can
   * appear the instant it is asked, rather than only when the answer arrives. */
  const [pending, setPending] = useState<{ question: string; waiting: string } | null>(null);
  const [failure, setFailure] = useState<string | null>(null);
  const end = useRef<HTMLDivElement>(null);

  useEffect(() => {
    collectionsFor(session.role).then(setPermitted);
  }, [session.role]);

  useEffect(() => {
    end.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns.length, pending]);

  async function send(text: string) {
    const asked = text.trim();
    if (!asked || pending !== null) return;
    setPending({ question: asked, waiting: pickWaiting() });
    setFailure(null);
    setQuestion("");
    try {
      const answer = await ask(asked, session.token);
      setTurns((previous) => [...previous, { question: asked, answer }]);
    } catch (error) {
      setFailure(error instanceof Error ? error.message : "MediBot could not answer that.");
      setQuestion(asked);
    } finally {
      setPending(null);
    }
  }

  const counts = canQueryDatabase(session.role);
  const withheld = [
    ...COLLECTIONS.filter((collection) => !permitted.includes(collection)),
    ...(counts ? [] : [RECORDS_SOURCE]),
  ];

  return (
    <div className={styles.shell}>
      <aside className={styles.rail}>
        <div className={styles.who}>
          <strong>{roleLabel(session.role)}</strong>
          <span>signed in</span>
        </div>

        <div className={styles.railGroup}>
          <h2>Answers may come from</h2>
          <div className={styles.reads}>
            {permitted.map((collection) => (
              <span className={styles.chip} key={collection}>
                <span className={styles.swatch} style={{ background: COLLECTION_COLOUR[collection] }} />
                <span>
                  {collection}
                  <br />
                  <em>{COLLECTION_HOLDS[collection as Collection]}</em>
                </span>
              </span>
            ))}
            {counts && (
              <span className={styles.chip}>
                <span className={`${styles.swatch} ${styles.recordsSwatch}`} />
                <span>
                  {RECORDS_SOURCE}
                  <br />
                  <em>{RECORDS_HOLDS}</em>
                </span>
              </span>
            )}
          </div>
        </div>

        {/* Same frame, the other half. The database sits in whichever list applies, so a
            role reads one answer to one question rather than three unrelated blocks. */}
        {withheld.length > 0 && (
          <div className={styles.railGroup}>
            <h2>Never from</h2>
            <p className={styles.denied}>{withheld.join(", ")}</p>
          </div>
        )}

        <button
          className={styles.signOut}
          onClick={() => {
            clearSession();
            onSignedOut();
          }}
        >
          Sign out
        </button>
      </aside>

      <div className={styles.main}>
        <div className={styles.thread}>
          {turns.length === 0 && (
            <div className={styles.empty}>
              <h1>Ask MediBot</h1>
              <p>
                Answers are quoted from the collections listed beside you, with the document
                and section they came from.
              </p>
              <ul className={styles.tryList}>
                {(OPENERS[session.role] ?? []).map((opener) => (
                  <li key={opener}>
                    <button className={styles.try} onClick={() => send(opener)}>
                      {opener}
                    </button>
                  </li>
                ))}
              </ul>
            </div>
          )}

          {turns.map((turn, index) => {
            const blocked = turn.answer.sources.length === 0;
            const { body, caveat } = split(turn.answer.answer);
            const sql = turn.answer.retrieval_type === "sql_rag";
            /* Spec asks for the retrieval type on every response. When nothing was
             * retrieved, saying only "database query" would claim a query that never
             * ran, so the label says what actually happened. */
            /* The spec names these labels (Component 6): "Hybrid RAG" or "SQL RAG" on
             * each response. Its words rather than plainer ones, because this label is
             * read by whoever is assessing the system, not by the nurse asking.
             *
             * The qualifier comes from the backend's refusal reason rather than from an
             * empty sources list, which cannot tell a permission decision from an empty
             * search. Saying "nothing matched" about a question that was never searched
             * for would contradict the message directly above it. */
            const route = sql ? "SQL RAG" : "Hybrid RAG";
            const marker =
              turn.answer.refusal === "role"
                ? `${route}, blocked by role`
                : turn.answer.refusal === "not_found"
                  ? `${route}, nothing matched`
                  : turn.answer.refusal === "no_query"
                    ? `${route}, no query formed`
                    : route;
            return (
              <article className={styles.turn} key={index}>
                <p className={styles.question}>{turn.question}</p>

                <p className={blocked ? styles.refusal : styles.answer}>{render(body)}</p>
                {caveat && <p className={styles.caveat}>{caveat}</p>}

                {turn.answer.sources.length > 0 && (
                  <table className={styles.cite}>
                    <caption>
                      {turn.answer.retrieval_type === "sql_rag" ? "Counted from" : "Quoted from"}
                    </caption>
                    <tbody>
                      {turn.answer.sources.map((source, i) => (
                        <tr key={i}>
                          <td>
                            <span
                              className={
                                COLLECTION_COLOUR[source.collection]
                                  ? styles.swatch
                                  : `${styles.swatch} ${styles.recordsSwatch}`
                              }
                              style={
                                COLLECTION_COLOUR[source.collection]
                                  ? { background: COLLECTION_COLOUR[source.collection] }
                                  : undefined
                              }
                            />
                          </td>
                          {/* The name beside the swatch, so the colour is a second way of
                              reading the row rather than the only way. */}
                          <td className={styles.collection}>{sourceLabel(source.collection)}</td>
                          <td className={styles.doc}>{source.source_document}</td>
                          <td className={styles.section}>{source.section_title}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                )}

                {turn.answer.sql && (
                  <details className={styles.sql}>
                    <summary>Show the query that was run</summary>
                    <pre>{turn.answer.sql}</pre>
                  </details>
                )}

                <p className={styles.marker}>{marker}</p>
              </article>
            );
          })}

          {pending && (
            <article className={styles.turn}>
              <p className={styles.question}>{pending.question}</p>
              <p className={styles.progress} aria-live="polite">
                {pending.waiting}
              </p>
            </article>
          )}

          {failure && <p className={styles.failed}>{failure}</p>}
          <div ref={end} />
        </div>

        <form
          className={styles.ask}
          onSubmit={(event) => {
            event.preventDefault();
            send(question);
          }}
        >
          <div className={styles.askInner}>
            <input
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              placeholder="Ask a question"
              aria-label="Ask a question"
              disabled={pending !== null}
            />
            <button type="submit" disabled={pending !== null || !question.trim()}>
              Ask
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
