"use client";

import { useState } from "react";
import { login, saveSession, type Session } from "@/lib/api";
import { COLLECTION_COLOUR, DEMO_ACCOUNTS, demoPassword } from "@/lib/collections";
import styles from "./login.module.css";

/* The sign-in screen states the access model before anyone signs in, rather than saving
 * it for the first refusal. Each account shows exactly which collections it can read, so
 * comparing two rows shows what a role gives up. */
export default function Login({ onSignedIn }: { onSignedIn: (session: Session) => void }) {
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  async function submit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      const session = await login(username.trim(), password);
      saveSession(session);
      onSignedIn(session);
    } catch (failure) {
      setError(failure instanceof Error ? failure.message : "Sign-in failed.");
      setBusy(false);
    }
  }

  function pick(name: string) {
    setUsername(name);
    setPassword(demoPassword(name));
    setError(null);
  }

  return (
    <main className={styles.screen}>
      <div className={styles.intro}>
        <h1>MediAssist staff assistant</h1>
        <p className={styles.lede}>
          Ask about the hospital&rsquo;s documents and operations. You will only ever be
          answered from the collections your role is cleared for.
        </p>

        <form className={styles.form} onSubmit={submit}>
          <label className={styles.field}>
            <span>Username</span>
            <input
              value={username}
              onChange={(e) => setUsername(e.target.value)}
              autoComplete="username"
              required
            />
          </label>
          <label className={styles.field}>
            <span>Password</span>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              autoComplete="current-password"
              required
            />
          </label>
          <button className={styles.submit} type="submit" disabled={busy}>
            {busy ? "Signing in" : "Sign in"}
          </button>
          {error && (
            <p className={styles.error} role="alert">
              {error}
            </p>
          )}
        </form>
      </div>

      <div>
        <table className={styles.accounts}>
          <caption>
            Five demo accounts, one per role. Pick one to fill the form, or type your own.
            Documents are filtered per collection; counts over the claims and maintenance
            records are limited to the two roles whose work needs them.
          </caption>
          <thead>
            <tr>
              <th scope="col">Account</th>
              <th scope="col">Role</th>
              <th scope="col">Can read</th>
              <th scope="col">Can count</th>
            </tr>
          </thead>
          <tbody>
            {DEMO_ACCOUNTS.map((account) => (
              <tr key={account.username}>
                <td>
                  <button
                    type="button"
                    className={styles.pick}
                    aria-pressed={username === account.username}
                    onClick={() => pick(account.username)}
                  >
                    {account.username}
                  </button>
                </td>
                <td className={styles.role}>{account.roleLabel}</td>
                <td>
                  <div className={styles.reads}>
                    {account.canRead.map((collection) => (
                      <span className={styles.chip} key={collection}>
                        <span
                          className={styles.swatch}
                          style={{ background: COLLECTION_COLOUR[collection] }}
                        />
                        {collection}
                      </span>
                    ))}
                  </div>
                </td>
                <td className={styles.query}>
                  {account.canQuery ? "claims, tickets" : <span aria-label="no">&mdash;</span>}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className={styles.passwordRule}>
          Each demo password is the username followed by <code>-demo</code>.
        </p>
      </div>
    </main>
  );
}
