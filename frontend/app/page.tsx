"use client";

import { useEffect, useState } from "react";
import { loadSession, type Session } from "@/lib/api";
import Chat from "./chat";
import Login from "./login";

export default function Page() {
  const [session, setSession] = useState<Session | null>(null);
  const [ready, setReady] = useState(false);

  /* sessionStorage is not available while rendering on the server, so the restored
   * session arrives after mount. Until then, render nothing rather than flashing the
   * sign-in screen at someone who is already signed in. */
  useEffect(() => {
    setSession(loadSession());
    setReady(true);
  }, []);

  if (!ready) return null;
  if (!session) return <Login onSignedIn={setSession} />;
  return <Chat session={session} onSignedOut={() => setSession(null)} />;
}
