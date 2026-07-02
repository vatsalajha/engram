import { useState, useEffect } from "react";
import { ChatView } from "./views/ChatView.tsx";
import { DashboardView } from "./views/DashboardView.tsx";

type Tab = "chat" | "dashboard";

/** Hash routing: "/" ↔ chat, "#dashboard" ↔ dashboard (deep-linkable). */
function tabFromHash(): Tab {
  return window.location.hash === "#dashboard" ? "dashboard" : "chat";
}

export default function App() {
  const [tab, setTabState] = useState<Tab>(tabFromHash);

  useEffect(() => {
    const onHash = () => setTabState(tabFromHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const setTab = (t: Tab) => {
    window.location.hash = t === "dashboard" ? "#dashboard" : "";
    setTabState(t);
  };

  return (
    <div className="flex flex-col h-full">
      {/* ── Header ── */}
      <header className="flex items-center gap-6 px-5 py-3 border-b border-border bg-surface/80 backdrop-blur-sm flex-shrink-0">
        <div className="flex items-center gap-2">
          <span className="text-accent text-lg">◉</span>
          <span className="font-medium text-sm text-text tracking-tight">Engram</span>
          <span className="text-[10px] text-muted border border-border rounded px-1 py-0.5 ml-1">
            memory agent
          </span>
        </div>

        <nav className="flex gap-1">
          {(["chat", "dashboard"] as Tab[]).map((t) => (
            <button
              key={t}
              onClick={() => setTab(t)}
              className={`px-3 py-1.5 rounded-md text-xs font-medium transition-colors
                ${tab === t
                  ? "bg-accent/20 text-accent border border-accent/30"
                  : "text-muted hover:text-text hover:bg-border/30"
                }`}
            >
              {t === "chat" ? "Chat" : "Dashboard"}
            </button>
          ))}
        </nav>

        <div className="ml-auto flex items-center gap-2">
          <span className="text-[10px] text-muted font-mono">user: demo</span>
          <span className="w-1.5 h-1.5 rounded-full bg-success" />
        </div>
      </header>

      {/* ── Body ── */}
      <main className="flex flex-1 min-h-0">
        {tab === "chat" ? <ChatView /> : <DashboardView />}
      </main>
    </div>
  );
}
