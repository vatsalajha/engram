import { useRef, useEffect, useState, type FormEvent } from "react";
import { useChat } from "../hooks/useChat.ts";
import { MessageBubble } from "../components/MessageBubble.tsx";
import { MemoryPanel } from "../components/MemoryPanel.tsx";
import { BudgetDial } from "../components/BudgetDial.tsx";
import { postSleep } from "../api.ts";
import type { SleepResult } from "../types.ts";

const USER_ID = "demo";
const SESSION_ID = `session-${Date.now()}`;
const DEFAULT_BUDGET = 1500;
/** How long sleep-born memories stay highlighted in the panel. */
const SLEEP_HIGHLIGHT_MS = 12_000;

const EXAMPLE_TURNS = [
  "I'm building a side project called Artmoji — a React app that turns selfies into emoji mosaics.",
  "I switched the database from Postgres to SQLite last week.",
  "What database should my new endpoint use?",
  "I always prefer concise, bullet-pointed responses.",
];

export function ChatView() {
  const [budget, setBudget] = useState(DEFAULT_BUDGET);
  const budgetRef = useRef(DEFAULT_BUDGET);
  budgetRef.current = budget;

  const { messages, streaming, tokensSaved, send, clear } = useChat(USER_ID, SESSION_ID, budgetRef);
  const [input, setInput] = useState("");
  const [sleeping, setSleeping] = useState(false);
  const [sleepNote, setSleepNote] = useState<string | null>(null);
  const [highlightSince, setHighlightSince] = useState<number | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    const text = input.trim();
    if (!text || streaming) return;
    setInput("");
    await send(text);
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      void handleSubmit(e as unknown as FormEvent);
    }
  };

  const handleSleep = async () => {
    if (sleeping) return;
    setSleeping(true);
    setSleepNote(null);
    const sleepStart = Date.now();
    try {
      const result: SleepResult = await postSleep(USER_ID);
      if (result.skipped) {
        setSleepNote("already sleeping…");
      } else {
        setSleepNote(
          `consolidated ${result.consolidated_n} · decayed ${result.decayed_n} · dreamed ${result.inferred_n}`,
        );
        if (result.consolidated_n > 0 || result.inferred_n > 0) {
          setHighlightSince(sleepStart);
          setTimeout(() => setHighlightSince(null), SLEEP_HIGHLIGHT_MS);
        }
      }
    } catch (err) {
      setSleepNote(`sleep failed: ${err instanceof Error ? err.message : String(err)}`);
    } finally {
      setSleeping(false);
      setTimeout(() => setSleepNote(null), 8000);
    }
  };

  return (
    <div className="flex flex-1 min-h-0">
      {/* ── Main chat area ── */}
      <div className="flex flex-col flex-1 min-w-0">
        {/* control bar: budget dial · cost · sleep */}
        <div className="flex items-center gap-4 px-4 py-2 border-b border-border bg-surface/50 text-xs text-muted">
          <BudgetDial budget={budget} onChange={setBudget} disabled={streaming} />
          <span className="text-muted/40 hidden md:inline">·</span>
          <span className="hidden md:inline">
            <span className="text-success font-mono">{tokensSaved.toLocaleString()}</span>
            {" "}tok saved
          </span>
          <div className="ml-auto flex items-center gap-3">
            {sleepNote && (
              <span className="font-mono text-[10px] text-fuchsia-300/90 animate-fade-in">
                {sleepNote}
              </span>
            )}
            <button
              onClick={() => void handleSleep()}
              disabled={sleeping}
              className="px-2.5 py-1 rounded-md border border-fuchsia-500/40 text-fuchsia-300
                         hover:bg-fuchsia-500/10 transition-colors disabled:opacity-40 text-[11px]"
              title="Run the sleep cycle: consolidate · decay · infer"
            >
              {sleeping ? "dreaming…" : "☾ Sleep now"}
            </button>
            <button
              onClick={clear}
              className="text-muted/50 hover:text-danger transition-colors"
            >
              clear
            </button>
          </div>
        </div>

        {/* messages */}
        <div className="flex-1 overflow-y-auto px-4 py-6 space-y-6">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full gap-6 text-center">
              <div>
                <h2 className="text-lg font-medium text-text mb-1">Engram</h2>
                <p className="text-sm text-muted max-w-sm">
                  Your project copilot with a hippocampus: it remembers you across
                  sessions, updates its beliefs, dreams between conversations, and
                  forgets what you've changed your mind about.
                </p>
              </div>
              <div className="grid grid-cols-1 gap-2 w-full max-w-md">
                {EXAMPLE_TURNS.map((t) => (
                  <button
                    key={t}
                    onClick={() => void send(t)}
                    className="text-left px-3 py-2 rounded-lg border border-border bg-surface
                               text-sm text-muted hover:text-text hover:border-accent/50 transition-colors"
                  >
                    {t}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            messages.map((m) => <MessageBubble key={m.id} message={m} budget={budget} />)
          )}
          <div ref={bottomRef} />
        </div>

        {/* input */}
        <form
          onSubmit={(e) => { void handleSubmit(e); }}
          className="px-4 py-3 border-t border-border bg-surface/30"
        >
          <div className="flex items-end gap-2 rounded-xl border border-border bg-surface
                          focus-within:border-accent/60 transition-colors p-2">
            <textarea
              ref={textareaRef}
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Message Engram…"
              disabled={streaming}
              className="flex-1 resize-none bg-transparent text-sm text-text placeholder-muted
                         focus:outline-none min-h-[1.5rem] max-h-32 py-1 px-1"
              style={{ height: "auto" }}
              onInput={(e) => {
                const el = e.currentTarget;
                el.style.height = "auto";
                el.style.height = `${el.scrollHeight}px`;
              }}
            />
            <button
              type="submit"
              disabled={!input.trim() || streaming}
              className="flex-shrink-0 px-3 py-1.5 rounded-lg bg-accent text-canvas text-sm font-medium
                         disabled:opacity-40 disabled:cursor-not-allowed hover:bg-accent/90 transition-colors"
            >
              {streaming ? "…" : "Send"}
            </button>
          </div>
          <p className="mt-1.5 text-[10px] text-muted/50 text-center">
            Enter to send · Shift+Enter for newline · answers use ≤{budget} context tokens
          </p>
        </form>
      </div>

      {/* ── Memory side panel ── */}
      <MemoryPanel userId={USER_ID} highlightSince={highlightSince} />
    </div>
  );
}
