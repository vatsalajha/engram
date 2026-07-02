import { useState, useCallback, useRef } from "react";
import { actStream } from "../api.ts";
import type { ChatMessage, ManifestEntry, UsageInfo, DonePayload } from "../types.ts";

export interface UseChatReturn {
  messages: ChatMessage[];
  streaming: boolean;
  tokensSaved: number;
  send: (input: string) => Promise<void>;
  clear: () => void;
}

/**
 * Chat state over POST /act SSE.
 * `budgetRef` is read at send-time so the budget dial applies to the NEXT
 * turn without re-creating the callback (and without stale closures).
 */
export function useChat(
  userId: string,
  sessionId: string,
  budgetRef?: React.MutableRefObject<number>,
): UseChatReturn {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streaming, setStreaming] = useState(false);
  const [tokensSaved, setTokensSaved] = useState(0);
  const assistantIdRef = useRef<string | null>(null);

  const send = useCallback(
    async (input: string) => {
      if (streaming) return;

      const userMsg: ChatMessage = {
        id:   crypto.randomUUID(),
        role: "user",
        content: input,
      };
      const assistantId = crypto.randomUUID();
      assistantIdRef.current = assistantId;

      setMessages((prev) => [
        ...prev,
        userMsg,
        { id: assistantId, role: "assistant", content: "", streaming: true },
      ]);
      setStreaming(true);

      try {
        await actStream(userId, sessionId, input, (event) => {
          if (event.type === "token") {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, content: m.content + event.data }
                  : m,
              ),
            );
          } else if (event.type === "manifest") {
            const manifest = event.data as ManifestEntry[];
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, manifest } : m,
              ),
            );
          } else if (event.type === "usage") {
            const usage = event.data as UsageInfo;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, usage } : m,
              ),
            );
            // Rough token savings: assume full history would be 2500 tokens
            const saved = Math.max(0, 2500 - (usage.contextTokens ?? 0));
            setTokensSaved((prev) => prev + saved);
          } else if (event.type === "done") {
            const done = event.data as DonePayload;
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId ? { ...m, turnId: done.turnId } : m,
              ),
            );
          } else if (event.type === "error") {
            setMessages((prev) =>
              prev.map((m) =>
                m.id === assistantId
                  ? { ...m, content: `[Error: ${event.data.error}]`, streaming: false }
                  : m,
              ),
            );
          }
        }, budgetRef?.current);
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId
              ? { ...m, content: `[Error: ${msg}]`, streaming: false }
              : m,
          ),
        );
      } finally {
        setMessages((prev) =>
          prev.map((m) =>
            m.id === assistantId ? { ...m, streaming: false } : m,
          ),
        );
        setStreaming(false);
        assistantIdRef.current = null;
      }
    },
    [streaming, userId, sessionId, budgetRef],
  );

  const clear = useCallback(() => {
    setMessages([]);
    setTokensSaved(0);
  }, []);

  return { messages, streaming, tokensSaved, send, clear };
}
