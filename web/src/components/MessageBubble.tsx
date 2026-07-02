import { ContextManifest } from "./ContextManifest.tsx";
import type { ChatMessage } from "../types.ts";

interface Props {
  message: ChatMessage;
  /** Budget in force when this message was produced (for the provenance bar). */
  budget?: number;
}

export function MessageBubble({ message, budget }: Props) {
  const isUser = message.role === "user";

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"} animate-fade-in`}>
      <div className={`max-w-[80%] ${isUser ? "items-end" : "items-start"} flex flex-col gap-1`}>
        {/* role label */}
        <span className="text-[10px] text-muted uppercase tracking-widest px-1">
          {isUser ? "you" : "engram"}
        </span>

        {/* bubble */}
        <div
          className={`rounded-xl px-4 py-2.5 text-sm leading-relaxed whitespace-pre-wrap
            ${isUser
              ? "bg-accent/20 border border-accent/30 text-text"
              : "bg-surface border border-border text-text"
            }`}
        >
          {message.content || (message.streaming ? "" : <span className="text-muted italic">—</span>)}
          {message.streaming && !message.content && (
            <span className="typing-cursor" />
          )}
          {message.streaming && message.content && (
            <span className="typing-cursor" />
          )}
        </div>

        {/* context manifest (only on assistant messages, once we have a manifest) */}
        {!isUser && message.manifest && message.manifest.length >= 0 && !message.streaming && (
          <ContextManifest manifest={message.manifest} usage={message.usage} budget={budget} />
        )}
      </div>
    </div>
  );
}
