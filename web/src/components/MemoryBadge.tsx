import type { MemoryType } from "../types.ts";

const TYPE_STYLES: Record<MemoryType, string> = {
  fact:       "bg-blue-900/40  text-blue-300  border-blue-700/50",
  preference: "bg-purple-900/40 text-purple-300 border-purple-700/50",
  decision:   "bg-orange-900/40 text-orange-300 border-orange-700/50",
  event:      "bg-teal-900/40  text-teal-300  border-teal-700/50",
  skill:      "bg-green-900/40 text-green-300 border-green-700/50",
  hypothesis: "bg-fuchsia-900/30 text-fuchsia-300 border-fuchsia-600/60 border-dashed",
  note:       "bg-zinc-800/60  text-zinc-400  border-zinc-700/50",
};

export function MemoryBadge({ type }: { type: MemoryType }) {
  return (
    <span
      className={`inline-flex items-center px-1.5 py-0.5 text-[10px] font-medium rounded border
        ${TYPE_STYLES[type] ?? TYPE_STYLES.note}`}
    >
      {type}
    </span>
  );
}
