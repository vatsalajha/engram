import type { Config } from "tailwindcss";

export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        canvas:  "#0d1117",
        surface: "#161b22",
        border:  "#30363d",
        muted:   "#8b949e",
        text:    "#e6edf3",
        accent:  "#58a6ff",
        success: "#3fb950",
        warn:    "#d29922",
        danger:  "#f85149",
      },
      fontFamily: {
        mono: ["'JetBrains Mono'", "Menlo", "Monaco", "monospace"],
      },
    },
  },
  plugins: [],
} satisfies Config;
