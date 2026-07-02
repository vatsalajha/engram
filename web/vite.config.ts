import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      "/act":          { target: "http://localhost:3000", changeOrigin: true },
      "/ingest":       { target: "http://localhost:3000", changeOrigin: true },
      "/recall":       { target: "http://localhost:3000", changeOrigin: true },
      "/feedback":     { target: "http://localhost:3000", changeOrigin: true },
      "/sleep":        { target: "http://localhost:3000", changeOrigin: true },
      "/memories":     { target: "http://localhost:3000", changeOrigin: true },
      "/health":       { target: "http://localhost:3000", changeOrigin: true },
      "/admin":        { target: "http://localhost:3000", changeOrigin: true },
      "/api":          { target: "http://localhost:3000", changeOrigin: true },
      "/mcp":          { target: "http://localhost:3000", changeOrigin: true },
    },
  },
});
