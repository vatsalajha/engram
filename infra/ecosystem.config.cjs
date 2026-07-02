/**
 * pm2 ecosystem config for Engram.
 * Usage:
 *   pm2 start ecosystem.config.cjs
 *   pm2 save && pm2 startup
 *
 * The app loads .env itself via dotenv/config (see src/config.ts),
 * so just ensure /opt/engram/.env exists with all required variables.
 *
 * Do NOT use cluster mode — SQLite WAL handles concurrent reads but
 * write serialisation relies on a single-process SQLite connection.
 */

module.exports = {
  apps: [
    {
      name:         "engram",
      script:       "dist/api/server.js",
      cwd:          "/opt/engram",
      interpreter:  "node",
      instances:    1,
      exec_mode:    "fork",

      // Restart policy
      restart_delay: 5000,    // ms before restart after crash
      max_restarts:  10,
      min_uptime:    "10s",   // must stay up ≥10 s to count as stable
      watch:         false,

      // Log files — rotated by pm2-logrotate
      error_file:       "/var/log/engram/error.log",
      out_file:         "/var/log/engram/out.log",
      merge_logs:       true,
      log_date_format:  "YYYY-MM-DD HH:mm:ss Z",

      // NODE_ENV for production — dotenv still loads .env
      env: {
        NODE_ENV: "production",
      },
    },
  ],
};
