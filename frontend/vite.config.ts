import path from "node:path";
import { fileURLToPath } from "node:url";
import type { IncomingMessage, ServerResponse } from "node:http";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig, loadEnv, type Plugin } from "vite";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const repoRoot = path.resolve(__dirname, "..");

const DEV_HOST = "127.0.0.1";
const DEFAULT_DEV_PORT = 5173;

/** FRONTEND_PORT / DEV_PORT override for isolated manual envs (CSRF origins follow). */
function resolveDevPort(): number {
  const raw = process.env.FRONTEND_PORT || process.env.DEV_PORT || String(DEFAULT_DEV_PORT);
  const n = Number(raw);
  if (!Number.isInteger(n) || n < 1 || n > 65535) {
    throw new Error(`Invalid FRONTEND_PORT/DEV_PORT: ${JSON.stringify(raw)}`);
  }
  return n;
}

/** Custom non-simple header required on decision routes (CSRF preflight). */
export const OPERATOR_CSRF_HEADER = "x-casepilot-operator";
export const OPERATOR_CSRF_VALUE = "1";

/** Inject operator token only on these exact backend paths (after /api strip). */
const OPERATOR_PATH =
  /^\/v1\/proposals\/[^/]+\/(approve|reject)(?:\?|$)/;

function stripApiPrefix(url: string): string {
  const pathOnly = url.split("?")[0] ?? url;
  const stripped = pathOnly.replace(/^\/api/, "") || "/";
  return stripped.startsWith("/") ? stripped : `/${stripped}`;
}

function isLoopbackHost(hostname: string): boolean {
  const h = hostname.toLowerCase().replace(/^\[|\]$/g, "");
  return h === "127.0.0.1" || h === "localhost" || h === "::1";
}

function isLoopbackClient(remote: string | undefined): boolean {
  if (!remote) return false;
  const r = remote.replace(/^\[|\]$/g, "");
  if (r === "127.0.0.1" || r === "::1" || r === "localhost") return true;
  if (r.startsWith("127.")) return true;
  if (r.startsWith("::ffff:127.")) return true;
  return false;
}

/** BACKEND_BASE_URL must be loopback-only http(s) — prevents token exfiltration. */
function resolveBackendBaseUrl(raw: string): string {
  let parsed: URL;
  try {
    parsed = new URL(raw);
  } catch {
    throw new Error(
      `BACKEND_BASE_URL is not a valid URL: ${JSON.stringify(raw)}`,
    );
  }
  if (parsed.protocol !== "http:" && parsed.protocol !== "https:") {
    throw new Error(
      `BACKEND_BASE_URL must be http(s); got protocol ${parsed.protocol}`,
    );
  }
  if (!isLoopbackHost(parsed.hostname)) {
    throw new Error(
      `BACKEND_BASE_URL must target loopback (127.0.0.1/localhost); got ${parsed.hostname}`,
    );
  }
  if (parsed.username || parsed.password) {
    throw new Error("BACKEND_BASE_URL must not include credentials");
  }
  return raw.replace(/\/$/, "");
}

function rejectProxy(
  res: ServerResponse,
  status: number,
  detail: string,
  proxyReq: { destroy: () => void },
): void {
  try {
    proxyReq.destroy();
  } catch {
    /* ignore */
  }
  if (res.headersSent || res.writableEnded) return;
  res.statusCode = status;
  res.setHeader("Content-Type", "application/json");
  res.end(JSON.stringify({ detail }));
}

function headerValue(
  req: IncomingMessage,
  name: string,
): string | undefined {
  const raw = req.headers[name.toLowerCase()];
  if (Array.isArray(raw)) return raw[0];
  return raw;
}

export default defineConfig(({ mode }) => {
  // Root .env only — never VITE_ secrets; server-side proxy use only.
  const env = loadEnv(mode, repoRoot, "");
  const devPort = resolveDevPort();
  const allowedOrigins = new Set([
    `http://127.0.0.1:${devPort}`,
    `http://localhost:${devPort}`,
  ]);
  const backend = resolveBackendBaseUrl(
    env.BACKEND_BASE_URL || process.env.BACKEND_BASE_URL || "http://127.0.0.1:8000",
  );
  const operatorToken = env.OPERATOR_TOKEN || process.env.OPERATOR_TOKEN || "";

  const csrfGuard: Plugin = {
    name: "casepilot-decision-csrf-guard",
    configureServer(server) {
      // Fail closed before http-proxy for decision routes that lack CSRF markers.
      server.middlewares.use((req, res, next) => {
        const url = req.url ?? "";
        if (!url.startsWith("/api/")) {
          next();
          return;
        }
        const backendPath = stripApiPrefix(url);
        if (!OPERATOR_PATH.test(backendPath)) {
          next();
          return;
        }

        const origin = headerValue(req, "origin");
        const csrf = headerValue(req, OPERATOR_CSRF_HEADER);
        const clientOk = isLoopbackClient(req.socket?.remoteAddress);

        if (!clientOk) {
          res.statusCode = 403;
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ detail: "loopback client required" }));
          return;
        }
        if (!origin || !allowedOrigins.has(origin)) {
          res.statusCode = 403;
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ detail: "exact local origin required" }));
          return;
        }
        if (csrf !== OPERATOR_CSRF_VALUE) {
          res.statusCode = 403;
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ detail: "operator CSRF header required" }));
          return;
        }
        if (!operatorToken) {
          res.statusCode = 503;
          res.setHeader("Content-Type", "application/json");
          res.end(JSON.stringify({ detail: "operator token not configured" }));
          return;
        }
        next();
      });
    },
  };

  return {
    plugins: [react(), tailwindcss(), csrfGuard],
    resolve: {
      alias: {
        "@": path.resolve(__dirname, "./src"),
      },
    },
    server: {
      host: DEV_HOST,
      port: devPort,
      strictPort: true,
      proxy: {
        "/api": {
          target: backend,
          changeOrigin: false,
          rewrite: (p) => p.replace(/^\/api/, "") || "/",
          configure: (proxy) => {
            proxy.on("proxyReq", (proxyReq, req, res) => {
              // Never forward a browser-supplied operator credential.
              proxyReq.removeHeader("authorization");

              const outboundPath = stripApiPrefix(req.url ?? "");
              if (!OPERATOR_PATH.test(outboundPath)) {
                return;
              }

              // Defense in depth: re-check even after middleware (reject, don't skip).
              const origin = headerValue(req, "origin");
              const csrf = headerValue(req, OPERATOR_CSRF_HEADER);
              const clientOk = isLoopbackClient(req.socket?.remoteAddress);

              if (!clientOk) {
                rejectProxy(res, 403, "loopback client required", proxyReq);
                return;
              }
              if (!origin || !allowedOrigins.has(origin)) {
                rejectProxy(res, 403, "exact local origin required", proxyReq);
                return;
              }
              if (csrf !== OPERATOR_CSRF_VALUE) {
                rejectProxy(
                  res,
                  403,
                  "operator CSRF header required",
                  proxyReq,
                );
                return;
              }
              if (!operatorToken) {
                rejectProxy(
                  res,
                  503,
                  "operator token not configured",
                  proxyReq,
                );
                return;
              }

              proxyReq.setHeader(
                "Authorization",
                `Bearer ${operatorToken}`,
              );
            });
          },
        },
      },
    },
  };
});
