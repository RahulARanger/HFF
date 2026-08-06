import { existsSync } from "node:fs";
import { spawn } from "node:child_process";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const viewerDirectory = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const projectDirectory = resolve(viewerDirectory, "..");
const pythonPath = existsSync(join(projectDirectory, ".venv", "bin", "python"))
  ? join(projectDirectory, ".venv", "bin", "python")
  : "python3";

const backend = spawn(
  pythonPath,
  [join(projectDirectory, "web_viewer_api.py")],
  { cwd: projectDirectory, stdio: "inherit" },
);
const frontend = spawn(
  "npm",
  ["run", "dev", "--", "--host", "127.0.0.1", "--port", "5173"],
  { cwd: viewerDirectory, stdio: "inherit" },
);

let shuttingDown = false;
const shutdown = (signal) => {
  if (shuttingDown) return;
  shuttingDown = true;
  backend.kill(signal);
  frontend.kill(signal);
};

process.on("SIGINT", () => shutdown("SIGINT"));
process.on("SIGTERM", () => shutdown("SIGTERM"));

backend.on("exit", (code) => {
  if (!shuttingDown) {
    console.error(`FastAPI exited with code ${code ?? "unknown"}.`);
    shutdown("SIGTERM");
    process.exitCode = code || 1;
  }
});

frontend.on("exit", (code) => {
  if (!shuttingDown) {
    console.error(`Vite exited with code ${code ?? "unknown"}.`);
    shutdown("SIGTERM");
    process.exitCode = code || 1;
  }
});

