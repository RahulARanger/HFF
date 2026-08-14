const apiPath = (path) => `/api${path}`;
const REQUEST_TIMEOUT_MS = 30000;

async function parseResponse(response) {
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = await response.json();
      detail = payload.detail || detail;
    } catch {
      // Keep the HTTP status when the server did not return JSON.
    }
    throw new Error(detail);
  }
  return response;
}

async function fetchResponse(path, options = {}) {
  const controller = new AbortController();
  const timeoutId = setTimeout(() => controller.abort(), REQUEST_TIMEOUT_MS);

  try {
    return await parseResponse(
      await fetch(apiPath(path), { ...options, signal: controller.signal }),
    );
  } catch (error) {
    if (error.name === "AbortError") {
      throw new Error(
          `The viewer API did not respond within ${REQUEST_TIMEOUT_MS / 1000}s. ` +
          "Make sure the FastAPI server is running on port 8010.",
      );
    }
    throw error;
  } finally {
    clearTimeout(timeoutId);
  }
}

export async function fetchJson(path, options) {
  const response = await fetchResponse(path, options);
  return response.json();
}

export async function fetchMonitorRuns() {
  return fetchJson("/monitor/runs");
}

export async function fetchMonitorRun(runId, limit = 240) {
  return fetchJson(`/monitor/runs/${runId.split("/").map(encodeURIComponent).join("/")}?limit=${limit}`);
}

export async function fetchEvaluationOptions() {
  return fetchJson("/eval/options");
}

export async function fetchEvaluationJobs() {
  return fetchJson("/eval/jobs");
}

export async function fetchEvaluationJob(jobId) {
  return fetchJson(`/eval/jobs/${encodeURIComponent(jobId)}`);
}

export async function startEvaluation(request) {
  return fetchJson("/eval/jobs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(request),
  });
}

export async function fetchValidationRuns() {
  return fetchJson("/validation/runs");
}

export async function fetchValidationRun(runId) {
  return fetchJson(`/validation/runs/${encodeURIComponent(runId)}`);
}
