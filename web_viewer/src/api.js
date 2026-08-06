const apiPath = (path) => `/api${path}`;
const REQUEST_TIMEOUT_MS = 30000;

const subjectPath = (subjectId) =>
  subjectId
    .split("/")
    .map((part) => encodeURIComponent(part))
    .join("/");

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

export async function fetchSubjects() {
  return fetchJson("/subjects");
}

export async function fetchMetadata(subjectId) {
  return fetchJson(`/subjects/${subjectPath(subjectId)}/metadata`);
}

export async function fetchCheckpoints() {
  return fetchJson("/checkpoints");
}

export async function fetchMonitorRuns() {
  return fetchJson("/monitor/runs");
}

export async function fetchMonitorRun(runId, limit = 240) {
  return fetchJson(`/monitor/runs/${runId.split("/").map(encodeURIComponent).join("/")}?limit=${limit}`);
}

export async function generateOutput(subjectId, checkpointId) {
  return fetchJson(`/subjects/${subjectPath(subjectId)}/generate`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ checkpoint_id: checkpointId }),
  });
}

function parseNumberList(header) {
  return header
    .split(",")
    .map((value) => Number(value.trim()))
    .filter((value) => Number.isFinite(value));
}

export async function fetchBinaryVolume(path) {
  const response = await fetchResponse(path);
  const shape = parseNumberList(response.headers.get("x-shape") || "");
  const spacing = parseNumberList(response.headers.get("x-spacing") || "1,1,1");
  const intensityRange = parseNumberList(
    response.headers.get("x-intensity-range") || "0,1",
  );
  const dtype = response.headers.get("x-dtype") || "float32";
  const buffer = await response.arrayBuffer();
  const values = dtype === "uint8" ? new Uint8Array(buffer) : new Float32Array(buffer);
  return { values, shape, spacing, intensityRange, dtype };
}

export async function fetchVolume(subjectId, modality) {
  return fetchBinaryVolume(`/subjects/${subjectPath(subjectId)}/volumes/${modality}`);
}

export async function fetchFrequency(subjectId, modality, band) {
  return fetchBinaryVolume(
    `/subjects/${subjectPath(subjectId)}/frequency/${modality}/${band}`,
  );
}

export async function fetchMask(subjectId, maskKind, checkpointId) {
  const query = checkpointId ? `?checkpoint_id=${encodeURIComponent(checkpointId)}` : "";
  return fetchBinaryVolume(
    `/subjects/${subjectPath(subjectId)}/masks/${maskKind}${query}`,
  );
}
