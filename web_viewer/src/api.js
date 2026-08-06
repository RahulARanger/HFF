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

export async function fetchFolders(path = "") {
  const query = path ? `?path=${encodeURIComponent(path)}` : "";
  return fetchJson(`/folders${query}`);
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

export function resourceSource(subjectId, resource, revision = 0) {
  const subject = subjectPath(subjectId);
  let endpoint;
  let name;

  if (resource.kind === "volume") {
    endpoint = `/subjects/${subject}/volumes/${encodeURIComponent(resource.modality)}/nifti`;
    name = `${resource.modality.toLowerCase()}.nii.gz`;
  } else if (resource.kind === "frequency") {
    endpoint = `/subjects/${subject}/frequency/${encodeURIComponent(resource.modality)}/${encodeURIComponent(resource.band)}/nifti`;
    name = `${resource.modality.toLowerCase()}_${resource.band.toLowerCase()}.nii.gz`;
  } else if (resource.kind === "mask") {
    endpoint = `/subjects/${subject}/masks/${encodeURIComponent(resource.maskKind)}/nifti`;
    name = `${resource.maskKind}.nii.gz`;
  } else {
    throw new Error(`Unsupported viewer resource: ${resource.kind}`);
  }

  const query = new URLSearchParams();
  if (resource.maskKind === "output" && resource.checkpointId) {
    query.set("checkpoint_id", resource.checkpointId);
  }
  if (resource.maskKind === "output" && revision) query.set("revision", String(revision));
  const suffix = query.toString() ? `?${query.toString()}` : "";
  return { url: apiPath(`${endpoint}${suffix}`), name };
}
