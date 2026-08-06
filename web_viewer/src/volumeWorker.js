function parseNumberList(header) {
  return header
    .split(",")
    .map((value) => Number(value.trim()))
    .filter((value) => Number.isFinite(value));
}

self.onmessage = async (event) => {
  const { id, path } = event.data;
  try {
    const response = await fetch(path);
    if (!response.ok) {
      let detail = `${response.status} ${response.statusText}`;
      try {
        const payload = await response.json();
        detail = payload.detail || detail;
      } catch {
        // Keep the HTTP status when the API did not return JSON.
      }
      throw new Error(detail);
    }

    const shape = parseNumberList(response.headers.get("x-shape") || "");
    const spacing = parseNumberList(response.headers.get("x-spacing") || "1,1,1");
    const intensityRange = parseNumberList(
      response.headers.get("x-intensity-range") || "0,1",
    );
    const dtype = response.headers.get("x-dtype") || "float32";
    const buffer = await response.arrayBuffer();
    const values = dtype === "uint8" ? new Uint8Array(buffer) : new Float32Array(buffer);

    self.postMessage(
      { id, values, shape, spacing, intensityRange, dtype },
      [values.buffer],
    );
  } catch (error) {
    self.postMessage({ id, error: error instanceof Error ? error.message : String(error) });
  }
};
