import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";

const ViewerPane = lazy(() => import("./components/ViewerPane.jsx"));
import MonitorView from "./components/MonitorView.jsx";
import {
  fetchCheckpoints,
  fetchFrequency,
  fetchMask,
  fetchMetadata,
  fetchSubjects,
  fetchVolume,
  generateOutput,
} from "./api.js";

const MODALITIES = ["FLAIR", "T1", "T1CE", "T2"];
const MODE_OPTIONS = [
  { id: "input", label: "Input analysis" },
  { id: "frequency", label: "Frequency decomposition" },
  { id: "output", label: "Output analysis" },
  { id: "monitor", label: "Training monitor" },
];
const AXIS_OPTIONS = [
  { id: "axial", label: "Axial" },
  { id: "coronal", label: "Coronal" },
  { id: "sagittal", label: "Sagittal" },
];

function resourceKey(resource) {
  return [resource.kind, resource.modality, resource.band || "", resource.maskKind || "", resource.checkpointId || ""].join(":");
}

function buildPanes(mode, selectedScan, frequencyBand) {
  if (mode === "monitor") return [];
  if (mode === "input") {
    return [
      ...MODALITIES.map((modality) => ({ id: modality, title: modality, volume: { kind: "volume", modality } })),
      {
        id: "selected-expected",
        title: `${selectedScan} + EXPECTED`,
        volume: { kind: "volume", modality: selectedScan },
        mask: { kind: "mask", maskKind: "expected" },
      },
      { id: "expected", title: "EXPECTED MASK", maskOnly: { kind: "mask", maskKind: "expected" } },
    ];
  }

  if (mode === "frequency") {
    return [
      { id: "actual", title: `${selectedScan} · actual`, volume: { kind: "volume", modality: selectedScan } },
      { id: "low", title: `${selectedScan} · low frequency`, volume: { kind: "frequency", modality: selectedScan, band: "L" } },
      {
        id: "low-expected",
        title: `${selectedScan} · low + EXPECTED`,
        volume: { kind: "frequency", modality: selectedScan, band: "L" },
        mask: { kind: "mask", maskKind: "expected" },
      },
      { id: "actual-high", title: `${selectedScan} · actual (${frequencyBand})`, volume: { kind: "volume", modality: selectedScan } },
      { id: "high", title: `${selectedScan} · ${frequencyBand} high frequency`, volume: { kind: "frequency", modality: selectedScan, band: frequencyBand } },
      {
        id: "high-expected",
        title: `${selectedScan} · ${frequencyBand} + EXPECTED`,
        volume: { kind: "frequency", modality: selectedScan, band: frequencyBand },
        mask: { kind: "mask", maskKind: "expected" },
      },
    ];
  }

  return [
    { id: "output-input", title: `${selectedScan} · input scan`, volume: { kind: "volume", modality: selectedScan } },
    {
      id: "output-expected-overlay",
      title: `${selectedScan} + EXPECTED`,
      volume: { kind: "volume", modality: selectedScan },
      mask: { kind: "mask", maskKind: "expected" },
    },
    {
      id: "output-prediction-overlay",
      title: `${selectedScan} + OUTPUT`,
      volume: { kind: "volume", modality: selectedScan },
      mask: { kind: "mask", maskKind: "output" },
    },
    { id: "expected-segmentation", title: "EXPECTED segmentation", maskOnly: { kind: "mask", maskKind: "expected" } },
    { id: "output-segmentation", title: "OUTPUT segmentation", maskOnly: { kind: "mask", maskKind: "output" } },
    { id: "output-summary", title: "SEGMENTATION comparison", volume: { kind: "volume", modality: selectedScan }, mask: { kind: "mask", maskKind: "output" } },
  ];
}

function SelectField({ label, value, onChange, options, disabled = false }) {
  return (
    <label className="control-field">
      <span className="control-label">{label}</span>
      <select value={value} onChange={(event) => onChange(event.target.value)} disabled={disabled}>
        {options.map((option) => (
          <option key={option.value ?? option.id} value={option.value ?? option.id}>
            {option.label}
          </option>
        ))}
      </select>
    </label>
  );
}

function StatusDot({ state }) {
  return <span className={`status-dot ${state}`} aria-hidden="true" />;
}

export default function App() {
  const [subjects, setSubjects] = useState([]);
  const [subjectId, setSubjectId] = useState("");
  const [metadata, setMetadata] = useState(null);
  const [checkpoints, setCheckpoints] = useState([]);
  const [mode, setMode] = useState("input");
  const [selectedScan, setSelectedScan] = useState("FLAIR");
  const [frequencyBand, setFrequencyBand] = useState("H1");
  const [checkpointId, setCheckpointId] = useState("");
  const [renderingMode, setRenderingMode] = useState("volume");
  const [volumeMode, setVolumeMode] = useState("MIP");
  const [sliceAxis, setSliceAxis] = useState("axial");
  const [sliceIndex, setSliceIndex] = useState(0);
  const [resources, setResources] = useState({});
  const [loadingResources, setLoadingResources] = useState(false);
  const [resourceErrors, setResourceErrors] = useState({});
  const [loadingSubjects, setLoadingSubjects] = useState(true);
  const [appError, setAppError] = useState("");
  const [checkpointError, setCheckpointError] = useState("");
  const [resourceProgress, setResourceProgress] = useState({ completed: 0, total: 0 });
  const [startupAttempt, setStartupAttempt] = useState(0);
  const [generating, setGenerating] = useState(false);
  const cacheRef = useRef(new Map());

  useEffect(() => {
    let cancelled = false;
    setLoadingSubjects(true);
    setAppError("");
    setCheckpointError("");

    fetchSubjects()
      .then((subjectPayload) => {
        if (cancelled) return;
        const nextSubjects = subjectPayload.subjects || [];
        setSubjects(nextSubjects);
        setSubjectId((current) => current || nextSubjects[0]?.id || "");
        setAppError("");
      })
      .catch((error) => setAppError(error.message))
      .finally(() => setLoadingSubjects(false));

    fetchCheckpoints()
      .then((checkpointPayload) => {
        if (cancelled) return;
        const nextCheckpoints = checkpointPayload.checkpoints || [];
        setCheckpoints(nextCheckpoints);
        setCheckpointId((current) => current || nextCheckpoints[0]?.id || "");
      })
      .catch((error) => {
        if (!cancelled) setCheckpointError(error.message);
      });

    return () => {
      cancelled = true;
    };
  }, [startupAttempt]);

  useEffect(() => {
    if (!subjectId) return;
    setMetadata(null);
    fetchMetadata(subjectId)
      .then((payload) => {
        setMetadata(payload);
        const available = payload.modalities || [];
        if (available.length && !available.includes(selectedScan)) setSelectedScan(available[0]);
        const shape = payload.scans?.[selectedScan]?.shape;
        if (shape) setSliceIndex(Math.floor(shape[0] / 2));
      })
      .catch((error) => setAppError(error.message));
  }, [subjectId]);

  const panes = useMemo(() => buildPanes(mode, selectedScan, frequencyBand), [mode, selectedScan, frequencyBand]);

  const loadResource = useCallback(async (resource) => {
    const key = resourceKey(resource);
    if (cacheRef.current.has(`${subjectId}:${key}`)) return cacheRef.current.get(`${subjectId}:${key}`);
    let loaded;
    if (resource.kind === "volume") loaded = await fetchVolume(subjectId, resource.modality);
    if (resource.kind === "frequency") loaded = await fetchFrequency(subjectId, resource.modality, resource.band);
    if (resource.kind === "mask") loaded = await fetchMask(subjectId, resource.maskKind, resource.checkpointId);
    cacheRef.current.set(`${subjectId}:${key}`, loaded);
    return loaded;
  }, [subjectId]);

  useEffect(() => {
    if (!subjectId || !panes.length) return;
    const allResources = panes.flatMap((pane) => [pane.volume, pane.mask, pane.maskOnly]).filter(Boolean).map((resource) => ({ ...resource, checkpointId: resource.maskKind === "output" ? checkpointId : undefined }));
    const uniqueResources = Array.from(new Map(allResources.map((resource) => [resourceKey(resource), resource])).values());
    let cancelled = false;
    setLoadingResources(true);
    setResourceProgress({ completed: 0, total: uniqueResources.length });
    setResourceErrors({});
    const loadOneResource = async (resource) => {
      const key = resourceKey(resource);
      try {
        const loaded = await loadResource(resource);
        if (!cancelled) setResources((current) => ({ ...current, [key]: loaded }));
      } catch (error) {
        if (!cancelled) setResourceErrors((current) => ({ ...current, [key]: error.message }));
      } finally {
        if (!cancelled) setResourceProgress((current) => ({ ...current, completed: current.completed + 1 }));
      }
    };

    Promise.all(uniqueResources.map(loadOneResource)).finally(() => {
      if (!cancelled) setLoadingResources(false);
    });
    return () => {
      cancelled = true;
    };
  }, [subjectId, panes, checkpointId, loadResource]);

  const resourceFor = useCallback((resource) => {
    if (!resource) return null;
    const actualResource = { ...resource, checkpointId: resource.maskKind === "output" ? checkpointId : undefined };
    return resources[resourceKey(actualResource)] || null;
  }, [resources, checkpointId]);

  const sliceMaximum = metadata?.scans?.[selectedScan]?.shape?.[{ axial: 0, coronal: 1, sagittal: 2 }[sliceAxis]] - 1 || 0;
  useEffect(() => {
    setSliceIndex((current) => Math.max(0, Math.min(current, sliceMaximum)));
  }, [sliceMaximum]);

  const handleGenerate = async () => {
    if (!subjectId || !checkpointId) return;
    setGenerating(true);
    setAppError("");
    try {
      await generateOutput(subjectId, checkpointId);
      const outputResource = { kind: "mask", maskKind: "output", checkpointId };
      const output = await fetchMask(subjectId, "output", checkpointId);
      cacheRef.current.set(`${subjectId}:${resourceKey(outputResource)}`, output);
      setResources((current) => ({ ...current, [resourceKey(outputResource)]: output }));
    } catch (error) {
      setAppError(error.message);
    } finally {
      setGenerating(false);
    }
  };

  const selectedSubject = subjects.find((subject) => subject.id === subjectId);
  const status = appError ? "error" : loadingSubjects || loadingResources ? "loading" : "ready";
  const statusLabel = appError
    ? "Viewer error"
    : loadingSubjects
      ? "Loading dataset catalog"
      : loadingResources
        ? `Loading views ${resourceProgress.completed}/${resourceProgress.total}`
        : "Viewer ready";

  const retryStartup = () => {
    setSubjects([]);
    setSubjectId("");
    setMetadata(null);
    setResources({});
    cacheRef.current.clear();
    setStartupAttempt((attempt) => attempt + 1);
  };

  return (
    <main className="app-shell">
      <aside className="control-rail">
        <div className="brand-block">
          <div className="brand-mark">H</div>
          <div>
            <div className="brand-name">HFF-Net</div>
            <div className="brand-subtitle">BraTS viewer</div>
          </div>
        </div>

        <div className="rail-scroll">
          <section className="rail-section mode-section">
            <div className="section-kicker">View type</div>
            <div className="mode-list">
              {MODE_OPTIONS.map((option) => (
                <button key={option.id} className={`mode-button ${mode === option.id ? "active" : ""}`} onClick={() => setMode(option.id)}>
                  <span className="mode-bar" />
                  {option.label}
                </button>
              ))}
            </div>
          </section>

          {mode !== "monitor" && <>
            <section className="rail-section">
              <SelectField label="Select record" value={subjectId} onChange={setSubjectId} options={subjects.map((subject) => ({ value: subject.id, label: subject.label }))} disabled={!subjects.length} />
              <div className="selected-path">
                {selectedSubject?.id || (loadingSubjects ? "Loading dataset catalog…" : "Dataset unavailable")}
              </div>
            </section>

            <section className="rail-section">
              <SelectField label="Actual scan" value={selectedScan} onChange={setSelectedScan} options={(metadata?.modalities || MODALITIES).map((modality) => ({ value: modality, label: modality }))} disabled={!metadata} />
              {mode === "frequency" && <SelectField label="High-frequency band" value={frequencyBand} onChange={setFrequencyBand} options={["H1", "H2", "H3", "H4"].map((band) => ({ value: band, label: band }))} disabled={!metadata} />}
            </section>

            {mode === "output" && (
              <section className="rail-section">
                <SelectField label="Checkpoint" value={checkpointId} onChange={setCheckpointId} options={checkpoints.map((checkpoint) => ({ value: checkpoint.id, label: checkpoint.label }))} disabled={!checkpoints.length} />
                {checkpointError && <div className="inline-error">Checkpoint list unavailable: {checkpointError}</div>}
                <button className="primary-action" onClick={handleGenerate} disabled={generating || !checkpointId || !subjectId}>
                  <span className="action-icon">{generating ? "…" : "✦"}</span>
                  {generating ? "Generating output…" : "Generate output segmentation"}
                </button>
              </section>
            )}
          </>}

          {mode === "monitor" && <section className="rail-section monitor-rail-note">
            <div className="section-kicker">Training telemetry</div>
            <p>Process-scoped RAM and accelerator samples from cross-validation folds.</p>
            <div className="selected-path">Logs refresh automatically while a fold is running.</div>
          </section>}

          <section className="rail-section legend-section">
            <div className="section-kicker">Segmentation labels</div>
            <div className="legend-row"><span className="legend-swatch core" /> Necrotic / core</div>
            <div className="legend-row"><span className="legend-swatch edema" /> Edema</div>
            <div className="legend-row"><span className="legend-swatch enhancing" /> Enhancing tumour</div>
          </section>
        </div>

        <div className="rail-footer">
          <StatusDot state={status} />
          <span>{statusLabel}</span>
        </div>
      </aside>

      <section className={`workspace ${mode === "monitor" ? "monitor-workspace" : ""}`}>
        <header className="topbar">
          <div className="breadcrumb"><span>Research workspace</span><span className="crumb-divider">/</span><strong>{mode === "input" ? "Input analysis" : mode === "frequency" ? "Frequency decomposition" : mode === "monitor" ? "Training monitor" : "Output analysis"}</strong></div>
          {mode !== "monitor" && <div className="topbar-actions">
            <label className="compact-control"><span>Render</span><select value={renderingMode} onChange={(event) => setRenderingMode(event.target.value)}><option value="volume">3D volume</option><option value="slice">2D slice</option></select></label>
            <label className="compact-control"><span>Blend</span><select value={volumeMode} onChange={(event) => setVolumeMode(event.target.value)} disabled={renderingMode === "slice"}><option value="MIP">MIP</option><option value="Composite">Composite</option></select></label>
          </div>}
        </header>

        {mode === "monitor" ? <MonitorView /> : <>
        <div className="viewer-toolbar">
          <div className="toolbar-group"><span className="toolbar-label">Slice navigation</span><div className="axis-buttons">{AXIS_OPTIONS.map((axis) => <button key={axis.id} className={sliceAxis === axis.id ? "active" : ""} onClick={() => setSliceAxis(axis.id)}>{axis.label}</button>)}</div></div>
          <div className="slice-control"><span className="slice-value">{sliceIndex + 1}<span>/</span>{sliceMaximum + 1}</span><input type="range" min="0" max={sliceMaximum} value={sliceIndex} onChange={(event) => setSliceIndex(Number(event.target.value))} disabled={renderingMode !== "slice" || !metadata} /></div>
          <button className="reset-button" onClick={() => setSliceIndex(Math.floor(sliceMaximum / 2))}>Reset view</button>
        </div>

        {appError && (
          <div className="app-alert">
            <span>{appError}</span>
            <button className="alert-action" onClick={retryStartup}>Retry connection</button>
          </div>
        )}
        <div className="pane-grid">
          {panes.map((pane) => {
            const volume = resourceFor(pane.volume);
            const mask = resourceFor(pane.mask || pane.maskOnly);
            const resourceError = resourceErrors[resourceKey({ ...(pane.mask || pane.maskOnly), checkpointId: (pane.mask || pane.maskOnly)?.maskKind === "output" ? checkpointId : undefined })];
            return (
              <Suspense key={pane.id} fallback={<section className="viewer-pane pane-suspense"><div className="pane-loading"><span className="spinner" /> Loading renderer</div></section>}>
                <ViewerPane title={pane.title} volume={volume} mask={mask} renderingMode={renderingMode} sliceAxis={sliceAxis} sliceIndex={sliceIndex} loading={loadingResources && !volume && !mask} error={resourceError} />
              </Suspense>
            );
          })}
        </div>

        <footer className="status-bar">
          <span>Subject: <strong>{selectedSubject?.label || "—"}</strong></span>
          <span>Scan: <strong>{selectedScan}</strong></span>
          <span>Dimensions: <strong>{metadata?.scans?.[selectedScan]?.shape?.slice().reverse().join(" × ") || "—"}</strong></span>
          <span>Voxel: <strong>{metadata?.scans?.[selectedScan]?.spacing?.map((value) => value.toFixed(1)).join(" × ") || "—"} mm</strong></span>
          <span>Rendering: <strong>{renderingMode === "slice" ? "2D slice" : volumeMode}</strong></span>
          <span>Model: <strong>HFF-Net</strong></span>
          <span className="status-ready"><StatusDot state={status} /> {status === "ready" ? "Ready" : status === "loading" ? "Loading" : "Error"}</span>
        </footer>
        </>}
      </section>
    </main>
  );
}
