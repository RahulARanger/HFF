import { lazy, Suspense, useCallback, useDeferredValue, useEffect, useMemo, useRef, useState } from "react";

const ViewerPane = lazy(() => import("./components/ViewerPane.jsx"));
import MonitorView from "./components/MonitorView.jsx";
import {
  fetchCheckpoints,
  fetchFolders,
  fetchFrequency,
  fetchMask,
  fetchMetadata,
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
const MIN_2D_ZOOM = 1;
const MAX_2D_ZOOM = 6;

function resourceKey(resource) {
  return [resource.kind, resource.modality, resource.band || "", resource.maskKind || "", resource.checkpointId || ""].join(":");
}

function selectionErrorMessage(error) {
  return /not found|404|not a scan folder|missing .* scan/i.test(error?.message || "")
    ? "Please retry your selection."
    : error.message;
}

function scaleCameraPosition(cameraState, factor) {
  if (!cameraState) return cameraState;
  const position = cameraState.position.map((value, index) => (
    cameraState.focalPoint[index] + (value - cameraState.focalPoint[index]) * factor
  ));
  return { ...cameraState, position };
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

function FolderPicker({ value, onChange, disabled = false }) {
  const [open, setOpen] = useState(false);
  const [rootLabel, setRootLabel] = useState("Dataset");
  const [currentPath, setCurrentPath] = useState("");
  const [folders, setFolders] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [currentSelectable, setCurrentSelectable] = useState(false);

  const close = () => {
    setOpen(false);
  };

  const browse = useCallback(async (path) => {
    setLoading(true);
    setError("");
    try {
      const payload = await fetchFolders(path);
      setRootLabel(payload.root || "Dataset");
      setCurrentPath(payload.path || "");
      setFolders(payload.folders || []);
      setCurrentSelectable(Boolean(payload.selectable));
    } catch (browseError) {
      setError(browseError.message);
      setFolders([]);
    } finally {
      setLoading(false);
    }
  }, []);

  const openPicker = () => {
    setOpen(true);
    browse("");
  };

  const currentLabel = currentPath ? currentPath.split("/").at(-1) : rootLabel;
  const parentPath = currentPath.includes("/") ? currentPath.slice(0, currentPath.lastIndexOf("/")) : "";

  return (
    <>
      <span className="control-label">Select folder</span>
      <button className="folder-picker-trigger" type="button" onClick={openPicker} disabled={disabled}>
        <span>{value || "Choose a folder…"}</span>
        <span className="folder-picker-icon" aria-hidden="true">⌕</span>
      </button>
      {open && (
        <div className="folder-picker-backdrop" role="presentation" onMouseDown={(event) => event.target === event.currentTarget && close()}>
          <section className="folder-picker-dialog" role="dialog" aria-modal="true" aria-label="Browse scan folders">
            <div className="folder-picker-header">
              <div>
                <div className="section-kicker">Folder browser</div>
                <h2>{rootLabel}</h2>
              </div>
              <button className="folder-picker-close" type="button" onClick={close} aria-label="Close folder browser">×</button>
            </div>
            <div className="folder-picker-breadcrumb">
              <button type="button" onClick={() => browse("")} disabled={!currentPath}>Dataset root</button>
              {currentPath && <><span>/</span><strong>{currentPath}</strong></>}
            </div>
            <div className="folder-picker-actions">
              <button type="button" onClick={() => { onChange(currentPath || "."); close(); }} disabled={loading || !currentSelectable}>Use this scan folder</button>
              <button type="button" onClick={() => browse(parentPath)} disabled={!currentPath || loading}>Up one level</button>
            </div>
            {loading && <div className="folder-picker-count">Loading folders…</div>}
            {!loading && !currentSelectable && <div className="folder-picker-count">Choose a folder marked “scan folder” to load the viewer.</div>}
            {error && <div className="folder-picker-error">{error}</div>}
            <div className="folder-picker-list">
              {folders.map((folder) => (
                <button
                  className="folder-picker-option"
                  type="button"
                  key={folder.id}
                  onClick={() => browse(folder.path)}
                >
                  <span className="folder-picker-folder-icon" aria-hidden="true">▰</span>
                  <span>
                    <strong>{folder.label}</strong>
                    <small>{folder.path}{folder.selectable ? " · scan folder" : " · browse"}</small>
                  </span>
                </button>
              ))}
              {!loading && !error && !folders.length && <div className="folder-picker-empty">No subfolders found in {currentLabel}.</div>}
            </div>
          </section>
        </div>
      )}
    </>
  );
}

function StatusDot({ state }) {
  return <span className={`status-dot ${state}`} aria-hidden="true" />;
}

export default function App() {
  const [subjectId, setSubjectId] = useState("");
  const [selectedSubject, setSelectedSubject] = useState(null);
  const [metadata, setMetadata] = useState(null);
  const [checkpoints, setCheckpoints] = useState([]);
  const [mode, setMode] = useState("input");
  const [selectedScan, setSelectedScan] = useState("FLAIR");
  const [frequencyBand, setFrequencyBand] = useState("H1");
  const [checkpointId, setCheckpointId] = useState("");
  const [renderingMode, setRenderingMode] = useState("volume");
  const [volumeMode, setVolumeMode] = useState("MIP");
  const [zoom, setZoom] = useState(1);
  const [cameraState, setCameraState] = useState(null);
  const [sliceAxis, setSliceAxis] = useState("axial");
  const [sliceIndex, setSliceIndex] = useState(0);
  const [resources, setResources] = useState({});
  const [renderedPaneCount, setRenderedPaneCount] = useState(0);
  const [loadingResources, setLoadingResources] = useState(false);
  const [resourceErrors, setResourceErrors] = useState({});
  const [loadingSubjects, setLoadingSubjects] = useState(false);
  const [appError, setAppError] = useState("");
  const [checkpointError, setCheckpointError] = useState("");
  const [resourceProgress, setResourceProgress] = useState({ completed: 0, total: 0 });
  const [startupAttempt, setStartupAttempt] = useState(0);
  const [generating, setGenerating] = useState(false);
  const cacheRef = useRef(new Map());

  useEffect(() => {
    let cancelled = false;
    setAppError("");
    setCheckpointError("");

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
    setLoadingSubjects(true);
    fetchMetadata(subjectId)
      .then((payload) => {
        setMetadata(payload);
        const available = payload.modalities || [];
        if (available.length && !available.includes(selectedScan)) setSelectedScan(available[0]);
        const shape = payload.scans?.[selectedScan]?.shape;
        if (shape) setSliceIndex(Math.floor(shape[0] / 2));
      })
      .catch((error) => setAppError(selectionErrorMessage(error)))
      .finally(() => setLoadingSubjects(false));
  }, [subjectId]);

  // Keep the sidebar and workspace controls urgent while VTK tears down and
  // rebuilds the expensive multi-pane scene in the background.
  const deferredMode = useDeferredValue(mode);
  const panes = useMemo(() => buildPanes(deferredMode, selectedScan, frequencyBand), [deferredMode, selectedScan, frequencyBand]);

  useEffect(() => {
    setCameraState(null);
  }, [subjectId, mode, renderingMode]);

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
    if (!subjectId || !panes.length || !renderedPaneCount) return;
    const allResources = panes.flatMap((pane) => [pane.volume, pane.mask, pane.maskOnly]).filter(Boolean).map((resource) => ({ ...resource, checkpointId: resource.maskKind === "output" ? checkpointId : undefined }));
    const uniqueResources = Array.from(new Map(allResources.map((resource) => [resourceKey(resource), resource])).values());
    let cancelled = false;
    setLoadingResources(true);
    setResourceProgress({ completed: 0, total: uniqueResources.length });
    if (renderedPaneCount === 1) setResourceErrors({});
    const loadOneResource = async (resource) => {
      const key = resourceKey(resource);
      try {
        const loaded = await loadResource(resource);
        if (!cancelled) setResources((current) => ({ ...current, [key]: loaded }));
      } catch (error) {
        if (!cancelled) setResourceErrors((current) => ({ ...current, [key]: selectionErrorMessage(error) }));
      } finally {
        if (!cancelled) setResourceProgress((current) => ({ ...current, completed: current.completed + 1 }));
      }
    };

    (async () => {
      // Fetch and decode all resources in the worker pool while the complete
      // pane grid remains visible to the user.
      let nextResourceIndex = 0;
      const loadWorker = async () => {
        while (!cancelled) {
          const resource = uniqueResources[nextResourceIndex];
          nextResourceIndex += 1;
          if (!resource) return;
          await loadOneResource(resource);
        }
      };

      // Keep the UI responsive while two requests at a time warm the backend
      // and browser caches. The full pane grid remains visible while loading.
      const workerCount = Math.min(2, uniqueResources.length);
      await Promise.all(Array.from({ length: workerCount }, loadWorker));
      if (!cancelled) {
        setLoadingResources(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [subjectId, panes, checkpointId, loadResource, renderedPaneCount]);

  useEffect(() => {
    // Keep every requested view visible while its data loads. This makes the
    // workspace map stable and prevents users from mistaking progressive
    // renderer mounting for missing views.
    setRenderedPaneCount(panes.length);
  }, [panes]);

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

  const status = appError ? "error" : loadingSubjects || loadingResources ? "loading" : "ready";
  const statusLabel = appError
    ? "Viewer error"
    : loadingSubjects
      ? "Loading dataset catalog"
      : loadingResources
        ? `Loading views ${resourceProgress.completed}/${resourceProgress.total}`
        : "Viewer ready";

  const retrySelection = () => {
    setSubjectId("");
    setSelectedSubject(null);
    setMetadata(null);
    setResources({});
    setResourceErrors({});
    setAppError("");
    cacheRef.current.clear();
  };

  const adjustZoom = useCallback((direction) => {
    setZoom((current) => {
      const next = direction > 0 ? current * 1.2 : current / 1.2;
      return Math.min(MAX_2D_ZOOM, Math.max(MIN_2D_ZOOM, next));
    });
  }, []);

  const adjust3DZoom = useCallback((direction) => {
    setCameraState((current) => scaleCameraPosition(current, direction > 0 ? 1 / 1.2 : 1.2));
  }, []);

  const resetView = () => {
    setZoom(MIN_2D_ZOOM);
    setCameraState(null);
    setSliceIndex(Math.floor(sliceMaximum / 2));
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
              <FolderPicker
                value={selectedSubject?.path || (subjectId === "." ? "Dataset root" : "")}
                onChange={(folderPath) => {
                  setAppError("");
                  setMetadata(null);
                  setResources({});
                  setResourceErrors({});
                  setLoadingResources(false);
                  setSubjectId(folderPath);
                  setSelectedSubject({ path: folderPath === "." ? "Dataset root" : folderPath, label: folderPath.split("/").at(-1) || "Dataset root" });
                }}
                disabled={loadingSubjects}
              />
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
          <div className="zoom-control" aria-label={renderingMode === "slice" ? "Linked 2D zoom" : "Linked 3D zoom"}>
            <span className="toolbar-label">Zoom</span>
            <button aria-label="Zoom out" onClick={() => (renderingMode === "slice" ? adjustZoom(-1) : adjust3DZoom(-1))} disabled={renderingMode === "slice" ? zoom <= MIN_2D_ZOOM : !cameraState}>−</button>
            <span className="zoom-value">{renderingMode === "slice" ? `${Math.round(zoom * 100)}%` : "3D"}</span>
            <button aria-label="Zoom in" onClick={() => (renderingMode === "slice" ? adjustZoom(1) : adjust3DZoom(1))} disabled={renderingMode === "slice" ? zoom >= MAX_2D_ZOOM : !cameraState}>+</button>
          </div>
          <button className="reset-button" onClick={resetView}>Reset view</button>
        </div>

        {appError && (
          <div className="app-alert">
            <span>{appError}</span>
            <button className="alert-action" onClick={retrySelection}>Retry selection</button>
          </div>
        )}
        <div className="pane-grid" aria-busy={renderedPaneCount < panes.length}>
          {panes.slice(0, renderedPaneCount).map((pane) => {
            const volume = resourceFor(pane.volume);
            const mask = resourceFor(pane.mask || pane.maskOnly);
            const resourceError = resourceErrors[resourceKey({ ...(pane.mask || pane.maskOnly), checkpointId: (pane.mask || pane.maskOnly)?.maskKind === "output" ? checkpointId : undefined })];
            return (
              <Suspense key={pane.id} fallback={<section className="viewer-pane pane-suspense"><div className="pane-loading"><span className="spinner" /> Loading renderer</div></section>}>
                <ViewerPane title={pane.title} volume={volume} mask={mask} renderingMode={renderingMode} volumeMode={volumeMode} sliceAxis={sliceAxis} sliceIndex={sliceIndex} zoom={zoom} onZoom={adjustZoom} cameraState={cameraState} onCameraChange={setCameraState} loading={loadingResources && !volume && !mask} error={resourceError} />
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
