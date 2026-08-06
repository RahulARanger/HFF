import { useEffect, useMemo, useRef, useState } from "react";
import { cmapper, Niivue } from "@niivue/niivue";

const AXIS_TO_INDEX = { sagittal: 0, coronal: 1, axial: 2 };
const AXIS_TO_SLICE_TYPE = {
  axial: "sliceTypeAxial",
  coronal: "sliceTypeCoronal",
  sagittal: "sliceTypeSagittal",
};
const OVERVIEW_AXES = {
  axial: ["coronal", "sagittal"],
  coronal: ["axial", "sagittal"],
  sagittal: ["axial", "coronal"],
};

const LABEL_LUT = cmapper.makeLabelLut({
  R: [0, 242, 26, 255],
  G: [0, 38, 140, 217],
  B: [0, 51, 255, 13],
  A: [0, 214, 214, 230],
  I: [0, 1, 2, 3],
  labels: ["Background", "Necrotic / non-enhancing core", "Edema", "Enhancing tumour"],
});

function shapeFor(volume, mask) {
  return volume?.shape || mask?.shape || null;
}

function sliceMaximum(shape, axis) {
  if (!shape) return 0;
  const [z, y, x] = shape;
  return ({ axial: z, coronal: y, sagittal: x }[axis] || 1) - 1;
}

function normalizedCrosshair(shape, axis, sliceIndex) {
  const [z, y, x] = shape;
  const focus = [(x - 1) / 2, (y - 1) / 2, (z - 1) / 2];
  const axisIndex = AXIS_TO_INDEX[axis];
  const maximum = sliceMaximum(shape, axis);
  focus[axisIndex] = Math.max(0, Math.min(sliceIndex, maximum));
  return focus.map((value, index) => value / Math.max(shape.slice().reverse()[index] - 1, 1));
}

function configureSliceLayout(nv, sliceAxis) {
  const otherAxes = OVERVIEW_AXES[sliceAxis];
  const sliceType = (axis) => nv[AXIS_TO_SLICE_TYPE[axis]];
  nv.setSliceType(nv.sliceTypeMultiplanar);
  nv.setCustomLayout([
    { sliceType: sliceType(sliceAxis), position: [0, 0, 0.70, 1] },
    { sliceType: sliceType(otherAxes[0]), position: [0.715, 0, 0.285, 0.485] },
    { sliceType: sliceType(otherAxes[1]), position: [0.715, 0.515, 0.285, 0.485] },
  ]);
  nv.setMultiplanarPadPixels(4);
  nv.setCornerOrientationText(true);
  nv.setIsOrientationTextVisible(true);
}

function configureScene(nv, renderingMode, sliceAxis, volumeScale) {
  if (renderingMode === "slice") {
    configureSliceLayout(nv, sliceAxis);
  } else {
    nv.clearCustomLayout();
    nv.setSliceType(nv.sliceTypeRender);
    nv.setScale(volumeScale);
  }
  nv.drawScene();
}

function setCrosshair(nv, shape, sliceAxis, sliceIndex, zoom) {
  if (!shape) return;
  nv.scene.crosshairPos = normalizedCrosshair(shape, sliceAxis, sliceIndex);
  const pan = nv.scene.pan2Dxyzmm || [0, 0, 0, 1];
  nv.setPan2Dxyzmm([pan[0], pan[1], pan[2], zoom]);
  nv.drawScene();
}

function imageOptions(volume, mask) {
  const options = [];
  if (volume) {
    options.push({
      url: volume.url,
      name: volume.name,
      colormap: "gray",
      opacity: 1,
    });
  }
  if (mask) {
    options.push({
      url: mask.url,
      name: mask.name,
      colormapLabel: LABEL_LUT,
      ignoreZeroVoxels: true,
      opacity: 0.84,
    });
  }
  return options;
}

function applyMaskColormap(nv, hasVolume) {
  const maskIndex = hasVolume ? 1 : 0;
  const maskVolume = nv.volumes[maskIndex];
  if (!maskVolume) return;

  // Niivue's batch loader does not currently forward colormapLabel or
  // ignoreZeroVoxels into NVImage, so apply the overlay settings after load.
  maskVolume.colormapLabel = LABEL_LUT;
  maskVolume.ignoreZeroVoxels = true;
  nv.updateGLVolume();
}

export default function ViewerPane({
  title,
  volume,
  mask,
  renderingMode,
  sliceAxis,
  sliceIndex,
  zoom = 1,
  volumeScale = 1,
  onZoom,
  onSliceChange,
}) {
  const canvasRef = useRef(null);
  const niivueRef = useRef(null);
  const onSliceChangeRef = useRef(onSliceChange);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const shape = useMemo(() => shapeFor(volume, mask), [volume, mask]);
  const sourceKey = [volume?.url || "", volume?.name || "", mask?.url || "", mask?.name || ""].join("|");

  useEffect(() => {
    onSliceChangeRef.current = onSliceChange;
  }, [onSliceChange]);

  useEffect(() => {
    if (!canvasRef.current || !sourceKey || (!volume && !mask)) return undefined;

    let cancelled = false;
    const nv = new Niivue({
      backColor: [0.015, 0.025, 0.035, 1],
      crosshairColor: [0.24, 0.76, 0.94, 0.85],
      crosshairWidth: 1,
      dragAndDropEnabled: false,
      isColorbar: false,
      isOrientCube: false,
      isResizeCanvas: true,
      isSliceMM: false,
      logLevel: "error",
      multiplanarShowRender: 0,
      textHeight: 0.025,
    });
    niivueRef.current = nv;
    setLoading(true);
    setError("");

    nv.onLocationChange = (location) => {
      const voxels = location?.vox;
      const axisIndex = AXIS_TO_INDEX[sliceAxis];
      if (Array.isArray(voxels) && Number.isFinite(voxels[axisIndex])) {
        onSliceChangeRef.current?.(Math.round(voxels[axisIndex]));
      }
    };

    (async () => {
      await nv.attachToCanvas(canvasRef.current);
      await nv.loadVolumes(imageOptions(volume, mask));
      if (cancelled) return;
      if (mask) applyMaskColormap(nv, Boolean(volume));
      configureScene(nv, renderingMode, sliceAxis, volumeScale);
      setCrosshair(nv, shape, sliceAxis, sliceIndex, zoom);
      setLoading(false);
    })().catch((loadError) => {
      if (!cancelled) {
        setLoading(false);
        setError(loadError instanceof Error ? loadError.message : String(loadError));
      }
    });

    return () => {
      cancelled = true;
      if (niivueRef.current === nv) niivueRef.current = null;
      nv.cleanup();
    };
  }, [sourceKey]);

  useEffect(() => {
    const nv = niivueRef.current;
    if (!nv || loading) return;
    configureScene(nv, renderingMode, sliceAxis, volumeScale);
    if (renderingMode === "slice") setCrosshair(nv, shape, sliceAxis, sliceIndex, zoom);
  }, [renderingMode, sliceAxis, volumeScale, loading, shape, sliceIndex, zoom]);

  const handleWheel = (event) => {
    if (renderingMode !== "slice" || !onZoom || !event.ctrlKey) return;
    event.preventDefault();
    event.stopPropagation();
    onZoom(event.deltaY < 0 ? 1 : -1);
  };

  return (
    <section className="viewer-pane" aria-label={`${title} visualization`}>
      <header className="pane-header">
        <span className="pane-title">{title}</span>
        <span className="pane-meta">
          {shape ? `${shape[2]} × ${shape[1]} × ${shape[0]}` : "Waiting"}
        </span>
        <span className="pane-actions" aria-hidden="true">
          <span className="pane-action">⌖</span>
          <span className="pane-action">□</span>
          <span className="pane-action">↗</span>
        </span>
      </header>
      <div className={`pane-canvas ${renderingMode === "slice" ? "slice-canvas" : ""}`} onWheelCapture={handleWheel}>
        {!volume && !mask && <div className="pane-empty">Select a record to load this view</div>}
        {(volume || mask) && <canvas ref={canvasRef} className="niivue-canvas" aria-label={`${title} NiiVue canvas`} />}
        {loading && <div className="pane-loading"><span className="spinner" /> Loading NiiVue view</div>}
        {error && <div className="pane-error">{error}</div>}
        {shape && renderingMode === "slice" && (
          <span className="slice-readout">
            {sliceAxis} · {sliceIndex + 1} / {sliceMaximum(shape, sliceAxis) + 1} · {Math.round(zoom * 100)}%
          </span>
        )}
      </div>
    </section>
  );
}
