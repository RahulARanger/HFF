import { useEffect, useMemo, useRef } from "react";
import vtkColorTransferFunction from "@kitware/vtk.js/Rendering/Core/ColorTransferFunction";
import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray";
import vtkGenericRenderWindow from "@kitware/vtk.js/Rendering/Misc/GenericRenderWindow";
import vtkImageData from "@kitware/vtk.js/Common/DataModel/ImageData";
import vtkImageMapper from "@kitware/vtk.js/Rendering/Core/ImageMapper";
import { SlicingMode } from "@kitware/vtk.js/Rendering/Core/ImageMapper/Constants";
import vtkImageSlice from "@kitware/vtk.js/Rendering/Core/ImageSlice";
import vtkPiecewiseFunction from "@kitware/vtk.js/Common/DataModel/PiecewiseFunction";
import vtkRenderer from "@kitware/vtk.js/Rendering/Core/Renderer";
import vtkVolume from "@kitware/vtk.js/Rendering/Core/Volume";
import vtkVolumeMapper from "@kitware/vtk.js/Rendering/Core/VolumeMapper";
import "@kitware/vtk.js/Rendering/Profiles/Volume";

const AXIS_TO_SLICING_MODE = {
  axial: SlicingMode.K,
  coronal: SlicingMode.J,
  sagittal: SlicingMode.I,
};

const AXIS_TO_WORLD_INDEX = { sagittal: 0, coronal: 1, axial: 2 };
const OVERVIEW_AXES = {
  axial: ["coronal", "sagittal"],
  coronal: ["axial", "sagittal"],
  sagittal: ["axial", "coronal"],
};

const VIEWPORTS = {
  main: [0, 0, 0.70, 1],
  overviewOne: [0.715, 0.515, 1, 1],
  overviewTwo: [0.715, 0, 1, 0.485],
};

const OVERLAY_LAYOUTS = {
  main: { left: 0, top: 0, width: 0.70, height: 1 },
  overviewOne: { left: 0.715, top: 0, width: 0.285, height: 0.485 },
  overviewTwo: { left: 0.715, top: 0.515, width: 0.285, height: 0.485 },
};

const LABEL_COLORS = {
  1: [0.95, 0.15, 0.2],
  2: [0.1, 0.55, 1.0],
  3: [1.0, 0.85, 0.05],
};

function createImageData(resource) {
  const [z, y, x] = resource.shape;
  const imageData = vtkImageData.newInstance();
  imageData.setDimensions(x, y, z);
  imageData.getPointData().setScalars(
    vtkDataArray.newInstance({
      name: resource.dtype === "uint8" ? "labels" : "intensity",
      numberOfComponents: 1,
      values: resource.values,
    }),
  );
  imageData.modified();
  return imageData;
}

function createIntensityTransferFunctions(resource) {
  const [lower, upper] = resource.intensityRange;
  const span = Math.max(upper - lower, 1);
  const color = vtkColorTransferFunction.newInstance();
  color.addRGBPoint(lower, 0.02, 0.03, 0.04);
  color.addRGBPoint(lower + span * 0.45, 0.52, 0.62, 0.68);
  color.addRGBPoint(upper, 1.0, 1.0, 1.0);
  const opacity = vtkPiecewiseFunction.newInstance();
  opacity.addPoint(lower, 0.0);
  opacity.addPoint(lower + span * 0.06, 0.02);
  opacity.addPoint(lower + span * 0.34, 0.16);
  opacity.addPoint(upper, 0.72);
  return { color, opacity };
}

function createLabelTransferFunctions() {
  const color = vtkColorTransferFunction.newInstance();
  color.addRGBPointLong(0, 0, 0, 0, 0.5, 1.0);
  Object.entries(LABEL_COLORS).forEach(([value, rgb]) => {
    // Sharp transfer-function transitions preserve the exact BraTS label
    // boundaries instead of blending adjacent tumour regions together.
    color.addRGBPointLong(Number(value), ...rgb, 0.5, 1.0);
  });
  const opacity = vtkPiecewiseFunction.newInstance();
  opacity.addPoint(0, 0.0);
  opacity.addPoint(0.5, 0.0);
  opacity.addPoint(0.99, 0.0);
  opacity.addPoint(1, 0.84);
  opacity.addPoint(1.99, 0.84);
  opacity.addPoint(2, 0.84);
  opacity.addPoint(2.99, 0.84);
  opacity.addPoint(3, 0.9);
  return { color, opacity };
}

function applySliceProperty(actor, resource) {
  const property = actor.getProperty();
  const [lower, upper] = resource.intensityRange;
  property.setColorWindow(Math.max(upper - lower, 1));
  property.setColorLevel((upper + lower) / 2);
  property.setInterpolationTypeToLinear();
}

function addSliceLayer(renderer, resource, isLabel = false) {
  const imageData = createImageData(resource);
  const mapper = vtkImageMapper.newInstance();
  mapper.setInputData(imageData);
  const actor = vtkImageSlice.newInstance();
  actor.setMapper(mapper);
  if (isLabel) {
    const transfer = createLabelTransferFunctions();
    actor.getProperty().setRGBTransferFunction(0, transfer.color);
    actor.getProperty().setPiecewiseFunction(0, transfer.opacity);
    actor.getProperty().setUseLookupTableScalarRange(true);
    actor.getProperty().setInterpolationTypeToNearest();
  } else {
    applySliceProperty(actor, resource);
  }
  renderer.addActor(actor);
  return { actor, mapper };
}

function addVolumeLayer(renderer, resource, mode, isLabel = false) {
  const imageData = createImageData(resource);
  const mapper = vtkVolumeMapper.newInstance();
  mapper.setInputData(imageData);
  mapper.setSampleDistance(isLabel ? 1.2 : 1.0);
  if (mode === "MIP") mapper.setBlendModeToMaximumIntensity();
  else mapper.setBlendModeToComposite();

  const actor = vtkVolume.newInstance();
  actor.setMapper(mapper);
  const property = actor.getProperty();
  const transfer = isLabel ? createLabelTransferFunctions() : createIntensityTransferFunctions(resource);
  property.setRGBTransferFunction(0, transfer.color);
  property.setScalarOpacity(0, transfer.opacity);
  property.setInterpolationTypeToNearest();
  if (isLabel) {
    property.setShade(false);
  } else {
    property.setShade(true);
    property.setAmbient(0.25);
    property.setDiffuse(0.7);
    property.setSpecular(0.1);
  }
  renderer.addVolume(actor);
  return { actor, mapper };
}

function sliceMaximum(resource, axis) {
  const [z, y, x] = resource.shape;
  return { axial: z, coronal: y, sagittal: x }[axis] - 1;
}

function focusFromSlice(resource, axis, sliceIndex) {
  const [z, y, x] = resource.shape;
  const focus = [(x - 1) / 2, (y - 1) / 2, (z - 1) / 2];
  focus[AXIS_TO_WORLD_INDEX[axis]] = Math.max(0, Math.min(sliceIndex, sliceMaximum(resource, axis)));
  return focus;
}

function configureSliceView(view, focus, zoom) {
  const primaryResource = view.volume || view.mask;
  if (!primaryResource) return;

  const slice = Math.round(focus[AXIS_TO_WORLD_INDEX[view.axis]]);
  view.layers.forEach((layer) => {
    layer.mapper.setSlicingMode(AXIS_TO_SLICING_MODE[view.axis]);
    layer.mapper.setSlice(Math.max(0, Math.min(slice, sliceMaximum(layer.resource, view.axis))));
  });

  const [z, y, x] = primaryResource.shape;
  // Establish a valid clipping range from the image bounds before applying
  // the linked camera position and parallel zoom.
  view.renderer.resetCamera();
  const camera = view.renderer.getActiveCamera();
  const distance = Math.max(x, y, z) * 2;
  const position = [...focus];
  const axisIndex = AXIS_TO_WORLD_INDEX[view.axis];
  position[axisIndex] += distance;
  camera.setPosition(...position);
  camera.setFocalPoint(...focus);
  camera.setParallelProjection(true);

  const baseScale = view.axis === "axial" ? Math.max(x, y) : view.axis === "coronal" ? Math.max(x, z) : Math.max(y, z);
  // The image volume contains a generous black border around the brain.  A
  // slightly tighter fit keeps the 2D panes readable like the Napari view
  // while still leaving enough margin for navigation and crosshairs.
  const fitScale = baseScale * 0.72;
  camera.setParallelScale(fitScale / Math.max(zoom, 1));
  if (view.axis === "axial") camera.setViewUp(0, 1, 0);
  else camera.setViewUp(0, 0, 1);
  view.renderer.resetCameraClippingRange();
}

function readCameraState(camera) {
  return {
    position: [...camera.getPosition()],
    focalPoint: [...camera.getFocalPoint()],
    viewUp: [...camera.getViewUp()],
  };
}

function applyCameraState(camera, cameraState) {
  camera.setPosition(...cameraState.position);
  camera.setFocalPoint(...cameraState.focalPoint);
  camera.setViewUp(...cameraState.viewUp);
}

function crosshairPosition(shape, focus, axis) {
  const [z, y, x] = shape;
  const normalized = {
    x: focus[0] / Math.max(x - 1, 1),
    y: focus[1] / Math.max(y - 1, 1),
    z: focus[2] / Math.max(z - 1, 1),
  };
  if (axis === "axial") return { x: normalized.x, y: 1 - normalized.y };
  if (axis === "coronal") return { x: normalized.x, y: 1 - normalized.z };
  return { x: normalized.y, y: 1 - normalized.z };
}

function CrosshairOverlay({ resource, axis, focus }) {
  if (!resource) return null;
  const layouts = [
    { id: "main", axis, layout: OVERLAY_LAYOUTS.main },
    { id: "overview-one", axis: OVERVIEW_AXES[axis][0], layout: OVERLAY_LAYOUTS.overviewOne },
    { id: "overview-two", axis: OVERVIEW_AXES[axis][1], layout: OVERLAY_LAYOUTS.overviewTwo },
  ];
  return (
    <div className="slice-crosshair-layer" aria-hidden="true">
      {layouts.map(({ id, axis: viewAxis, layout }) => {
        const point = crosshairPosition(resource.shape, focus, viewAxis);
        return (
          <div className="slice-crosshair" key={id}>
            <span
              className="slice-crosshair-line vertical"
              style={{ left: `${(layout.left + layout.width * point.x) * 100}%`, top: `${layout.top * 100}%`, height: `${layout.height * 100}%` }}
            />
            <span
              className="slice-crosshair-line horizontal"
              style={{ left: `${layout.left * 100}%`, top: `${(layout.top + layout.height * point.y) * 100}%`, width: `${layout.width * 100}%` }}
            />
          </div>
        );
      })}
    </div>
  );
}

export default function ViewerPane({
  title,
  volume,
  mask,
  renderingMode,
  volumeMode,
  sliceAxis,
  sliceIndex,
  zoom = 1,
  onZoom,
  cameraState = null,
  onCameraChange,
  loading = false,
  error = null,
}) {
  const containerRef = useRef(null);
  const sceneRef = useRef(null);
  const primaryResource = volume || mask;
  const overviewAxes = useMemo(() => OVERVIEW_AXES[sliceAxis], [sliceAxis]);
  const focus = useMemo(
    () => (primaryResource ? focusFromSlice(primaryResource, sliceAxis, sliceIndex) : null),
    [primaryResource, sliceAxis, sliceIndex],
  );

  useEffect(() => {
    if (!containerRef.current || !primaryResource) return undefined;

    const genericRenderWindow = vtkGenericRenderWindow.newInstance({
      background: [0.015, 0.025, 0.035],
    });
    genericRenderWindow.setContainer(containerRef.current);
    const renderWindow = genericRenderWindow.getRenderWindow();
    const mainRenderer = genericRenderWindow.getRenderer();
    const extraRenderers = [];
    const scene = { genericRenderWindow, renderWindow, views: [], extraRenderers, renderer: mainRenderer, camera: null, applyingCameraState: false };

    if (renderingMode === "slice") {
      mainRenderer.setViewport(...VIEWPORTS.main);
      mainRenderer.setBackground(0.015, 0.025, 0.035);
      const overviewOne = vtkRenderer.newInstance({ background: [0.015, 0.025, 0.035] });
      const overviewTwo = vtkRenderer.newInstance({ background: [0.015, 0.025, 0.035] });
      overviewOne.setViewport(...VIEWPORTS.overviewOne);
      overviewTwo.setViewport(...VIEWPORTS.overviewTwo);
      renderWindow.addRenderer(overviewOne);
      renderWindow.addRenderer(overviewTwo);
      extraRenderers.push(overviewOne, overviewTwo);

      const viewDefinitions = [
        { renderer: mainRenderer, axis: sliceAxis, overview: false },
        { renderer: overviewOne, axis: overviewAxes[0], overview: true },
        { renderer: overviewTwo, axis: overviewAxes[1], overview: true },
      ];
      viewDefinitions.forEach((definition) => {
        const layers = [];
        if (volume) layers.push({ ...addSliceLayer(definition.renderer, volume, false), resource: volume });
        if (mask) layers.push({ ...addSliceLayer(definition.renderer, mask, true), resource: mask });
        scene.views.push({ ...definition, volume, mask, layers });
      });
    } else {
      const layers = [];
      if (volume) layers.push(addVolumeLayer(mainRenderer, volume, volumeMode, false));
      if (mask) layers.push(addVolumeLayer(mainRenderer, mask, volumeMode, true));
      scene.views.push({ renderer: mainRenderer, volume, mask, layers, overview: false });
      mainRenderer.resetCamera();
    }

    sceneRef.current = scene;
    if (renderingMode === "slice" && focus) {
      scene.views.forEach((view) => configureSliceView(view, focus, zoom));
    }
    renderWindow.render();

    const camera = mainRenderer.getActiveCamera();
    const cameraSubscription = renderingMode === "volume" && onCameraChange
      ? camera.onModified(() => {
          if (!scene.applyingCameraState) onCameraChange(readCameraState(camera));
        })
      : null;
    scene.cameraSubscription = cameraSubscription;
    scene.camera = camera;
    if (renderingMode === "volume" && onCameraChange) {
      if (cameraState) {
        scene.applyingCameraState = true;
        applyCameraState(camera, cameraState);
        scene.applyingCameraState = false;
      } else {
        onCameraChange(readCameraState(camera));
      }
    }

    let disposed = false;
    const resizeObserver = new ResizeObserver(() => {
      if (disposed || !sceneRef.current) return;
      genericRenderWindow.resize();
      renderWindow.render();
    });
    resizeObserver.observe(containerRef.current);

    return () => {
      disposed = true;
      resizeObserver.disconnect();
      cameraSubscription?.unsubscribe();
      sceneRef.current = null;
      extraRenderers.forEach((renderer) => {
        renderWindow.removeRenderer(renderer);
        renderer.delete();
      });
      genericRenderWindow.delete();
    };
  }, [primaryResource, volume, mask, renderingMode, volumeMode, sliceAxis, overviewAxes]);

  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene || renderingMode !== "volume" || !scene.camera) return;
    scene.applyingCameraState = true;
    try {
      if (cameraState) {
        applyCameraState(scene.camera, cameraState);
      } else {
        scene.renderer.resetCamera();
      }
    } finally {
      scene.applyingCameraState = false;
    }
    scene.renderWindow.render();
  }, [renderingMode, cameraState]);

  useEffect(() => {
    const scene = sceneRef.current;
    if (!scene || renderingMode !== "slice" || !focus) return;
    scene.views.forEach((view) => configureSliceView(view, focus, zoom));
    scene.renderWindow.render();
  }, [renderingMode, focus, zoom]);

  const handleWheel = (event) => {
    if (renderingMode !== "slice" || !onZoom) return;
    event.preventDefault();
    event.stopPropagation();
    onZoom(event.deltaY < 0 ? 1 : -1);
  };

  return (
    <section className="viewer-pane" aria-label={`${title} visualization`}>
      <header className="pane-header">
        <span className="pane-title">{title}</span>
        <span className="pane-meta">
          {primaryResource ? `${primaryResource.shape[2]} × ${primaryResource.shape[1]} × ${primaryResource.shape[0]}` : "Waiting"}
        </span>
        <span className="pane-actions" aria-hidden="true">
          <span className="pane-action">⌖</span>
          <span className="pane-action">□</span>
          <span className="pane-action">↗</span>
        </span>
      </header>
      <div className={`pane-canvas ${renderingMode === "slice" ? "slice-canvas" : ""}`} ref={containerRef} onWheelCapture={handleWheel}>
        {!volume && !mask && !loading && !error && <div className="pane-empty">Select a record to load this view</div>}
        {loading && <div className="pane-loading"><span className="spinner" /> Loading volume</div>}
        {error && <div className="pane-error">{error}</div>}
        {renderingMode === "slice" && primaryResource && focus && <CrosshairOverlay resource={primaryResource} axis={sliceAxis} focus={focus} />}
        {primaryResource && renderingMode === "slice" && (
          <span className="slice-readout">{sliceAxis} · {sliceIndex + 1} / {sliceMaximum(primaryResource, sliceAxis) + 1} · {Math.round(zoom * 100)}%</span>
        )}
      </div>
    </section>
  );
}
