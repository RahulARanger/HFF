import { useEffect, useRef } from "react";
import vtkColorTransferFunction from "@kitware/vtk.js/Rendering/Core/ColorTransferFunction";
import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray";
import vtkGenericRenderWindow from "@kitware/vtk.js/Rendering/Misc/GenericRenderWindow";
import vtkImageData from "@kitware/vtk.js/Common/DataModel/ImageData";
import vtkImageMapper from "@kitware/vtk.js/Rendering/Core/ImageMapper";
import { SlicingMode } from "@kitware/vtk.js/Rendering/Core/ImageMapper/Constants";
import vtkImageSlice from "@kitware/vtk.js/Rendering/Core/ImageSlice";
import vtkPiecewiseFunction from "@kitware/vtk.js/Common/DataModel/PiecewiseFunction";
import vtkVolume from "@kitware/vtk.js/Rendering/Core/Volume";
import vtkVolumeMapper from "@kitware/vtk.js/Rendering/Core/VolumeMapper";
import "@kitware/vtk.js/Rendering/Profiles/Volume";

const AXIS_TO_SLICING_MODE = {
  axial: SlicingMode.K,
  coronal: SlicingMode.J,
  sagittal: SlicingMode.I,
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
  color.addRGBPoint(0, 0, 0, 0);
  Object.entries(LABEL_COLORS).forEach(([value, rgb]) => {
    color.addRGBPoint(Number(value), ...rgb);
  });
  const opacity = vtkPiecewiseFunction.newInstance();
  opacity.addPoint(0, 0.0);
  opacity.addPoint(0.5, 0.0);
  opacity.addPoint(1, 0.78);
  opacity.addPoint(2, 0.78);
  opacity.addPoint(3, 0.86);
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
  if (mode === "MIP") {
    mapper.setBlendModeToMaximumIntensity();
  } else {
    mapper.setBlendModeToComposite();
  }

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

function orientSliceCamera(renderer, resource, axis) {
  const [z, y, x] = resource.shape;
  const center = [(x - 1) / 2, (y - 1) / 2, (z - 1) / 2];
  const distance = Math.max(x, y, z) * 2;
  const camera = renderer.getActiveCamera();
  if (axis === "axial") {
    camera.setPosition(center[0], center[1], center[2] + distance);
    camera.setViewUp(0, 1, 0);
    camera.setParallelScale(Math.max(x, y));
  } else if (axis === "coronal") {
    camera.setPosition(center[0], center[1] + distance, center[2]);
    camera.setViewUp(0, 0, 1);
    camera.setParallelScale(Math.max(x, z));
  } else {
    camera.setPosition(center[0] + distance, center[1], center[2]);
    camera.setViewUp(0, 0, 1);
    camera.setParallelScale(Math.max(y, z));
  }
  camera.setFocalPoint(...center);
  camera.setParallelProjection(true);
  renderer.resetCameraClippingRange();
}

export default function ViewerPane({
  title,
  volume,
  mask,
  renderingMode,
  sliceAxis,
  sliceIndex,
  loading = false,
  error = null,
}) {
  const containerRef = useRef(null);
  const sceneRef = useRef(null);

  useEffect(() => {
    const primaryResource = volume || mask;
    if (!containerRef.current || !primaryResource) return undefined;

    const genericRenderWindow = vtkGenericRenderWindow.newInstance({
      background: [0.015, 0.025, 0.035],
    });
    genericRenderWindow.setContainer(containerRef.current);
    const renderer = genericRenderWindow.getRenderer();
    const renderWindow = genericRenderWindow.getRenderWindow();
    const scene = { genericRenderWindow, renderer, renderWindow, layers: [] };

    if (renderingMode === "slice") {
      if (volume) {
        scene.layers.push(addSliceLayer(renderer, volume, false));
        if (mask) scene.layers.push(addSliceLayer(renderer, mask, true));
      } else {
        scene.layers.push(addSliceLayer(renderer, primaryResource, true));
      }
    } else {
      if (volume) {
        scene.layers.push(addVolumeLayer(renderer, volume, renderingMode, false));
        if (mask) scene.layers.push(addVolumeLayer(renderer, mask, renderingMode, true));
      } else {
        scene.layers.push(addVolumeLayer(renderer, primaryResource, renderingMode, true));
      }
    }

    renderer.resetCamera();
    if (renderingMode === "slice") {
      renderer.getActiveCamera().setParallelProjection(true);
    }
    renderWindow.render();
    sceneRef.current = scene;

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
      sceneRef.current = null;
      genericRenderWindow.delete();
    };
  }, [volume, mask, renderingMode]);

  useEffect(() => {
    const scene = sceneRef.current;
    const primaryResource = volume || mask;
    if (!scene || renderingMode !== "slice" || !primaryResource) return;
    const firstLayer = scene.layers[0];
    if (!firstLayer?.mapper) return;
    firstLayer.mapper.setSlicingMode(AXIS_TO_SLICING_MODE[sliceAxis]);
    firstLayer.mapper.setSlice(Math.max(0, Math.min(sliceIndex, sliceMaximum(primaryResource, sliceAxis))));
    const labelLayer = scene.layers[1];
    if (labelLayer?.mapper) {
      labelLayer.mapper.setSlicingMode(AXIS_TO_SLICING_MODE[sliceAxis]);
      labelLayer.mapper.setSlice(Math.max(0, Math.min(sliceIndex, sliceMaximum(mask, sliceAxis))));
    }
    orientSliceCamera(scene.renderer, primaryResource, sliceAxis);
    scene.renderWindow.render();
  }, [renderingMode, sliceAxis, sliceIndex, volume, mask]);

  return (
    <section className="viewer-pane" aria-label={`${title} visualization`}>
      <header className="pane-header">
        <span className="pane-title">{title}</span>
        <span className="pane-meta">
          {(volume || mask) ? `${(volume || mask).shape[2]} × ${(volume || mask).shape[1]} × ${(volume || mask).shape[0]}` : "Waiting"}
        </span>
        <span className="pane-actions" aria-hidden="true">
          <span className="pane-action">⌖</span>
          <span className="pane-action">□</span>
          <span className="pane-action">↗</span>
        </span>
      </header>
      <div className="pane-canvas" ref={containerRef}>
        {!volume && !mask && !loading && <div className="pane-empty">{error || "Select a record to load this view"}</div>}
        {loading && <div className="pane-loading"><span className="spinner" /> Loading volume</div>}
        {error && (volume || mask) && <div className="pane-error">{error}</div>}
        {(volume || mask) && renderingMode === "slice" && (
          <span className="slice-readout">{sliceAxis} · {sliceIndex + 1} / {sliceMaximum(volume || mask, sliceAxis) + 1}</span>
        )}
      </div>
    </section>
  );
}
