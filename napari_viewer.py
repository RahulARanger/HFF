"""Interactive Napari viewer for input and frequency-decomposition analysis."""

from __future__ import annotations

import argparse
from pathlib import Path

import napari
import numpy as np
import SimpleITK as sitk
import torch
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QComboBox,
    QCompleter,
    QFileDialog,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)
from napari.utils.colormaps import Colormap, DirectLabelColormap

from model.HFF import HFFNet
from utils.utils import get_device


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "dataset"
DEFAULT_RESULTS_ROOT = PROJECT_ROOT / "result"
MODALITY_PRIORITY = ("t1", "t1ce", "t2", "flair")
MODEL_LOW_MODALITIES = ("flair_L", "t1_L", "t1ce_L", "t2_L")
MODEL_HIGH_MODALITIES = tuple(
    f"{modality}_H{band}"
    for modality in ("flair", "t1", "t1ce", "t2")
    for band in range(1, 5)
)


def strip_nifti_suffix(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    return path.stem


def is_nifti_file(path: Path) -> bool:
    return path.name.endswith(".nii") or path.name.endswith(".nii.gz")


def is_frequency_file(path: Path) -> bool:
    stem = strip_nifti_suffix(path).lower()
    return stem.rsplit("_", 1)[-1] in {"l", "h", "h1", "h2", "h3", "h4"}


def modality_sort_key(path: Path) -> tuple[int, str]:
    stem = strip_nifti_suffix(path).lower()
    for index, modality in enumerate(MODALITY_PRIORITY):
        if stem.endswith(f"_{modality}"):
            return index, stem
    return 99, stem


def subject_files(subject_dir: Path) -> tuple[list[Path], Path | None]:
    files = [path for path in subject_dir.iterdir() if path.is_file() and is_nifti_file(path)]
    scans = sorted(
        [path for path in files if "_seg" not in path.name and not is_frequency_file(path)],
        key=modality_sort_key,
    )
    segmentation = next((path for path in files if "_seg" in path.name), None)
    return scans[:4], segmentation


def load_volume(path: Path) -> np.ndarray:
    return sitk.GetArrayFromImage(sitk.ReadImage(str(path))).astype(np.float32)


def load_subject(subject_dir: Path) -> tuple[dict[str, np.ndarray], np.ndarray | None]:
    scans, segmentation = subject_files(subject_dir)

    volumes = {
        strip_nifti_suffix(path).rsplit("_", 1)[-1].upper(): load_volume(path)
        for path in scans
    }
    mask = load_volume(segmentation).astype(np.uint8) if segmentation is not None else None
    return volumes, mask


def contrast_limits(volume: np.ndarray) -> tuple[float, float]:
    values = volume[np.isfinite(volume)]
    if values.size == 0:
        return 0.0, 1.0
    lower, upper = (float(value) for value in np.percentile(values, [1, 99]))
    if upper <= lower:
        return lower, lower + 1.0
    return lower, upper


def resize_mask_to_shape(mask: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Resize a label mask with nearest-neighbour sampling for low-pass data."""
    if mask.shape == target_shape:
        return mask
    if mask.ndim != len(target_shape):
        return mask

    source_indices = [
        np.minimum(
            (np.arange(target_size) * source_size / target_size).astype(np.intp),
            source_size - 1,
        )
        for source_size, target_size in zip(mask.shape, target_shape)
    ]
    return mask[np.ix_(*source_indices)]


def resize_volume_to_shape(volume: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Resample a frequency volume to the source scan's display shape."""
    if volume.shape == target_shape:
        return volume

    image = sitk.GetImageFromArray(volume)
    source_size = image.GetSize()
    target_size = tuple(reversed(target_shape))
    source_spacing = image.GetSpacing()
    target_spacing = tuple(
        source_spacing[index] * source_size[index] / target_size[index]
        for index in range(len(target_size))
    )

    resampler = sitk.ResampleImageFilter()
    resampler.SetSize(target_size)
    resampler.SetOutputSpacing(target_spacing)
    resampler.SetOutputOrigin(image.GetOrigin())
    resampler.SetOutputDirection(image.GetDirection())
    resampler.SetInterpolator(sitk.sitkLinear)
    resampler.SetDefaultPixelValue(0.0)
    return sitk.GetArrayFromImage(resampler.Execute(image)).astype(np.float32)


def suppress_background(volume: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Remove transform padding outside the original scan's foreground."""
    foreground = resize_mask_to_shape(reference > 0, volume.shape)
    cleaned = volume.copy()
    cleaned[~foreground] = 0
    return cleaned


BRATS_SEGMENTATION_COLORS = {
    1: [0.95, 0.15, 0.20, 1.0],  # necrotic / non-enhancing tumour core
    2: [0.10, 0.55, 1.00, 1.0],  # peritumoral edema
    4: [1.00, 0.85, 0.05, 1.0],  # enhancing tumour (BraTS ground truth)
}
BRATS_LABEL_COLORMAP = DirectLabelColormap(
    color_dict={
        None: [0.0, 0.0, 0.0, 0.0],
        0: [0.0, 0.0, 0.0, 0.0],
        1: BRATS_SEGMENTATION_COLORS[1],
        2: BRATS_SEGMENTATION_COLORS[2],
        3: BRATS_SEGMENTATION_COLORS[4],  # model ET class
        4: BRATS_SEGMENTATION_COLORS[4],  # BraTS ET class
    },
    name="BraTS categorical labels",
)
SEGMENTATION_OVERLAY_VALUES = {1: 0.65, 2: 0.82, 3: 1.0}


def canonical_segmentation_labels(mask: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """Use 1/core, 2/edema, and 3/ET for both BraTS targets and model output."""
    labels = resize_mask_to_shape(mask, target_shape).astype(np.uint8, copy=True)
    # BraTS targets encode enhancing tumour as 4; the network's multiclass
    # output encodes the corresponding class as 3.
    labels[labels == 4] = 3
    return labels


def build_segmentation_overlay(volume: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Encode MRI plus labels for Napari's scalar 3D renderer.

    The generated image is only a display composite. Output masks remain
    discrete ``Labels`` layers with ``BRATS_LABEL_COLORMAP``.
    """
    lower, upper = contrast_limits(volume)
    if upper <= lower:
        normalized = np.zeros_like(volume, dtype=np.float32)
    else:
        normalized = np.clip((volume - lower) / (upper - lower), 0, 1)

    overlay = normalized * 0.55
    if mask is not None:
        labels = canonical_segmentation_labels(mask, volume.shape)
        for label, value in SEGMENTATION_OVERLAY_VALUES.items():
            overlay[labels == label] = value
    return overlay.astype(np.float32)


SEGMENTATION_BLEND_COLORMAP = Colormap(
    colors=np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.55, 0.55, 0.55, 1.0],
            BRATS_SEGMENTATION_COLORS[1],
            BRATS_SEGMENTATION_COLORS[2],
            BRATS_SEGMENTATION_COLORS[4],
        ],
        dtype=np.float32,
    ),
    controls=np.array([0.0, 0.55, 0.65, 0.82, 1.0], dtype=np.float32),
    name="BraTS segmentation overlay",
)


def build_output_overlay(volume: np.ndarray, output_mask: np.ndarray | None) -> np.ndarray:
    """Encode a scan and class-coloured generated prediction."""
    return build_segmentation_overlay(volume, output_mask)


def discover_checkpoints(results_root: Path) -> list[Path]:
    """Return all HFF-Net checkpoints, newest first, on every refresh."""
    if not results_root.exists():
        return []
    return sorted(
        (path for path in results_root.rglob("*.pth") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )


def checkpoint_output_path(results_root: Path, checkpoint: Path, subject_dir: Path) -> Path:
    """Keep generated masks per checkpoint so switching checkpoints is unambiguous."""
    return results_root / "eval" / checkpoint.stem / f"{subject_dir.name}_OSeg.nii.gz"


def normalize_for_model(volume: np.ndarray) -> np.ndarray:
    """Mirror the existing loader's per-volume [-1, 1] normalization exactly."""
    minimum = float(np.min(volume))
    maximum = float(np.max(volume))
    if maximum <= minimum:
        return np.zeros_like(volume, dtype=np.float32)
    return (2.0 * ((volume - minimum) / (maximum - minimum)) - 1.0).astype(np.float32)


def restrict_mask_to_scan_foreground(mask: np.ndarray, reference_scan: np.ndarray) -> np.ndarray:
    """Remove predictions in transform padding outside the acquired MRI volume.

    The HFF-Net predicts a fixed 128³ crop. Without this constraint, a model
    can label the crop's zero-padded corners and the viewer shows a rectangular
    box. The scan foreground is independent of the ground-truth segmentation.
    """
    foreground = reference_scan != reference_scan.flat[0]
    cleaned = np.asarray(mask, dtype=np.uint8).copy()
    cleaned[~foreground] = 0
    return cleaned


def foreground_center_crop(volumes: list[np.ndarray], crop_size: int = 128) -> tuple[list[np.ndarray], tuple[slice, slice, slice]]:
    """Reproduce the deterministic validation crop used by ``loader/dataload3d.py``."""
    working = [volume[3:, :, :] for volume in volumes]
    foreground = np.zeros_like(working[0], dtype=bool)
    for volume in working:
        foreground |= volume != volume[0, 0, 0]
    coordinates = np.where(foreground)
    if coordinates[0].size == 0:
        raise ValueError("Cannot crop an empty scan foreground.")
    centre = tuple((int(axis.min()) + int(axis.max())) // 2 for axis in coordinates)
    bounds = (152, 240, 240)
    slices: list[slice] = []
    for coordinate, upper_bound in zip(centre, bounds):
        start = coordinate - crop_size // 2
        end = coordinate + crop_size // 2
        if start < 0:
            start, end = 0, crop_size
        elif end >= upper_bound:
            end, start = upper_bound - 1, upper_bound - 1 - crop_size
        slices.append(slice(start, end))
    crop = tuple(slices)
    return [volume[crop] for volume in working], crop


def generate_output_segmentation(checkpoint: Path, subject_dir: Path, results_root: Path) -> Path:
    """Run the selected HFF-Net checkpoint for one subject and write its ``_OSeg`` mask.

    This intentionally follows the repository's validation transform: discard the
    first three axial slices, take the deterministic foreground-centred 128³ crop,
    and restore the prediction into the original reference-image geometry.
    """
    modal_paths = {
        strip_nifti_suffix(path).rsplit("_", 1)[-1].lower(): path
        for path in subject_files(subject_dir)[0]
    }
    required = [*MODEL_LOW_MODALITIES, *MODEL_HIGH_MODALITIES]
    volumes: list[np.ndarray] = []
    for modality in required:
        base, band = modality.rsplit("_", 1)
        path = find_frequency_file(modal_paths[base], band)
        if path is None:
            raise FileNotFoundError(f"Missing required frequency volume for {modality} in {subject_dir}")
        volumes.append(load_volume(path))

    cropped_volumes, crop = foreground_center_crop(volumes)
    inputs = [torch.from_numpy(normalize_for_model(volume)).unsqueeze(0).unsqueeze(0) for volume in cropped_volumes]
    state_dict = torch.load(checkpoint, map_location="cpu", weights_only=True)
    num_classes = int(state_dict["l1_b1_f.weight"].shape[0])
    device = get_device()
    model = HFFNet(4, 16, num_classes).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.inference_mode():
        low = torch.cat(inputs[:4], dim=1).to(device)
        high = torch.cat(inputs[4:], dim=1).to(device)
        output_low, output_high, _, _ = model(low, high)
        # Training checkpoints named ``best_Result1`` were selected from the
        # low-frequency branch; all other existing checkpoint names select the
        # high-frequency branch, matching the repository's Result2 convention.
        output = output_low if "result1" in checkpoint.stem.lower() else output_high
        prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy().astype(np.uint8)

    # Keep categorical labels intact if a future architecture returns a
    # different spatial output size; never use linear interpolation for masks.
    target_crop_shape = tuple(axis_slice.stop - axis_slice.start for axis_slice in crop)
    prediction = resize_mask_to_shape(prediction, target_crop_shape).astype(np.uint8)

    # Restore the prediction to the ground-truth segmentation geometry.  In
    # BraTS these are normally identical to FLAIR, but using the segmentation
    # as the reference keeps the generated label aligned when a dataset has
    # different image and mask shapes or metadata.
    segmentation_path = subject_files(subject_dir)[1]
    reference_path = segmentation_path or modal_paths["flair"]
    reference_image = sitk.ReadImage(str(reference_path))
    reference_volume = sitk.GetArrayFromImage(reference_image).astype(np.float32)
    flair_volume = load_volume(modal_paths["flair"])
    if flair_volume.shape != reference_volume.shape:
        flair_volume = resize_volume_to_shape(flair_volume, reference_volume.shape)
    restored = np.zeros(tuple(reversed(reference_image.GetSize())), dtype=np.uint8)
    restored_crop = restored[3:, :, :]
    restored_crop[crop] = prediction
    restored = restrict_mask_to_scan_foreground(restored, flair_volume)
    output_path = checkpoint_output_path(results_root, checkpoint, subject_dir)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_image = sitk.GetImageFromArray(restored)
    output_image.CopyInformation(reference_image)
    sitk.WriteImage(output_image, str(output_path))
    return output_path


def find_frequency_file(scan_path: Path, band: str) -> Path | None:
    base = strip_nifti_suffix(scan_path)
    band = band.upper()
    # High-frequency generation writes four directional bands (H1-H4). Keep
    # support for the older single-file `_H` naming convention as a fallback.
    names = [band]
    if band in {"H1", "H2", "H3", "H4"}:
        names.append("H")
    elif band == "H":
        names = ["H1", "H2", "H3", "H4", "H"]
    candidates = [
        scan_path.with_name(f"{base}_{name}{suffix}")
        for name in names
        for suffix in (".nii.gz", ".nii")
    ]
    return next((path for path in candidates if path.exists()), None)


def load_frequency_volume(scan_path: Path, band: str, fallback: np.ndarray) -> tuple[np.ndarray, bool]:
    frequency_path = find_frequency_file(scan_path, band)
    if frequency_path is None:
        return np.zeros_like(fallback), False
    return resize_volume_to_shape(load_volume(frequency_path), fallback.shape), True


def enable_grid_layer_labels(layers: list[napari.layers.Layer]) -> None:
    """Show each layer name inside its grid cell."""
    for layer in layers:
        name_overlay = getattr(layer, "name_overlay", None)
        if name_overlay is None:
            continue
        name_overlay.visible = True
        name_overlay.gridded = True
        name_overlay.position = "top_left"


def setup_searchable_combo(combo: QComboBox) -> None:
    combo.setEditable(True)
    combo.setInsertPolicy(QComboBox.NoInsert)
    completer = QCompleter(combo.model(), combo)
    completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
    completer.setFilterMode(Qt.MatchFlag.MatchContains)
    completer.setCompletionMode(QCompleter.CompletionMode.PopupCompletion)
    combo.setCompleter(completer)
    if combo.lineEdit() is not None:
        combo.lineEdit().setPlaceholderText("Type to search")


class SubjectSelectorWidget(QWidget):
    def __init__(
        self,
        viewer: napari.Viewer,
        dataset_root: Path,
        initial_subject: Path,
        image_layers: list[napari.layers.Image],
        extra_layer: napari.layers.Image,
        extra_mask_layer: napari.layers.Labels,
        mask_layer: napari.layers.Labels | None,
        frequency_layers: list[napari.layers.Image],
        frequency_mask_layers: list[napari.layers.Labels],
        output_layers: list[napari.layers.Image | napari.layers.Labels],
        results_root: Path,
    ) -> None:
        super().__init__()
        # Keep the control dock compact so the image grid gets most of the
        # window width.
        self.setFixedWidth(320)
        self.viewer = viewer
        self.dataset_root = dataset_root
        self.image_layers = image_layers
        self.extra_layer = extra_layer
        self.extra_mask_layer = extra_mask_layer
        self.mask_layer = mask_layer
        self.frequency_layers = frequency_layers
        self.frequency_mask_layers = frequency_mask_layers
        self.output_layers = output_layers
        self.results_root = results_root
        self.current_scan_index: dict[str, Path] = {}
        self.current_mask: np.ndarray | None = None
        self.current_subject_dir: Path | None = None
        self.current_output_mask: np.ndarray | None = None
        self.checkpoint_index: dict[str, Path] = {}

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        mode_title = QLabel("View type")
        mode_title.setStyleSheet("font-weight: 600;")
        layout.addWidget(mode_title)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Input analysis", "Frequency decomposition", "Output Analysis"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        layout.addWidget(self.mode_combo)

        self.frequency_band_title = QLabel("High-frequency mode")
        self.frequency_band_title.setStyleSheet("font-weight: 600; margin-top: 6px;")
        layout.addWidget(self.frequency_band_title)

        self.frequency_band_combo = QComboBox()
        self.frequency_band_combo.addItems(["H1", "H2", "H3", "H4"])
        self.frequency_band_combo.currentTextChanged.connect(self.on_frequency_band_changed)
        layout.addWidget(self.frequency_band_combo)

        title = QLabel("Select record")
        title.setStyleSheet("font-weight: 600;")
        layout.addWidget(title)

        self.select_record_button = QPushButton("Select record folder…")
        self.select_record_button.clicked.connect(self.select_record_folder)
        layout.addWidget(self.select_record_button)

        self.selected_record = QLabel()
        self.selected_record.setWordWrap(True)
        self.selected_record.setStyleSheet("color: palette(mid); font-size: 11px;")
        layout.addWidget(self.selected_record)

        extra_title = QLabel("Actual scan type")
        extra_title.setStyleSheet("font-weight: 600; margin-top: 6px;")
        layout.addWidget(extra_title)

        self.extra_combo = QComboBox()
        setup_searchable_combo(self.extra_combo)
        self.extra_combo.currentTextChanged.connect(self.on_scan_changed)
        layout.addWidget(self.extra_combo)

        self.checkpoint_title = QLabel("Checkpoint")
        self.checkpoint_title.setStyleSheet("font-weight: 600; margin-top: 6px;")
        layout.addWidget(self.checkpoint_title)

        self.checkpoint_combo = QComboBox()
        setup_searchable_combo(self.checkpoint_combo)
        self.checkpoint_combo.currentTextChanged.connect(self.on_checkpoint_changed)
        layout.addWidget(self.checkpoint_combo)

        self.generate_button = QPushButton("Generate output segmentation")
        self.generate_button.clicked.connect(self.on_generate_output)
        layout.addWidget(self.generate_button)

        self.set_checkpoint_controls_visible(False)
        self.set_frequency_band_controls_visible(False)

        self.description = QLabel(
            "Input analysis shows the four scans, expected mask, and selected scan + expected. "
            "Frequency decomposition shows actual, low-frequency, and high-frequency views. "
            "Output Analysis compares the selected scan with its expected and generated masks."
        )
        self.description.setWordWrap(True)
        self.description.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.description)

        self.legend = QLabel(
            "Segmentation labels: "
            "■ Red — necrotic / non-enhancing core (1)   "
            "■ Blue — edema (2)   "
            "■ Yellow — enhancing tumour (4; model class 3)"
        )
        self.legend.setWordWrap(True)
        self.legend.setStyleSheet("font-size: 11px; margin-top: 6px;")
        layout.addWidget(self.legend)

        refresh = QPushButton("Reset view")
        refresh.clicked.connect(self.viewer.fit_to_view)
        layout.addWidget(refresh)

        layout.addStretch(1)

        self.on_subject_changed(initial_subject)

    def set_checkpoint_controls_visible(self, visible: bool) -> None:
        """Show checkpoint controls only for the output-analysis view."""
        for widget in (self.checkpoint_title, self.checkpoint_combo, self.generate_button):
            widget.setVisible(visible)

    def set_frequency_band_controls_visible(self, visible: bool) -> None:
        """Show the directional high-frequency selector only in frequency view."""
        for widget in (self.frequency_band_title, self.frequency_band_combo):
            widget.setVisible(visible)

    def select_record_folder(self) -> None:
        """Select and load one BraTS record without scanning the dataset root."""
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select BraTS record folder",
            str(self.dataset_root),
            QFileDialog.Option.ShowDirsOnly,
        )
        if not selected:
            return

        selected_path = Path(selected).expanduser().resolve()
        if not selected_path.is_dir() or subject_files(selected_path)[1] is None:
            self.description.setText(
                "That folder is not a BraTS record. Select a folder containing its _seg.nii or _seg.nii.gz file."
            )
            return

        self.on_subject_changed(selected_path)

    def refresh_checkpoint_options(self) -> None:
        """Re-scan ``result`` whenever Output Analysis is opened."""
        previous = self.checkpoint_combo.currentText()
        checkpoints = discover_checkpoints(self.results_root)
        self.checkpoint_index = {
            checkpoint.relative_to(self.results_root).as_posix(): checkpoint
            for checkpoint in checkpoints
        }
        self.checkpoint_combo.blockSignals(True)
        self.checkpoint_combo.clear()
        self.checkpoint_combo.addItems(self.checkpoint_index.keys())
        if previous in self.checkpoint_index:
            self.checkpoint_combo.setCurrentText(previous)
        self.checkpoint_combo.blockSignals(False)
        self.generate_button.setEnabled(bool(self.checkpoint_index))
        self.on_checkpoint_changed(self.checkpoint_combo.currentText())

    def refresh_output_layers(self) -> None:
        if not self.current_scan_index:
            return
        scan_name = self.extra_combo.currentText()
        if scan_name not in self.current_scan_index:
            return
        volume = load_volume(self.current_scan_index[scan_name])
        output = self.current_output_mask
        expected = self.current_mask
        expected_labels = np.zeros_like(volume, dtype=np.uint8) if expected is None else resize_mask_to_shape(expected, volume.shape).astype(np.uint8)
        output_labels = np.zeros_like(volume, dtype=np.uint8) if output is None else resize_mask_to_shape(output, volume.shape).astype(np.uint8)
        empty_image = np.zeros_like(volume, dtype=np.float32)
        empty_labels = np.zeros_like(volume, dtype=np.uint8)
        layer_data = [
            volume,
            empty_labels,
            volume,
            expected_labels,
            volume,
            output_labels,
            empty_image,
            expected_labels,
            empty_image,
            output_labels,
            empty_image,
            empty_labels,
        ]
        layer_names = [
            f"{scan_name} — input scan",
            "",
            f"{scan_name} — input + EXPECTED",
            "",
            f"{scan_name} — input + OUTPUT",
            "",
            "EXPECTED segmentation",
            "",
            "OUTPUT segmentation" if output is not None else "OUTPUT segmentation (generate to view)",
            "",
            "",
            "",
        ]
        for index, (layer, data, name) in enumerate(zip(self.output_layers, layer_data, layer_names)):
            layer.data = data
            layer.name = name
            if index in (0, 2, 4):
                layer.contrast_limits = contrast_limits(data)

    def on_checkpoint_changed(self, checkpoint_name: str) -> None:
        self.current_output_mask = None
        checkpoint = self.checkpoint_index.get(checkpoint_name)
        if checkpoint is not None and self.current_subject_dir is not None:
            output_path = checkpoint_output_path(self.results_root, checkpoint, self.current_subject_dir)
            if output_path.exists():
                output_mask = load_volume(output_path).astype(np.uint8)
                flair_path = self.current_scan_index.get("FLAIR")
                self.current_output_mask = (
                    restrict_mask_to_scan_foreground(output_mask, load_volume(flair_path))
                    if flair_path is not None else output_mask
                )
        self.refresh_output_layers()

    def on_generate_output(self) -> None:
        checkpoint = self.checkpoint_index.get(self.checkpoint_combo.currentText())
        if checkpoint is None or self.current_subject_dir is None:
            return
        self.generate_button.setEnabled(False)
        self.generate_button.setText("Generating output segmentation…")
        try:
            output_path = generate_output_segmentation(checkpoint, self.current_subject_dir, self.results_root)
            self.current_output_mask = load_volume(output_path).astype(np.uint8)
            self.refresh_output_layers()
            self.description.setText(f"Generated {output_path.relative_to(PROJECT_ROOT)}")
        except Exception as error:
            self.description.setText(f"Could not generate output segmentation: {error}")
        finally:
            self.generate_button.setText("Generate output segmentation")
            self.generate_button.setEnabled(bool(self.checkpoint_index))

    def refresh_scan_options(self) -> None:
        self.extra_combo.blockSignals(True)
        self.extra_combo.clear()
        self.extra_combo.addItems(list(self.current_scan_index.keys()))
        if self.current_scan_index:
            selected_name = next(reversed(self.current_scan_index))
            self.extra_combo.setCurrentText(selected_name)
        self.extra_combo.blockSignals(False)
        if self.current_scan_index:
            self.on_scan_changed(self.extra_combo.currentText())

    def on_scan_changed(self, scan_name: str) -> None:
        if scan_name not in self.current_scan_index:
            self.extra_layer.visible = False
            self.extra_mask_layer.visible = False
            return

        selected_path = self.current_scan_index[scan_name]
        volume = load_volume(selected_path)

        self.extra_layer.data = volume
        self.extra_layer.contrast_limits = contrast_limits(volume)
        self.extra_layer.name = f"{scan_name} + EXPECTED"
        self.extra_mask_layer.data = (
            np.zeros_like(volume, dtype=np.uint8)
            if self.current_mask is None
            else resize_mask_to_shape(self.current_mask, volume.shape).astype(np.uint8)
        )
        self.extra_layer.visible = True
        self.extra_mask_layer.visible = True
        self.refresh_frequency_layers(scan_name, volume)
        self.refresh_output_layers()
        self.viewer.reset_view()

    def on_frequency_band_changed(self, band: str) -> None:
        """Refresh the high-frequency row when H1–H4 selection changes."""
        if not band or not self.current_scan_index:
            return
        scan_name = self.extra_combo.currentText()
        if scan_name in self.current_scan_index:
            self.refresh_frequency_layers(scan_name, load_volume(self.current_scan_index[scan_name]))
            if self.mode_combo.currentText() == "Frequency decomposition":
                self.viewer.reset_view()

    def refresh_frequency_layers(self, scan_name: str, actual_volume: np.ndarray) -> None:
        scan_path = self.current_scan_index[scan_name]
        low_volume, low_available = load_frequency_volume(scan_path, "L", actual_volume)
        selected_band = self.frequency_band_combo.currentText() or "H1"
        high_volume, high_available = load_frequency_volume(scan_path, selected_band, actual_volume)
        low_volume = suppress_background(low_volume, actual_volume)
        high_volume = suppress_background(high_volume, actual_volume)

        low_label = "low frequency" if low_available else "low frequency (not available)"
        high_label = f"{selected_band} high frequency" if high_available else f"{selected_band} high frequency (not generated)"
        layer_data = [
            actual_volume,
            low_volume,
            low_volume,
            actual_volume,
            high_volume,
            high_volume,
        ]
        layer_names = [
            f"{scan_name} — actual",
            f"{scan_name} — {low_label}",
            f"{scan_name} — {low_label} + EXPECTED",
            f"{scan_name} — actual (high row)",
            f"{scan_name} — {high_label}",
            f"{scan_name} — {high_label} + EXPECTED",
        ]
        for layer, data, name in zip(self.frequency_layers, layer_data, layer_names):
            layer.data = data
            layer.name = name

        for layer in self.frequency_mask_layers:
            layer.data = (
                np.zeros_like(actual_volume, dtype=np.uint8)
                if self.current_mask is None
                else resize_mask_to_shape(self.current_mask, actual_volume.shape).astype(np.uint8)
            )

        for index in range(6):
            self.frequency_layers[index].contrast_limits = contrast_limits(layer_data[index])

    def on_mode_changed(self, mode: str) -> None:
        frequency_mode = mode == "Frequency decomposition"
        output_mode = mode == "Output Analysis"
        self.set_checkpoint_controls_visible(output_mode)
        self.set_frequency_band_controls_visible(frequency_mode)
        for layer in [*self.image_layers, self.extra_layer, self.extra_mask_layer]:
            layer.visible = not frequency_mode and not output_mode
        if self.mask_layer is not None:
            self.mask_layer.visible = not frequency_mode and not output_mode
        for layer in [*self.frequency_layers, *self.frequency_mask_layers]:
            layer.visible = frequency_mode
        for layer in self.output_layers:
            layer.visible = output_mode
        if output_mode:
            self.viewer.grid.stride = 2
            self.refresh_checkpoint_options()
        elif frequency_mode:
            self.viewer.grid.stride = 2
        else:
            self.viewer.grid.stride = 2
        self.description.setText(
            f"Frequency decomposition: actual scan, low-frequency scan, low-frequency + expected, "
            f"then the selected {self.frequency_band_combo.currentText() or 'H1'} high-frequency row."
            if frequency_mode
            else (
                "Output Analysis: selected input scan, expected segmentation, and output segmentation. "
                "Checkpoint options refresh from the result folder each time this view is opened."
                if output_mode
                else "Input analysis: the four scans, expected mask, and selected scan + expected."
            )
        )
        self.viewer.reset_view()

    def on_subject_changed(self, subject_dir: Path) -> None:
        subject_dir = subject_dir.expanduser().resolve()
        if subject_files(subject_dir)[1] is None:
            return

        self.selected_record.setText(f"Selected: {subject_dir}")
        volumes, mask = load_subject(subject_dir)
        scan_paths, _ = subject_files(subject_dir)
        scan_names = list(volumes.keys())
        self.current_scan_index = {name: path for name, path in zip(scan_names, scan_paths)}
        self.current_mask = mask
        self.current_subject_dir = subject_dir
        self.current_output_mask = None

        for layer, scan_name in zip(self.image_layers, scan_names):
            volume = volumes[scan_name]
            layer.data = volume
            layer.contrast_limits = contrast_limits(volume)
            layer.name = scan_name
            layer.visible = True

        for layer in self.image_layers[len(scan_names) :]:
            layer.visible = False

        if self.mask_layer is not None:
            if mask is None:
                self.mask_layer.visible = False
            else:
                self.mask_layer.data = mask.astype(np.uint8)
                self.mask_layer.visible = True

        target_shape = volumes[scan_names[0]].shape
        self.extra_mask_layer.data = (
            np.zeros(target_shape, dtype=np.uint8)
            if mask is None
            else resize_mask_to_shape(mask, target_shape).astype(np.uint8)
        )

        self.refresh_scan_options()
        self.on_mode_changed(self.mode_combo.currentText())
        self.viewer.title = f"BraTS viewer — {subject_dir.name}"
        self.viewer.reset_view()


def add_subject_layers(
    viewer: napari.Viewer,
    dataset_root: Path,
    initial_subject: Path,
    results_root: Path,
) -> None:
    viewer.layers.clear()

    volumes, mask = load_subject(initial_subject)

    image_layers: list[napari.layers.Image] = []
    input_grid_padding_layers: list[napari.layers.Image] = []
    for scan_name, volume in volumes.items():
        image_layers.append(
            viewer.add_image(
                volume,
                name=scan_name,
                colormap="gray",
                contrast_limits=contrast_limits(volume),
                rendering="mip",
                opacity=0.8,
                blending="translucent",
            )
        )
        # Input Analysis uses grid stride 2 so a scan and its optional overlay
        # can share a cell. Keep one hidden partner after each scan tile.
        input_grid_padding_layers.append(
            viewer.add_image(
                np.zeros_like(volume, dtype=np.float32),
                name="input grid padding",
                visible=False,
                rendering="mip",
            )
        )

    initial_scan_name = next(reversed(volumes))
    extra_layer = viewer.add_image(
        volumes[initial_scan_name],
        name=f"{initial_scan_name} + EXPECTED",
        contrast_limits=contrast_limits(volumes[initial_scan_name]),
        rendering="mip",
    )
    extra_mask_layer = viewer.add_labels(
        np.zeros_like(volumes[initial_scan_name], dtype=np.uint8)
        if mask is None
        else mask.astype(np.uint8),
        name="",
        colormap=BRATS_LABEL_COLORMAP,
        rendering="iso_categorical",
        opacity=0.65,
        visible=True,
    )

    # Keep the standalone expected mask at the sixth Input Analysis tile.
    mask_grid_padding = viewer.add_image(
        np.zeros_like(volumes[initial_scan_name], dtype=np.float32),
        name="input grid padding",
        visible=False,
        rendering="mip",
    )
    mask_layer = None
    if mask is not None:
        mask_layer = viewer.add_labels(
            mask.astype(np.uint8),
            name="EXPECTED MASK",
            colormap=BRATS_LABEL_COLORMAP,
            rendering="iso_categorical",
            opacity=0.45,
            blending="translucent",
        )

    initial_scan_path = next(path for path in subject_files(initial_subject)[0] if strip_nifti_suffix(path).rsplit("_", 1)[-1].upper() == initial_scan_name)
    initial_low, initial_low_available = load_frequency_volume(
        initial_scan_path,
        "L",
        volumes[initial_scan_name],
    )
    initial_high, initial_high_available = load_frequency_volume(
        initial_scan_path,
        "H1",
        volumes[initial_scan_name],
    )
    initial_low_label = "low frequency" if initial_low_available else "low frequency (not available)"
    initial_high_label = "H1 high frequency" if initial_high_available else "H1 high frequency (not generated)"
    frequency_layers: list[napari.layers.Image] = []
    frequency_mask_layers: list[napari.layers.Labels] = []

    def add_frequency_image(data: np.ndarray, name: str) -> napari.layers.Image:
        layer = viewer.add_image(data, name=name, visible=False, rendering="mip")
        frequency_layers.append(layer)
        return layer

    def add_frequency_padding(data: np.ndarray) -> None:
        viewer.add_image(
            np.zeros_like(data, dtype=np.float32),
            name="frequency grid padding",
            visible=False,
            rendering="mip",
        )

    add_frequency_image(volumes[initial_scan_name], f"{initial_scan_name} — actual")
    add_frequency_padding(volumes[initial_scan_name])
    add_frequency_image(initial_low, f"{initial_scan_name} — {initial_low_label}")
    add_frequency_padding(initial_low)
    add_frequency_image(initial_low, f"{initial_scan_name} — {initial_low_label} + EXPECTED")
    frequency_mask_layers.append(
        viewer.add_labels(
            np.zeros_like(initial_low, dtype=np.uint8) if mask is None else mask.astype(np.uint8),
            name="",
            colormap=BRATS_LABEL_COLORMAP,
            opacity=0.65,
            visible=False,
            rendering="iso_categorical",
        )
    )
    add_frequency_image(volumes[initial_scan_name], f"{initial_scan_name} — actual (high row)")
    add_frequency_padding(volumes[initial_scan_name])
    add_frequency_image(initial_high, f"{initial_scan_name} — {initial_high_label}")
    add_frequency_padding(initial_high)
    add_frequency_image(initial_high, f"{initial_scan_name} — {initial_high_label} + EXPECTED")
    frequency_mask_layers.append(
        viewer.add_labels(
            np.zeros_like(initial_high, dtype=np.uint8) if mask is None else mask.astype(np.uint8),
            name="",
            colormap=BRATS_LABEL_COLORMAP,
            opacity=0.65,
            visible=False,
            rendering="iso_categorical",
        )
    )

    # Output Analysis uses paired layers with grid stride 2. Each tile gets an
    # MRI/base image followed by a categorical Labels layer, so the combined
    # panels contain the exact same segmentation blob as the standalone masks.
    initial_empty_mask = np.zeros_like(volumes[initial_scan_name], dtype=np.uint8)
    viewer.add_image(
        initial_empty_mask.astype(np.float32),
        name="output grid padding",
        visible=False,
        rendering="mip",
    )
    viewer.add_image(
        initial_empty_mask.astype(np.float32),
        name="output grid padding",
        visible=False,
        rendering="mip",
    )
    output_layers = [
        viewer.add_image(
            volumes[initial_scan_name],
            name=f"{initial_scan_name} — input scan",
            visible=False,
            rendering="mip",
        ),
        viewer.add_labels(initial_empty_mask, name="", colormap=BRATS_LABEL_COLORMAP, visible=False, rendering="iso_categorical"),
        viewer.add_image(
            volumes[initial_scan_name],
            name=f"{initial_scan_name} — input + EXPECTED",
            visible=False,
            rendering="mip",
        ),
        viewer.add_labels(
            initial_empty_mask if mask is None else mask.astype(np.uint8),
            name="",
            colormap=BRATS_LABEL_COLORMAP,
            opacity=0.65,
            visible=False,
            rendering="iso_categorical",
        ),
        viewer.add_image(
            volumes[initial_scan_name],
            name=f"{initial_scan_name} — input + OUTPUT",
            visible=False,
            rendering="mip",
        ),
        viewer.add_labels(initial_empty_mask, name="", colormap=BRATS_LABEL_COLORMAP, opacity=0.65, visible=False, rendering="iso_categorical"),
        viewer.add_image(initial_empty_mask.astype(np.float32), name="EXPECTED segmentation", visible=False, rendering="mip"),
        viewer.add_labels(
            initial_empty_mask if mask is None else mask.astype(np.uint8),
            name="",
            colormap=BRATS_LABEL_COLORMAP,
            visible=False,
            rendering="iso_categorical",
        ),
        viewer.add_image(initial_empty_mask.astype(np.float32), name="OUTPUT segmentation", visible=False, rendering="mip"),
        viewer.add_labels(
            initial_empty_mask,
            name="",
            colormap=BRATS_LABEL_COLORMAP,
            visible=False,
            rendering="iso_categorical",
        ),
        viewer.add_image(initial_empty_mask.astype(np.float32), name="", visible=False, rendering="mip"),
        viewer.add_labels(initial_empty_mask, name="", colormap=BRATS_LABEL_COLORMAP, visible=False, rendering="iso_categorical"),
    ]

    enable_grid_layer_labels(
        [
            *image_layers,
            extra_layer,
            extra_mask_layer,
            *frequency_layers,
            *frequency_mask_layers,
            *output_layers,
            *input_grid_padding_layers,
            mask_grid_padding,
        ]
        + ([mask_layer] if mask_layer is not None else [])
    )
    # In Output Analysis, each grid cell has an image and a Labels layer.
    # Only the image/base layer should contribute a tile title.
    for layer in output_layers[1::2]:
        layer.name_overlay.visible = False

    selector = SubjectSelectorWidget(
        viewer,
        dataset_root,
        initial_subject,
        image_layers,
        extra_layer,
        extra_mask_layer,
        mask_layer,
        frequency_layers,
        frequency_mask_layers,
        output_layers,
        results_root,
    )
    viewer.window.add_dock_widget(
        selector,
        name="subject selector",
        area="left",
        allowed_areas=["left", "right"],
    )

    # Keep the layer list available, but hide Napari's per-layer controls so
    # the left sidebar only contains the layers list and subject selector.
    # ``dockLayerControls`` is separate from ``dockLayerList`` in Napari's Qt
    # viewer, so hiding it does not affect layer visibility or selection.
    layer_controls_dock = viewer.window._qt_viewer.dockLayerControls
    layer_controls_dock.setVisible(False)
    layer_controls_dock.toggleViewAction().setVisible(False)

    # Both view modes deliberately use the same 2×3 arrangement:
    # input-analysis layers or frequency-decomposition layers fill the grid
    # in their declared order.
    viewer.grid.shape = (2, 3)
    viewer.grid.stride = 2
    viewer.grid.enabled = True
    viewer.dims.ndisplay = 3
    viewer.dims.order = (0, 1, 2)
    viewer.scale_bar.visible = True
    viewer.dims.axis_labels = ("z", "y", "x")
    viewer.fit_to_view()
    viewer.title = f"BraTS viewer — {initial_subject.name}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--results-root",
        type=Path,
        default=DEFAULT_RESULTS_ROOT,
        help="Directory containing model checkpoints and generated output masks.",
    )
    parser.add_argument("--subject", help="Subject folder name or relative path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    results_root = args.results_root.expanduser().resolve()
    viewer = napari.Viewer(ndisplay=3, title="BraTS Napari viewer")

    if args.subject:
        requested = Path(args.subject).expanduser()
        initial_subject = requested if requested.is_absolute() else dataset_root / requested
    else:
        selected = QFileDialog.getExistingDirectory(
            None,
            "Select BraTS record folder",
            str(dataset_root),
            QFileDialog.Option.ShowDirsOnly,
        )
        if not selected:
            viewer.close()
            raise SystemExit("No BraTS record selected.")
        initial_subject = Path(selected).expanduser().resolve()

    if not initial_subject.is_dir() or subject_files(initial_subject)[1] is None:
        viewer.close()
        raise SystemExit(
            f"Not a BraTS record folder: {initial_subject}. "
            "Expected a folder containing _seg.nii or _seg.nii.gz."
        )

    add_subject_layers(viewer, dataset_root, initial_subject, results_root)
    napari.run()


if __name__ == "__main__":
    main()
