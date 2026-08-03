"""Interactive Napari viewer for input and frequency-decomposition analysis."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import napari
import numpy as np
import SimpleITK as sitk
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QComboBox, QCompleter, QLabel, QPushButton, QVBoxLayout, QWidget
from napari.utils.colormaps import Colormap


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "dataset"
MODALITY_PRIORITY = ("t1", "t1ce", "t2", "flair")


def strip_nifti_suffix(path: Path) -> str:
    if path.name.endswith(".nii.gz"):
        return path.name[:-7]
    return path.stem


def is_nifti_file(path: Path) -> bool:
    return path.name.endswith(".nii") or path.name.endswith(".nii.gz")


def is_frequency_file(path: Path) -> bool:
    stem = strip_nifti_suffix(path).lower()
    return stem.rsplit("_", 1)[-1] in {"l", "h", "h1", "h2", "h3", "h4"}


def discover_subject_dirs(dataset_root: Path) -> list[Path]:
    subjects = []
    for current_dir, _, files in os.walk(dataset_root):
        paths = [Path(current_dir) / name for name in files]
        if any(path.name.endswith("_seg.nii") or path.name.endswith("_seg.nii.gz") for path in paths):
            subjects.append(Path(current_dir))
    return sorted(set(subjects))


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


def build_expected_overlay(volume: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    """Encode a scan and expected mask for Napari's 3D scalar renderer.

    Napari's 3D MIP renderer treats RGB volumes as scalar data.  Reserve the
    top part of the scalar range for the expected mask and map that range to
    red with ``EXPECTED_BLEND_COLORMAP``.
    """
    lower, upper = contrast_limits(volume)
    if upper <= lower:
        normalized = np.zeros_like(volume, dtype=np.float32)
    else:
        normalized = np.clip((volume - lower) / (upper - lower), 0, 1)

    overlay = normalized * 0.72
    if mask is not None:
        expected = resize_mask_to_shape(mask, volume.shape) > 0
        overlay[expected] = 1.0
    return overlay.astype(np.float32)


EXPECTED_BLEND_COLORMAP = Colormap(
    colors=np.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.72, 0.72, 0.72, 1.0],
            [1.0, 0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    ),
    controls=np.array([0.0, 0.72, 0.73, 1.0], dtype=np.float32),
    name="scan + expected",
)


def find_frequency_file(scan_path: Path, band: str) -> Path | None:
    base = strip_nifti_suffix(scan_path)
    band = band.upper()
    # High-frequency generation writes four directional bands (H1-H4), while
    # the viewer has one high-frequency panel. Use H1 for that panel and keep
    # support for the older single-file `_H` naming convention.
    names = [band]
    if band == "H":
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
        subject_index: dict[str, Path],
        image_layers: list[napari.layers.Image],
        extra_layer: napari.layers.Image,
        mask_layer: napari.layers.Labels | None,
        frequency_layers: list[napari.layers.Image],
    ) -> None:
        super().__init__()
        # Keep the control dock compact so the image grid gets most of the
        # window width. Long record paths are still searchable in the combo.
        self.setFixedWidth(320)
        self.viewer = viewer
        self.dataset_root = dataset_root
        self.subject_index = subject_index
        self.image_layers = image_layers
        self.extra_layer = extra_layer
        self.mask_layer = mask_layer
        self.frequency_layers = frequency_layers
        self.current_scan_index: dict[str, Path] = {}
        self.current_mask: np.ndarray | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        mode_title = QLabel("View type")
        mode_title.setStyleSheet("font-weight: 600;")
        layout.addWidget(mode_title)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Input analysis", "Frequency decomposition"])
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        layout.addWidget(self.mode_combo)

        title = QLabel("Select record")
        title.setStyleSheet("font-weight: 600;")
        layout.addWidget(title)

        self.combo = QComboBox()
        self.combo.addItems(list(self.subject_index.keys()))
        setup_searchable_combo(self.combo)
        self.combo.currentTextChanged.connect(self.on_subject_changed)
        layout.addWidget(self.combo)

        extra_title = QLabel("Actual scan type")
        extra_title.setStyleSheet("font-weight: 600; margin-top: 6px;")
        layout.addWidget(extra_title)

        self.extra_combo = QComboBox()
        setup_searchable_combo(self.extra_combo)
        self.extra_combo.currentTextChanged.connect(self.on_scan_changed)
        layout.addWidget(self.extra_combo)

        self.description = QLabel(
            "Input analysis shows the four scans, expected mask, and selected scan + expected. "
            "Frequency decomposition shows actual, low-frequency, and high-frequency views."
        )
        self.description.setWordWrap(True)
        self.description.setAlignment(Qt.AlignmentFlag.AlignTop)
        layout.addWidget(self.description)

        refresh = QPushButton("Reset view")
        refresh.clicked.connect(self.viewer.fit_to_view)
        layout.addWidget(refresh)

        layout.addStretch(1)

        initial_subject_name = next(iter(self.subject_index.keys()))
        self.combo.setCurrentText(initial_subject_name)
        self.on_subject_changed(initial_subject_name)

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
            return

        selected_path = self.current_scan_index[scan_name]
        volume = load_volume(selected_path)

        self.extra_layer.data = build_expected_overlay(volume, self.current_mask)
        self.extra_layer.name = f"{scan_name} + EXPECTED"
        self.extra_layer.visible = True
        self.refresh_frequency_layers(scan_name, volume)
        self.viewer.reset_view()

    def refresh_frequency_layers(self, scan_name: str, actual_volume: np.ndarray) -> None:
        scan_path = self.current_scan_index[scan_name]
        low_volume, low_available = load_frequency_volume(scan_path, "L", actual_volume)
        high_volume, high_available = load_frequency_volume(scan_path, "H", actual_volume)
        low_volume = suppress_background(low_volume, actual_volume)
        high_volume = suppress_background(high_volume, actual_volume)

        low_label = "low frequency" if low_available else "low frequency (not available)"
        high_label = "high frequency" if high_available else "high frequency (not generated)"
        layer_data = [
            actual_volume,
            low_volume,
            build_expected_overlay(low_volume, self.current_mask),
            actual_volume,
            high_volume,
            build_expected_overlay(high_volume, self.current_mask),
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

        for index in (0, 1, 3, 4):
            self.frequency_layers[index].contrast_limits = contrast_limits(layer_data[index])
        for index in (2, 5):
            self.frequency_layers[index].contrast_limits = (0.0, 1.0)

    def on_mode_changed(self, mode: str) -> None:
        frequency_mode = mode == "Frequency decomposition"
        for layer in [*self.image_layers, self.extra_layer]:
            layer.visible = not frequency_mode
        if self.mask_layer is not None:
            self.mask_layer.visible = not frequency_mode
        for layer in self.frequency_layers:
            layer.visible = frequency_mode
        self.description.setText(
            "Frequency decomposition: actual scan, low-frequency scan, low-frequency + expected, "
            "then the corresponding high-frequency row."
            if frequency_mode
            else "Input analysis: the four scans, expected mask, and selected scan + expected."
        )
        self.viewer.reset_view()

    def on_subject_changed(self, subject_name: str) -> None:
        # QComboBox emits currentTextChanged for every character typed into
        # the editable search field. Ignore partial/non-matching text until
        # the user picks a real subject from the completer.
        if subject_name not in self.subject_index:
            return

        subject_dir = self.subject_index[subject_name]
        volumes, mask = load_subject(subject_dir)
        scan_paths, _ = subject_files(subject_dir)
        scan_names = list(volumes.keys())
        self.current_scan_index = {name: path for name, path in zip(scan_names, scan_paths)}
        self.current_mask = mask

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
                self.mask_layer.data = mask
                self.mask_layer.visible = True

        self.refresh_scan_options()
        self.on_mode_changed(self.mode_combo.currentText())
        self.viewer.title = f"BraTS viewer — {subject_dir.name}"
        self.viewer.reset_view()


def add_subject_layers(viewer: napari.Viewer, dataset_root: Path, initial_subject: Path) -> None:
    viewer.layers.clear()

    subject_index = {subject.relative_to(dataset_root).as_posix(): subject for subject in discover_subject_dirs(dataset_root)}
    volumes, mask = load_subject(initial_subject)

    image_layers: list[napari.layers.Image] = []
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

    initial_scan_name = next(reversed(volumes))
    extra_layer = viewer.add_image(
        build_expected_overlay(volumes[initial_scan_name], mask),
        name=f"{initial_scan_name} + EXPECTED",
        colormap=EXPECTED_BLEND_COLORMAP,
        contrast_limits=(0.0, 1.0),
        rendering="mip",
        opacity=0.6,
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
        "H",
        volumes[initial_scan_name],
    )
    initial_low_label = "low frequency" if initial_low_available else "low frequency (not available)"
    initial_high_label = "high frequency" if initial_high_available else "high frequency (not generated)"
    frequency_layers = [
        viewer.add_image(volumes[initial_scan_name], name=f"{initial_scan_name} — actual", visible=False, rendering="mip"),
        viewer.add_image(initial_low, name=f"{initial_scan_name} — {initial_low_label}", visible=False, rendering="mip"),
        viewer.add_image(
            build_expected_overlay(initial_low, mask),
            name=f"{initial_scan_name} — {initial_low_label} + EXPECTED",
            colormap=EXPECTED_BLEND_COLORMAP,
            contrast_limits=(0.0, 1.0),
            visible=False,
            rendering="mip",
        ),
        viewer.add_image(
            volumes[initial_scan_name],
            name=f"{initial_scan_name} — actual (high row)",
            visible=False,
            rendering="mip",
        ),
        viewer.add_image(initial_high, name=f"{initial_scan_name} — {initial_high_label}", visible=False, rendering="mip"),
        viewer.add_image(
            build_expected_overlay(initial_high, mask),
            name=f"{initial_scan_name} — {initial_high_label} + EXPECTED",
            colormap=EXPECTED_BLEND_COLORMAP,
            contrast_limits=(0.0, 1.0),
            visible=False,
            rendering="mip",
        ),
    ]

    mask_layer = None
    if mask is not None:
        mask_layer = viewer.add_labels(
            mask,
            name="EXPECTED MASK",
            opacity=0.45,
            blending="translucent",
        )

    enable_grid_layer_labels(
        [*image_layers, extra_layer, *frequency_layers] + ([mask_layer] if mask_layer is not None else [])
    )

    selector = SubjectSelectorWidget(
        viewer,
        dataset_root,
        subject_index,
        image_layers,
        extra_layer,
        mask_layer,
        frequency_layers,
    )
    viewer.window.add_dock_widget(
        selector,
        name="subject selector",
        area="left",
        allowed_areas=["left", "right"],
    )

    # Both view modes deliberately use the same 2×3 arrangement:
    # input-analysis layers or frequency-decomposition layers fill the grid
    # in their declared order.
    viewer.grid.shape = (2, 3)
    viewer.grid.stride = 1
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
    parser.add_argument("--subject", help="Subject folder name or relative path")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.expanduser().resolve()
    subjects = discover_subject_dirs(dataset_root)
    if not subjects:
        raise SystemExit(f"No BraTS subjects found under {dataset_root}")

    if args.subject:
        requested = Path(args.subject)
        matching_subjects = [path for path in subjects if path.name == args.subject or path.relative_to(dataset_root) == requested]
        if not matching_subjects:
            raise SystemExit(f"Subject not found: {args.subject}")
        initial_subject = matching_subjects[0]
    else:
        initial_subject = subjects[0]

    viewer = napari.Viewer(ndisplay=3, title="BraTS Napari viewer")
    add_subject_layers(viewer, dataset_root, initial_subject)
    napari.run()


if __name__ == "__main__":
    main()
