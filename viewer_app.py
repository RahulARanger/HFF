from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
import SimpleITK as sitk
from plotly.subplots import make_subplots


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = PROJECT_ROOT / "dataset" / "brats2019" / "extracted" / "MICCAI_BraTS_2019_Data_Training"
MODALITY_PRIORITY = (
    "flair",
    "t1ce",
    "t1",
    "t2",
)


def is_nifti_file(path: Path) -> bool:
    return path.name.endswith(".nii") or path.name.endswith(".nii.gz")


def strip_nifti_suffix(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    return path.stem


@st.cache_data(show_spinner=False)
def discover_subject_dirs(dataset_root: str) -> list[str]:
    root = Path(dataset_root).expanduser()
    if not root.exists():
        return []

    subject_dirs: list[str] = []
    for current_dir, _, files in os.walk(root):
        nifti_files = [name for name in files if name.endswith(".nii") or name.endswith(".nii.gz")]
        if not nifti_files:
            continue
        if any(name.endswith("_seg.nii") or name.endswith("_seg.nii.gz") for name in nifti_files):
            subject_dirs.append(str(Path(current_dir)))

    return sorted(set(subject_dirs))


@st.cache_data(show_spinner=False)
def list_nifti_files(subject_dir: str) -> list[str]:
    folder = Path(subject_dir)
    files = [path for path in folder.iterdir() if path.is_file() and is_nifti_file(path)]

    def sort_key(path: Path) -> tuple[int, str]:
        stem = strip_nifti_suffix(path).lower()
        if stem.endswith("_seg"):
            return (1, stem)
        for index, priority in enumerate(MODALITY_PRIORITY):
            if stem.endswith(f"_{priority.lower()}"):
                return (0, f"{index:02d}_{stem}")
        return (0, f"99_{stem}")

    return [str(path) for path in sorted(files, key=sort_key)]


@st.cache_data(show_spinner=False)
def load_nifti(file_path: str) -> np.ndarray:
    image = sitk.ReadImage(file_path)
    return sitk.GetArrayFromImage(image).astype(np.float32)


def parse_display_name(file_path: str, subject_dir: str) -> str:
    path = Path(file_path)
    subject_name = Path(subject_dir).name
    stem = strip_nifti_suffix(path)
    if stem.startswith(subject_name + "_"):
        return stem[len(subject_name) + 1 :]
    return stem


def normalize_volume(volume: np.ndarray) -> np.ndarray:
    values = volume[np.isfinite(volume)]
    if values.size == 0:
        return np.zeros_like(volume, dtype=np.float32)

    lower, upper = np.percentile(values, [1, 99])
    if upper <= lower:
        return np.zeros_like(volume, dtype=np.float32)

    clipped = np.clip(volume, lower, upper)
    return ((clipped - lower) / (upper - lower)).astype(np.float32)


def downsample_volume(volume: np.ndarray, max_dim: int = 72) -> np.ndarray:
    steps = [max(1, int(np.ceil(size / max_dim))) for size in volume.shape]
    return volume[::steps[0], ::steps[1], ::steps[2]]


def subject_default_files(file_paths: list[str]) -> list[str]:
    non_seg = [path for path in file_paths if "_seg" not in Path(path).name]
    selected: list[str] = []
    for priority in MODALITY_PRIORITY:
        match = next(
            (
                path
                for path in non_seg
                if strip_nifti_suffix(Path(path)).lower().endswith(f"_{priority.lower()}")
            ),
            None,
        )
        if match and match not in selected:
            selected.append(match)
    if len(selected) < len(MODALITY_PRIORITY):
        for path in non_seg:
            if path not in selected:
                selected.append(path)
            if len(selected) == len(MODALITY_PRIORITY):
                break
    return selected[: len(MODALITY_PRIORITY)]


def crop_to_foreground(mask: np.ndarray, padding: int = 8) -> np.ndarray | None:
    foreground = np.argwhere(mask > 0)
    if foreground.size == 0:
        return None

    lower = foreground.min(axis=0)
    upper = foreground.max(axis=0) + 1
    lower = np.maximum(lower - padding, 0)
    upper = np.minimum(upper + padding, mask.shape)

    return mask[lower[0] : upper[0], lower[1] : upper[1], lower[2] : upper[2]]


def build_volume_figure(volume: np.ndarray, title: str) -> go.Figure:
    sampled = downsample_volume(volume)
    normalized = normalize_volume(sampled)
    z_idx, y_idx, x_idx = np.indices(sampled.shape)

    figure = go.Figure(
        data=go.Volume(
            x=x_idx.flatten(),
            y=y_idx.flatten(),
            z=z_idx.flatten(),
            value=normalized.flatten(),
            isomin=0.05,
            isomax=0.95,
            opacity=0.09,
            surface_count=16,
            colorscale="Gray",
            caps=dict(x_show=False, y_show=False, z_show=False),
            showscale=False,
        )
    )
    figure.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=40, b=0),
        height=620,
        scene=dict(aspectmode="data", xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
    )
    return figure


def build_surface_figure(mask: np.ndarray, title: str) -> go.Figure | None:
    cropped = crop_to_foreground((mask > 0).astype(np.uint8))
    if cropped is None:
        return None

    sampled = downsample_volume(cropped, max_dim=96)
    z_idx, y_idx, x_idx = np.indices(sampled.shape)

    figure = go.Figure(
        data=go.Isosurface(
            x=x_idx.flatten(),
            y=y_idx.flatten(),
            z=z_idx.flatten(),
            value=sampled.flatten(),
            isomin=0.5,
            isomax=1.0,
            surface_count=1,
            caps=dict(x_show=False, y_show=False, z_show=False),
            colorscale=[[0.0, "rgba(255,80,80,0.8)"], [1.0, "rgba(255,80,80,0.8)"]],
            showscale=False,
            opacity=0.8,
        )
    )
    figure.update_layout(
        title=title,
        margin=dict(l=0, r=0, t=40, b=0),
        height=620,
        scene=dict(aspectmode="data", xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
    )
    return figure


def normalize_slice(slice_2d: np.ndarray) -> np.ndarray:
    values = slice_2d[np.isfinite(slice_2d)]
    if values.size == 0:
        return np.zeros_like(slice_2d, dtype=np.float32)

    lower, upper = np.percentile(values, [1, 99])
    if upper <= lower:
        return np.zeros_like(slice_2d, dtype=np.float32)

    return np.clip((slice_2d - lower) / (upper - lower), 0, 1).astype(np.float32)


def build_slice_figure(
    volume: np.ndarray,
    title: str,
    slice_fraction: float,
    mask: np.ndarray | None = None,
) -> go.Figure:
    z_idx = round(slice_fraction * (volume.shape[0] - 1))
    y_idx = round(slice_fraction * (volume.shape[1] - 1))
    x_idx = round(slice_fraction * (volume.shape[2] - 1))

    slices = [
        ("Axial", np.rot90(volume[z_idx, :, :])),
        ("Coronal", np.rot90(volume[:, y_idx, :])),
        ("Sagittal", np.rot90(volume[:, :, x_idx])),
    ]
    mask_slices = None
    if mask is not None and mask.shape == volume.shape:
        mask_slices = [
            np.rot90(mask[z_idx, :, :]),
            np.rot90(mask[:, y_idx, :]),
            np.rot90(mask[:, :, x_idx]),
        ]

    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[label for label, _ in slices],
        horizontal_spacing=0.025,
    )
    for column, (_, image_slice) in enumerate(slices, start=1):
        figure.add_trace(
            go.Heatmap(
                z=normalize_slice(image_slice),
                zmin=0,
                zmax=1,
                colorscale="Gray",
                showscale=False,
                hoverinfo="skip",
                zsmooth=False,
            ),
            row=1,
            col=column,
        )
        if mask_slices is not None:
            figure.add_trace(
                go.Contour(
                    z=(mask_slices[column - 1] > 0).astype(np.uint8),
                    contours=dict(start=0.5, end=0.5, size=1, coloring="lines"),
                    line=dict(color="#ff3b30", width=2),
                    showscale=False,
                    hoverinfo="skip",
                ),
                row=1,
                col=column,
            )

    figure.update_layout(
        title=dict(text=title, x=0.02, xanchor="left", font=dict(size=16)),
        margin=dict(l=0, r=0, t=48, b=0),
        height=380,
        showlegend=False,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    figure.update_xaxes(showticklabels=False, showgrid=False, zeroline=False, fixedrange=True)
    figure.update_yaxes(showticklabels=False, showgrid=False, zeroline=False, fixedrange=True, scaleanchor="x")
    return figure


def scan_modalities(file_paths: list[str]) -> list[str]:
    return [path for path in file_paths if "_seg" not in Path(path).name]


def clear_viewer_caches() -> None:
    discover_subject_dirs.clear()
    list_nifti_files.clear()
    load_nifti.clear()


def main() -> None:
    st.set_page_config(page_title="HFF Scan Viewer", layout="wide")
    st.markdown(
        """
        <style>
        div[data-testid="stHorizontalBlock"]:has(input[aria-label="Dataset"]) {
            position: sticky;
            top: 0.5rem;
            z-index: 1000;
            padding: 0.65rem 0.75rem 0.5rem;
            background: var(--background-color, #0e1117);
            border: 1px solid rgba(250, 250, 250, 0.2);
            border-radius: 0.5rem;
            box-shadow: 0 0.35rem 1rem rgba(0, 0, 0, 0.3);
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    header_slot = st.empty()

    dataset_root = st.session_state.get("dataset_root", str(DEFAULT_DATA_ROOT))

    subject_dirs = discover_subject_dirs(dataset_root)
    if not subject_dirs:
        st.warning("No BraTS subject folders were found. Check the dataset root and click Rescan dataset.")
        st.stop()

    root_path = Path(dataset_root).expanduser()
    subject_labels = {f"{Path(subject_dir).relative_to(root_path)}": subject_dir for subject_dir in subject_dirs}

    available_subject_labels = list(subject_labels.keys())
    subject_label = st.session_state.get("subject_label", available_subject_labels[0])
    if subject_label not in subject_labels:
        subject_label = available_subject_labels[0]
    subject_dir = subject_labels[subject_label]

    file_paths = list_nifti_files(subject_dir)
    if not file_paths:
        st.warning("The selected subject does not contain any NIfTI files.")
        st.stop()

    modality_paths = scan_modalities(file_paths)
    if not modality_paths:
        st.warning("The selected subject does not contain scan volumes.")
        st.stop()

    default_files = subject_default_files(modality_paths)
    display_options = {f"{parse_display_name(path, subject_dir)}": path for path in modality_paths}

    default_labels = [label for label, path in display_options.items() if path in default_files]
    selected_labels = [
        label
        for label in st.session_state.get("scan_labels", default_labels)
        if label in display_options
    ]

    with header_slot.container(horizontal=True, vertical_alignment="bottom", gap="small", border=True):
        st.text_input("Dataset", key="dataset_root", value=dataset_root)
        st.selectbox(
            "File name",
            options=available_subject_labels,
            key="subject_label",
            index=available_subject_labels.index(subject_label),
        )
        st.multiselect(
            "Scans",
            options=list(display_options.keys()),
            key="scan_labels",
            default=default_labels,
        )
        st.segmented_control(
            "View",
            options=["High-resolution slices", "3D volume"],
            key="view_mode",
            default="High-resolution slices",
        )
        st.slider("Slice", min_value=0, max_value=100, key="slice_position", value=50)
        st.button("Rescan", icon=":material/refresh:", on_click=clear_viewer_caches)

    selected_files = [display_options[label] for label in selected_labels]
    if not selected_files:
        st.info("Pick at least one volume to compare.")
        st.stop()

    loaded_volumes = {path: load_nifti(path) for path in selected_files}
    seg_path = next((path for path in file_paths if "_seg" in Path(path).name), None)
    seg_volume = load_nifti(seg_path) if seg_path else None
    view_mode = st.session_state.get("view_mode", "High-resolution slices")
    slice_fraction = st.session_state.get("slice_position", 50) / 100

    # A 3-over-2 layout keeps all five panels in the main view while giving
    # each 3D brain enough width and height to be inspected comfortably.
    panel_items = [("scan", file_path) for file_path in selected_files]
    if seg_volume is not None:
        panel_items.append(("mask", seg_volume))

    panel_rows = [st.columns(3, gap="medium")]
    if len(panel_items) > 3:
        panel_rows.append(st.columns(3, gap="medium"))

    for index, (kind, value) in enumerate(panel_items):
        column = panel_rows[index // 3][index % 3]
        with column.container(border=True, height=470):
            if kind == "scan":
                file_path = value
                volume = loaded_volumes[file_path]
                figure = (
                    build_slice_figure(volume, parse_display_name(file_path, subject_dir), slice_fraction)
                    if view_mode == "High-resolution slices"
                    else build_volume_figure(volume, parse_display_name(file_path, subject_dir))
                )
                st.plotly_chart(figure, width="stretch", config={"displaylogo": False, "responsive": True})
            else:
                if view_mode == "High-resolution slices":
                    mask_figure = build_slice_figure(
                        loaded_volumes[selected_files[0]],
                        "Expected mask (red overlay)",
                        slice_fraction,
                        mask=value,
                    )
                    st.plotly_chart(mask_figure, width="stretch", config={"displaylogo": False, "responsive": True})
                else:
                    surface_figure = build_surface_figure(value, "Expected mask")
                    if surface_figure is None:
                        st.info("No foreground mask voxels found.")
                    else:
                        st.plotly_chart(surface_figure, width="stretch", config={"displaylogo": False, "responsive": True})

    with st.expander("Subject details", expanded=False):
        st.write("Selected scan files:")
        for file_path in selected_files:
            volume = loaded_volumes[file_path]
            st.write(
                f"- {Path(file_path).name} | shape={tuple(volume.shape)} | min={float(np.nanmin(volume)):.2f} | max={float(np.nanmax(volume)):.2f}"
            )
        if seg_volume is not None:
            st.write(f"Segmentation shape={tuple(seg_volume.shape)} | labels={sorted(int(x) for x in np.unique(seg_volume) if x > 0)}")


if __name__ == "__main__":
    main()
