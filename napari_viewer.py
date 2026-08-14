"""Interactive Napari viewer for input and frequency-decomposition analysis."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import getpass
import os
import posixpath
from pathlib import Path
import re
import shlex
import stat
import tempfile
from typing import Callable

import napari
import numpy as np
import SimpleITK as sitk
import torch
from qtpy.QtCore import QObject, QSettings, QThread, Qt, Signal
from qtpy.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QComboBox,
    QCompleter,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QInputDialog,
    QPushButton,
    QProgressDialog,
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


VIEWER_SETTINGS = QSettings("HFF-Net", "NapariViewer")


def remembered_local_directory(fallback: Path) -> str:
    value = str(VIEWER_SETTINGS.value("lastLocalDirectory", "") or "")
    remembered = Path(value).expanduser()
    return str(remembered if remembered.is_dir() else fallback)


def remember_local_directory(selected_directory: Path) -> None:
    VIEWER_SETTINGS.setValue("lastLocalDirectory", str(selected_directory.parent))
    VIEWER_SETTINGS.sync()


def remembered_remote_directory(profile_name: str) -> str | None:
    if not profile_name:
        return None
    value = VIEWER_SETTINGS.value(f"lastRemoteDirectory/{profile_name}")
    return str(value) if value else None


def remember_remote_directory(profile_name: str, selected_directory: str) -> None:
    if not profile_name:
        return
    normalized = selected_directory.rstrip("/") or "/"
    parent = posixpath.dirname(normalized) or "/"
    VIEWER_SETTINGS.setValue(f"lastRemoteDirectory/{profile_name}", parent)
    VIEWER_SETTINGS.sync()


def strip_nifti_suffix(path: Path) -> str:
    if path.name.lower().endswith(".nii.gz"):
        return path.name[:-7]
    return path.stem


def is_nifti_name(name: str) -> bool:
    """Return whether a remote filename is a NIfTI volume."""
    lowered = name.lower()
    return lowered.endswith(".nii") or lowered.endswith(".nii.gz")


def is_nifti_file(path: Path) -> bool:
    return is_nifti_name(path.name)


def is_frequency_file(path: Path) -> bool:
    stem = strip_nifti_suffix(path).lower()
    return stem.rsplit("_", 1)[-1] in {"l", "h", "h1", "h2", "h3", "h4"}


@dataclass(frozen=True)
class SSHProfile:
    """One named host entry from the OpenSSH config used by VS Code."""

    name: str
    settings: dict[str, object]


def vscode_settings_paths() -> tuple[Path, ...]:
    """Return likely VS Code user-settings locations for the current platform."""
    home = Path.home()
    candidates = [
        PROJECT_ROOT / ".vscode/settings.json",
        home / "Library/Application Support/Code/User/settings.json",
        home / ".config/Code/User/settings.json",
    ]
    app_data = os.environ.get("APPDATA")
    if app_data:
        candidates.append(Path(app_data) / "Code/User/settings.json")
    return tuple(dict.fromkeys(candidates))


def vscode_ssh_config_path() -> Path:
    """Resolve VS Code Remote-SSH's config path, falling back to OpenSSH's default."""
    setting_pattern = re.compile(
        r'["\']remote\.SSH\.configFile["\']\s*:\s*["\']([^"\']+)["\']'
    )
    for settings_path in vscode_settings_paths():
        try:
            settings_text = settings_path.read_text(encoding="utf-8")
        except OSError:
            continue
        match = setting_pattern.search(settings_text)
        if match:
            return Path(os.path.expandvars(match.group(1))).expanduser().resolve()
    return Path.home() / ".ssh/config"


def load_ssh_profiles(config_path: Path) -> list[SSHProfile]:
    """Read named, non-wildcard hosts from an OpenSSH config file."""
    try:
        import paramiko
    except ImportError as error:
        raise RuntimeError(
            "SSH support requires Paramiko. Install the project requirements "
            "or run `pip install paramiko`."
        ) from error

    if not config_path.is_file():
        raise FileNotFoundError(f"SSH config file not found: {config_path}")

    ssh_config = paramiko.SSHConfig.from_path(str(config_path))
    profiles: list[SSHProfile] = []
    seen: set[str] = set()
    for name in ssh_config.get_hostnames():
        if name in seen or any(character in name for character in "*!?"):
            continue
        seen.add(name)
        profiles.append(SSHProfile(name, dict(ssh_config.lookup(name))))
    return profiles


class SSHSession:
    """Small SSH-backed bridge for loading one remote subject locally.

    Napari and the existing model helpers intentionally operate on local
    ``Path`` objects.  The session therefore downloads only the selected
    subject's NIfTI files into a temporary directory; no remote data is
    written or mounted.
    """

    def __init__(self, client: object, sftp: object | None, profile_name: str = "") -> None:
        self.client = client
        self.sftp = sftp
        self.profile_name = profile_name
        self.cache_directory = Path(tempfile.mkdtemp(prefix="hff_napari_ssh_"))
        self._closed = False

    @classmethod
    def connect(
        cls,
        *,
        host: str,
        port: int,
        username: str,
        password: str,
        identity_files: list[str],
        key_passphrase: str,
        proxy_command: str,
        profile_name: str = "",
    ) -> "SSHSession":
        try:
            import paramiko
        except ImportError as error:
            raise RuntimeError(
                "SSH support requires Paramiko. Install the project requirements "
                "or run `pip install paramiko`."
            ) from error

        if not host.strip():
            raise ValueError("SSH host is required.")

        client = paramiko.SSHClient()
        client.load_system_host_keys()
        # The session is interactive and does not persist credentials.  Keep
        # the usual SSH first-use behaviour for hosts not yet in known_hosts.
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        connect_kwargs: dict[str, object] = {
            "hostname": host.strip(),
            "port": port,
            "timeout": 15,
            "banner_timeout": 15,
            "auth_timeout": 15,
            "allow_agent": True,
            "look_for_keys": True,
        }
        if username.strip():
            connect_kwargs["username"] = username.strip()
        if password:
            connect_kwargs["password"] = password
        expanded_identity_files = [
            os.path.expandvars(identity_file).strip()
            for identity_file in identity_files
            if identity_file.strip() and identity_file.lower() != "none"
        ]
        if expanded_identity_files:
            connect_kwargs["key_filename"] = [
                str(Path(identity_file).expanduser())
                for identity_file in expanded_identity_files
            ]
        if key_passphrase:
            connect_kwargs["passphrase"] = key_passphrase

        proxy = None
        if proxy_command.strip() and proxy_command.strip().lower() != "none":
            proxy_command = proxy_command.strip()
            proxy_command = proxy_command.replace("%%", "%")
            proxy_command = proxy_command.replace("%h", host.strip())
            proxy_command = proxy_command.replace("%p", str(port))
            proxy_command = proxy_command.replace("%r", username.strip())
            proxy = paramiko.ProxyCommand(proxy_command)
            connect_kwargs["sock"] = proxy

        try:
            try:
                client.connect(**connect_kwargs)
            except paramiko.SSHException as authentication_error:
                # Paramiko's regular SSHClient.connect path does not attempt
                # keyboard-interactive authentication when no key or
                # password was supplied. OpenSSH/VS Code commonly uses this
                # method, especially with a ProxyCommand, so retry it with
                # the selected profile's optional password.
                can_retry_interactive = isinstance(
                    authentication_error, paramiko.AuthenticationException
                ) or "No authentication methods available" in str(authentication_error)
                if not can_retry_interactive:
                    raise
                transport = client.get_transport()
                if transport is None:
                    raise
                auth_username = username.strip() or getpass.getuser()

                if proxy_command.strip() and not password:
                    # NetBird SSH profiles can authenticate the connection at
                    # the proxy and intentionally use OpenSSH's ``none``
                    # method. Do not turn that valid passwordless flow into a
                    # password prompt.
                    try:
                        transport.auth_none(auth_username)
                    except Exception:
                        raise authentication_error
                else:
                    def interactive_handler(_title: str, _instructions: str, prompts: list[tuple[str, bool]]) -> tuple[str, ...]:
                        return tuple(password for _prompt, _echo in prompts)

                    try:
                        transport.auth_interactive(auth_username, interactive_handler)
                    except Exception:
                        raise authentication_error
            try:
                sftp = client.open_sftp()
            except Exception:
                # Some NetBird SSH endpoints support shell commands but close
                # SFTP subsystem requests. Keep the authenticated transport
                # and use the exec-based filesystem fallback below.
                sftp = None
        except Exception:
            if proxy is not None:
                proxy.close()
            client.close()
            raise
        return cls(client, sftp, profile_name)

    def execute(self, command: str) -> bytes:
        """Run a read-only remote command and return stdout bytes."""
        _stdin, stdout, stderr = self.client.exec_command(command)
        output = stdout.read()
        error = stderr.read()
        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            message = error.decode("utf-8", errors="replace").strip()
            raise OSError(message or f"Remote command failed with status {exit_status}.")
        return output

    def normalize(self, remote_path: str) -> str:
        if self.sftp is not None:
            return str(self.sftp.normalize(remote_path or "."))
        quoted_path = shlex.quote(remote_path or ".")
        return self.execute(f"cd {quoted_path} && pwd -P").decode("utf-8").strip()

    def directory_entries(self, remote_path: str) -> list[tuple[str, bool]]:
        """Return ``(name, is_directory)`` entries for a remote directory."""
        normalized = self.normalize(remote_path)
        if self.sftp is None:
            output = self.execute(
                "find {path} -mindepth 1 -maxdepth 1 "
                "-printf '%y\\t%f\\n'".format(path=shlex.quote(normalized))
            )
            entries = []
            for line in output.decode("utf-8", errors="replace").splitlines():
                kind, separator, name = line.partition("\t")
                if separator and name:
                    entries.append((name, kind == "d"))
            return sorted(entries, key=lambda item: (not item[1], item[0].lower()))

        entries = []
        for entry in self.sftp.listdir_attr(normalized):
            mode = getattr(entry, "st_mode", 0)
            entries.append((str(entry.filename), stat.S_ISDIR(mode)))
        return sorted(entries, key=lambda item: (not item[1], item[0].lower()))

    def is_subject_directory(self, remote_path: str) -> bool:
        try:
            entries = self.directory_entries(remote_path)
        except OSError:
            return False
        files = [name for name, is_directory in entries if not is_directory]
        return any(is_nifti_name(name) and "_seg" in name.lower() for name in files)

    def download_subject(
        self,
        remote_path: str,
        progress_callback: Callable[[int, int, str], None] | None = None,
    ) -> Path:
        """Download the selected record's NIfTI files and return its local path."""
        normalized = self.normalize(remote_path)
        entries = self.directory_entries(normalized)
        nifti_files = [name for name, is_directory in entries if not is_directory and is_nifti_name(name)]
        if not any("_seg" in name.lower() for name in nifti_files):
            raise ValueError("That remote folder is not a BraTS record: no _seg.nii file was found.")
        if not nifti_files:
            raise ValueError("That remote folder does not contain NIfTI files.")

        subject_name = Path(normalized).name or "remote_subject"
        local_directory = Path(tempfile.mkdtemp(prefix=f"{subject_name}_", dir=self.cache_directory))
        total_files = len(nifti_files)
        for index, filename in enumerate(nifti_files, start=1):
            remote_file = posixpath.join(normalized, filename)
            if self.sftp is not None:
                self.sftp.get(remote_file, str(local_directory / filename))
            else:
                file_bytes = self.execute(f"cat -- {shlex.quote(remote_file)}")
                (local_directory / filename).write_bytes(file_bytes)
            if progress_callback is not None:
                progress_callback(index, total_files, filename)
        return local_directory

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self.sftp is not None:
            self.sftp.close()
        self.client.close()
        # Temporary cache cleanup is deliberately best-effort; a viewer crash
        # must not prevent the application from exiting.
        try:
            import shutil

            shutil.rmtree(self.cache_directory, ignore_errors=True)
        except OSError:
            pass


class RemoteSubjectDownloadWorker(QObject):
    """Download a remote subject without blocking Napari's Qt event loop."""

    finished = Signal(object, object, object)
    failed = Signal(str)
    progress = Signal(int)
    message = Signal(str)

    def __init__(self, session: SSHSession, remote_path: str) -> None:
        super().__init__()
        self.session = session
        self.remote_path = remote_path

    def run(self) -> None:
        try:
            local_path = self.session.download_subject(
                self.remote_path,
                progress_callback=self.report_progress,
            )
            self.message.emit("Loading downloaded MRI volumes…")
            volumes, mask = load_subject(local_path)
        except Exception as error:
            self.failed.emit(str(error))
            return
        self.finished.emit(local_path, volumes, mask)

    def report_progress(self, index: int, total: int, filename: str) -> None:
        percentage = int(index * 100 / max(total, 1))
        self.progress.emit(percentage)
        self.message.emit(f"Downloading {index}/{total}: {filename}")


def start_remote_subject_download(
    session: SSHSession,
    remote_path: str,
    parent: QWidget,
    on_finished: Callable[[Path, dict[str, np.ndarray], np.ndarray | None], None],
    on_failed: Callable[[str], None],
) -> None:
    """Start a visible background download and return immediately."""
    progress = QProgressDialog("Preparing remote folder…", None, 0, 100, parent)
    progress.setWindowTitle("Loading remote BraTS folder")
    progress.setMinimumDuration(0)
    progress.setAutoClose(False)
    progress.setAutoReset(False)
    progress.setCancelButton(None)
    progress.setWindowModality(Qt.WindowModality.WindowModal)
    progress.show()

    thread = QThread(parent)
    worker = RemoteSubjectDownloadWorker(session, remote_path)
    worker.moveToThread(thread)
    # Keep strong references until the worker has emitted its completion
    # signal; otherwise Python can collect the wrapper while the Qt thread is
    # still running.
    parent._remote_download_thread = thread  # type: ignore[attr-defined]
    parent._remote_download_worker = worker  # type: ignore[attr-defined]
    parent._remote_download_progress = progress  # type: ignore[attr-defined]

    worker.progress.connect(progress.setValue)
    worker.message.connect(progress.setLabelText)
    thread.started.connect(worker.run)

    result: dict[str, object] = {}

    def record_finished(
        local_path: Path,
        volumes: dict[str, np.ndarray],
        mask: np.ndarray | None,
    ) -> None:
        result["finished"] = (local_path, volumes, mask)

    def record_failed(message: str) -> None:
        result["failed"] = message

    worker.finished.connect(record_finished)
    worker.failed.connect(record_failed)
    worker.finished.connect(thread.quit)
    worker.failed.connect(thread.quit)
    worker.finished.connect(worker.deleteLater)
    worker.failed.connect(worker.deleteLater)

    def finish_after_thread_stops() -> None:
        progress.close()
        for attribute, value in (
            ("_remote_download_thread", thread),
            ("_remote_download_worker", worker),
            ("_remote_download_progress", progress),
        ):
            if getattr(parent, attribute, None) is value:
                setattr(parent, attribute, None)
        if "finished" in result:
            on_finished(*result["finished"])  # type: ignore[arg-type]
        elif "failed" in result:
            on_failed(str(result["failed"]))

    thread.finished.connect(finish_after_thread_stops)
    thread.finished.connect(thread.deleteLater)
    thread.start()


class SSHConnectionDialog(QDialog):
    """Select a named host from the OpenSSH config used by VS Code."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Connect to BraTS folder over SSH")
        self.setMinimumWidth(430)
        self.config_path = vscode_ssh_config_path()
        self.profiles = load_ssh_profiles(self.config_path)
        self.profile_index = {profile.name: profile for profile in self.profiles}

        form = QFormLayout(self)
        config_label = QLabel(str(self.config_path))
        config_label.setWordWrap(True)
        config_label.setStyleSheet("color: palette(mid); font-size: 11px;")
        form.addRow("VS Code SSH config", config_label)

        self.profile_combo = QComboBox()
        self.profile_combo.addItems(self.profile_index)
        self.profile_combo.currentTextChanged.connect(self.update_profile_details)
        form.addRow("SSH profile", self.profile_combo)

        self.profile_details = QLabel()
        self.profile_details.setWordWrap(True)
        self.profile_details.setStyleSheet("color: palette(mid); font-size: 11px;")
        form.addRow("Resolved settings", self.profile_details)

        self.password_edit = QLineEdit()
        self.password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_edit.setPlaceholderText("optional; not stored in SSH config")
        form.addRow("Password", self.password_edit)

        self.key_passphrase_edit = QLineEdit()
        self.key_passphrase_edit.setEchoMode(QLineEdit.EchoMode.Password)
        self.key_passphrase_edit.setPlaceholderText("optional")
        form.addRow("Key passphrase", self.key_passphrase_edit)

        form.addRow(
            QLabel(
                "Credentials are used for this viewer session only. The selected "
                "subject is cached locally while Napari is open. Host, user, port, "
                "identity files, and proxy settings come from the selected profile."
            )
        )
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

        self.connect_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        if self.connect_button is not None:
            self.connect_button.setEnabled(bool(self.profiles))
        self.update_profile_details(self.profile_combo.currentText())
        if not self.profiles:
            self.profile_details.setText(
                "No named SSH profiles were found. Add a Host entry to the config "
                "and reopen this dialog."
            )

    def update_profile_details(self, profile_name: str) -> None:
        profile = self.profile_index.get(profile_name)
        if profile is None:
            return
        settings = profile.settings
        host = str(settings.get("hostname") or profile.name)
        username = str(settings.get("user") or "(local SSH username)")
        port = str(settings.get("port") or 22)
        identity_files = settings.get("identityfile", [])
        if isinstance(identity_files, str):
            identity_files = [identity_files]
        identity_summary = ", ".join(str(path) for path in identity_files) or "SSH agent / default keys"
        proxy = str(settings.get("proxycommand") or settings.get("proxyjump") or "none")
        self.profile_details.setText(
            f"Host: {host}\nUser: {username}\nPort: {port}\n"
            f"Identity: {identity_summary}\nProxy: {proxy}"
        )

    def settings(self) -> dict[str, object]:
        profile = self.profile_index[self.profile_combo.currentText()]
        profile_settings = profile.settings
        identity_files = profile_settings.get("identityfile", [])
        if isinstance(identity_files, str):
            identity_files = [identity_files]
        return {
            "host": str(profile_settings.get("hostname") or profile.name),
            "port": int(profile_settings.get("port") or 22),
            "username": str(profile_settings.get("user") or ""),
            "password": self.password_edit.text(),
            "identity_files": [str(path) for path in identity_files],
            "key_passphrase": self.key_passphrase_edit.text(),
            "proxy_command": str(profile_settings.get("proxycommand") or ""),
            "profile_name": profile.name,
        }


def open_ssh_session(parent: QWidget | None = None) -> SSHSession | None:
    try:
        dialog = SSHConnectionDialog(parent)
    except Exception as error:
        QMessageBox.critical(parent, "SSH profile loading failed", str(error))
        return None
    while dialog.exec() == QDialog.DialogCode.Accepted:
        settings = dialog.settings()
        try:
            return SSHSession.connect(**settings)
        except Exception as error:
            authentication_error = "No authentication methods available" in str(error)
            has_proxy = bool(str(settings.get("proxy_command") or "").strip())
            if authentication_error and not has_proxy and not dialog.password_edit.text():
                password, accepted = QInputDialog.getText(
                    parent,
                    "SSH authentication required",
                    "This profile has no available SSH key or agent identity.\n"
                    "Enter the SSH password, or cancel to return to the profile selector:",
                    QLineEdit.EchoMode.Password,
                )
                if accepted and password:
                    dialog.password_edit.setText(password)
                    continue
            QMessageBox.critical(parent, "SSH connection failed", str(error))
    return None


class RemoteDirectoryWorker(QObject):
    """Run one remote directory operation outside the Qt UI thread."""

    directory_loaded = Signal(str, object)
    validation_finished = Signal(str, bool)
    failed = Signal(str)

    def __init__(self, session: SSHSession, requested_path: str, validate: bool) -> None:
        super().__init__()
        self.session = session
        self.requested_path = requested_path
        self.validate = validate

    def run(self) -> None:
        try:
            normalized = self.session.normalize(self.requested_path)
            entries = self.session.directory_entries(normalized)
            if self.validate:
                valid = any(
                    not is_directory
                    and is_nifti_name(name)
                    and "_seg" in name.lower()
                    for name, is_directory in entries
                )
                self.validation_finished.emit(normalized, valid)
            else:
                self.directory_loaded.emit(normalized, entries)
        except Exception as error:
            self.failed.emit(str(error))


class RemoteDirectoryDialog(QDialog):
    """Browse remote directories without blocking the Napari UI."""

    def __init__(
        self,
        session: SSHSession,
        parent: QWidget | None = None,
        initial_path: str | None = None,
    ) -> None:
        super().__init__(parent)
        self.session = session
        self.selected_remote_path: str | None = None
        self.fallback_path = "."
        self.pending_validation: tuple[str, bool] | None = None
        self.operation_error: str | None = None
        self.operation_requested_path = "."
        self.operation_allow_fallback = True
        self.operation_thread: QThread | None = None
        self.operation_worker: RemoteDirectoryWorker | None = None
        self.setWindowTitle("Select remote BraTS record folder")
        self.setMinimumSize(560, 430)

        layout = QVBoxLayout(self)
        path_row = QHBoxLayout()
        self.path_edit = QLineEdit(initial_path or ".")
        self.path_edit.returnPressed.connect(self.refresh)
        path_row.addWidget(self.path_edit)
        self.go_button = QPushButton("Go")
        self.go_button.clicked.connect(self.refresh)
        path_row.addWidget(self.go_button)
        layout.addLayout(path_row)

        self.entries = QListWidget()
        self.entries.itemDoubleClicked.connect(self.open_entry)
        layout.addWidget(self.entries)
        self.status = QLabel("Loading remote directory…")
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        button_row = QHBoxLayout()
        self.up_button = QPushButton("Up")
        self.up_button.clicked.connect(self.go_up)
        button_row.addWidget(self.up_button)
        button_row.addStretch(1)
        self.select_button = QPushButton("Select current folder")
        self.select_button.clicked.connect(self.select_current)
        button_row.addWidget(self.select_button)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        button_row.addWidget(self.cancel_button)
        layout.addLayout(button_row)
        self.refresh()

    def set_controls_enabled(self, enabled: bool) -> None:
        self.path_edit.setEnabled(enabled)
        self.go_button.setEnabled(enabled)
        self.entries.setEnabled(enabled)
        self.up_button.setEnabled(enabled)
        self.select_button.setEnabled(enabled)
        self.cancel_button.setEnabled(enabled)

    def start_operation(self, requested_path: str, *, validate: bool = False, allow_fallback: bool = True) -> None:
        if self.operation_thread is not None:
            return
        self.operation_requested_path = requested_path
        self.operation_allow_fallback = allow_fallback
        self.operation_error = None
        self.pending_validation = None
        self.set_controls_enabled(False)
        self.status.setText("Checking remote folder…" if validate else "Loading remote directory…")

        thread = QThread(self)
        worker = RemoteDirectoryWorker(self.session, requested_path, validate)
        worker.moveToThread(thread)
        self.operation_thread = thread
        self.operation_worker = worker
        worker.directory_loaded.connect(self.directory_loaded)
        worker.validation_finished.connect(self.validation_finished)
        worker.failed.connect(self.operation_failed)
        thread.started.connect(worker.run)
        worker.directory_loaded.connect(thread.quit)
        worker.validation_finished.connect(thread.quit)
        worker.failed.connect(thread.quit)
        worker.directory_loaded.connect(worker.deleteLater)
        worker.validation_finished.connect(worker.deleteLater)
        worker.failed.connect(worker.deleteLater)
        thread.finished.connect(self.operation_finished)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    def refresh(self) -> None:
        requested = self.path_edit.text().strip() or "."
        self.start_operation(requested)

    def directory_loaded(self, normalized: str, entries: list[tuple[str, bool]]) -> None:
        self.fallback_path = normalized
        self.path_edit.setText(normalized)
        self.entries.clear()
        for name, is_directory in entries:
            if is_directory:
                self.entries.addItem(f"📁 {name}")
        if not entries:
            self.status.setText("This directory is empty.")
        elif not any(is_directory for _, is_directory in entries):
            self.status.setText("No subdirectories. If this is a record folder, select the current folder.")
        else:
            self.status.setText("Double-click a directory to open it, or select the current folder when it is a BraTS record.")

    def validation_finished(self, normalized: str, valid: bool) -> None:
        self.pending_validation = (normalized, valid)

    def operation_failed(self, message: str) -> None:
        self.operation_error = message

    def operation_finished(self) -> None:
        thread = self.operation_thread
        self.operation_thread = None
        self.operation_worker = None
        self.set_controls_enabled(True)

        if self.operation_error is not None:
            if self.operation_allow_fallback and self.operation_requested_path != self.fallback_path:
                self.path_edit.setText(self.fallback_path)
                self.status.setText(f"Remembered folder was unavailable; reopened {self.fallback_path}.")
                self.start_operation(self.fallback_path, allow_fallback=False)
                return
            self.status.setText(f"Could not read remote directory: {self.operation_error}")
            return

        if self.pending_validation is not None:
            normalized, valid = self.pending_validation
            self.pending_validation = None
            if not valid:
                QMessageBox.warning(
                    self,
                    "Not a BraTS record",
                    "Select a remote folder containing its _seg.nii or _seg.nii.gz file.",
                )
                return
            self.selected_remote_path = normalized
            remember_remote_directory(self.session.profile_name, normalized)
            self.accept()

    def open_entry(self, item: object) -> None:
        name = str(item.text()).removeprefix("📁 ")
        self.path_edit.setText(posixpath.join(self.path_edit.text(), name))
        self.refresh()

    def go_up(self) -> None:
        current = self.path_edit.text().rstrip("/") or "/"
        self.path_edit.setText(posixpath.dirname(current) or "/")
        self.refresh()

    def select_current(self) -> None:
        requested = self.path_edit.text().strip() or "."
        self.start_operation(requested, validate=True, allow_fallback=False)

    def reject(self) -> None:
        if self.operation_thread is not None:
            return
        super().reject()


def select_remote_directory(session: SSHSession, parent: QWidget | None = None) -> str | None:
    dialog = RemoteDirectoryDialog(
        session,
        parent,
        remembered_remote_directory(session.profile_name),
    )
    if dialog.exec() != QDialog.DialogCode.Accepted or dialog.selected_remote_path is None:
        return None
    return dialog.selected_remote_path


class InitialSubjectDialog(QDialog):
    """Choose the data source before the viewer's normal subject controls open."""

    def __init__(self, dataset_root: Path, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.dataset_root = dataset_root
        self.selected_subject: Path | None = None
        self.ssh_session: SSHSession | None = None
        self.remote_path: str | None = None
        self.preloaded_volumes: dict[str, np.ndarray] | None = None
        self.preloaded_mask: np.ndarray | None = None
        self.setWindowTitle("Choose BraTS data source")
        self.setMinimumWidth(450)

        layout = QVBoxLayout(self)
        title = QLabel("Choose a local folder, or connect over SSH before selecting a remote folder.")
        title.setWordWrap(True)
        layout.addWidget(title)

        self.local_button = QPushButton("Select local record folder…")
        self.local_button.clicked.connect(self.select_local_folder)
        layout.addWidget(self.local_button)

        self.ssh_button = QPushButton("Connect over SSH and select remote folder…")
        self.ssh_button.clicked.connect(self.select_remote_folder)
        layout.addWidget(self.ssh_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self.reject)
        layout.addWidget(self.cancel_button)
        self.remote_download_active = False

    def select_local_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "Select BraTS record folder",
            remembered_local_directory(self.dataset_root),
            QFileDialog.Option.ShowDirsOnly,
        )
        if not selected:
            return
        selected_path = Path(selected).expanduser().resolve()
        if not selected_path.is_dir() or subject_files(selected_path)[1] is None:
            QMessageBox.warning(
                self,
                "Not a BraTS record",
                "Select a folder containing its _seg.nii or _seg.nii.gz file.",
            )
            return
        if self.ssh_session is not None:
            self.ssh_session.close()
            self.ssh_session = None
        remember_local_directory(selected_path)
        self.selected_subject = selected_path
        self.accept()

    def select_remote_folder(self) -> None:
        if self.remote_download_active:
            return
        new_session = open_ssh_session(self)
        if new_session is None:
            return
        if self.ssh_session is not None:
            self.ssh_session.close()
        self.ssh_session = new_session
        selected_remote_path = select_remote_directory(self.ssh_session, self)
        if selected_remote_path is None:
            return
        self.remote_path = selected_remote_path
        self.remote_download_active = True
        self.local_button.setEnabled(False)
        self.ssh_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        start_remote_subject_download(
            self.ssh_session,
            selected_remote_path,
            self,
            self.remote_download_finished,
            self.remote_download_failed,
        )

    def remote_download_finished(
        self,
        local_path: Path,
        volumes: dict[str, np.ndarray],
        mask: np.ndarray | None,
    ) -> None:
        self.remote_download_active = False
        self.selected_subject = local_path
        self.preloaded_volumes = volumes
        self.preloaded_mask = mask
        self.accept()

    def remote_download_failed(self, message: str) -> None:
        self.remote_download_active = False
        self.remote_path = None
        self.local_button.setEnabled(True)
        self.ssh_button.setEnabled(True)
        self.cancel_button.setEnabled(True)
        QMessageBox.critical(self, "Remote folder download failed", message)

    def reject(self) -> None:
        if self.remote_download_active:
            return
        if self.ssh_session is not None and self.selected_subject is None:
            self.ssh_session.close()
            self.ssh_session = None
        super().reject()


def select_initial_subject(
    dataset_root: Path,
) -> tuple[Path, SSHSession | None, str | None, dict[str, np.ndarray] | None, np.ndarray | None] | None:
    dialog = InitialSubjectDialog(dataset_root)
    if dialog.exec() != QDialog.DialogCode.Accepted or dialog.selected_subject is None:
        return None
    return (
        dialog.selected_subject,
        dialog.ssh_session,
        dialog.remote_path,
        dialog.preloaded_volumes,
        dialog.preloaded_mask,
    )


def modality_sort_key(path: Path) -> tuple[int, str]:
    stem = strip_nifti_suffix(path).lower()
    for index, modality in enumerate(MODALITY_PRIORITY):
        if stem.endswith(f"_{modality}"):
            return index, stem
    return 99, stem


def subject_files(subject_dir: Path) -> tuple[list[Path], Path | None]:
    files = [path for path in subject_dir.iterdir() if path.is_file() and is_nifti_file(path)]
    scans = sorted(
        [path for path in files if "_seg" not in path.name.lower() and not is_frequency_file(path)],
        key=modality_sort_key,
    )
    segmentation = next((path for path in files if "_seg" in path.name.lower()), None)
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
        ssh_session: SSHSession | None = None,
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
        self.ssh_session = ssh_session
        self.current_scan_index: dict[str, Path] = {}
        self.current_mask: np.ndarray | None = None
        self.current_subject_dir: Path | None = None
        self.current_output_mask: np.ndarray | None = None
        self.checkpoint_index: dict[str, Path] = {}
        self.remote_download_active = False
        self.pending_remote_path: str | None = None

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

        self.ssh_button = QPushButton("Connect over SSH…")
        self.ssh_button.clicked.connect(self.connect_ssh)
        layout.addWidget(self.ssh_button)

        self.ssh_status = QLabel()
        self.ssh_status.setWordWrap(True)
        self.ssh_status.setStyleSheet("color: palette(mid); font-size: 11px;")
        layout.addWidget(self.ssh_status)

        self.select_record_button = QPushButton("Select local record folder…")
        self.select_record_button.clicked.connect(self.select_record_folder)
        layout.addWidget(self.select_record_button)

        self.select_remote_button = QPushButton("Select remote record folder…")
        self.select_remote_button.clicked.connect(self.select_remote_record_folder)
        self.select_remote_button.setEnabled(self.ssh_session is not None)
        layout.addWidget(self.select_remote_button)

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
        self.update_ssh_status()

    def update_ssh_status(self) -> None:
        if self.ssh_session is None:
            self.ssh_status.setText("SSH is optional. Local folder selection is ready.")
            self.select_remote_button.setEnabled(False)
            self.ssh_button.setText("Connect over SSH…")
        else:
            self.ssh_status.setText("SSH connected. The selected remote subject is cached locally for this session.")
            self.select_remote_button.setEnabled(True)
            self.ssh_button.setText("Reconnect over SSH…")

    def connect_ssh(self) -> None:
        new_session = open_ssh_session(self)
        if new_session is None:
            return
        if self.ssh_session is not None:
            self.ssh_session.close()
        self.ssh_session = new_session
        self.update_ssh_status()
        self.select_remote_record_folder()

    def close_resources(self) -> None:
        if self.ssh_session is not None:
            self.ssh_session.close()
            self.ssh_session = None

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
            remembered_local_directory(self.dataset_root),
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

        remember_local_directory(selected_path)
        self.on_subject_changed(selected_path)

    def select_remote_record_folder(self) -> None:
        """Browse remote directories and download one record in the background."""
        if self.remote_download_active:
            return
        if self.ssh_session is None:
            self.connect_ssh()
            return
        self.ssh_status.setText("Choose a remote record folder…")
        remote_path = select_remote_directory(self.ssh_session, self)
        if remote_path is None:
            self.update_ssh_status()
            return
        self.remote_download_active = True
        self.pending_remote_path = remote_path
        self.select_record_button.setEnabled(False)
        self.select_remote_button.setEnabled(False)
        self.ssh_button.setEnabled(False)
        self.ssh_status.setText("Preparing remote folder download…")
        start_remote_subject_download(
            self.ssh_session,
            remote_path,
            self,
            self.remote_download_finished,
            self.remote_download_failed,
        )

    def remote_download_finished(
        self,
        local_path: Path,
        volumes: dict[str, np.ndarray],
        mask: np.ndarray | None,
    ) -> None:
        self.remote_download_active = False
        remote_path = self.pending_remote_path or "(remote folder)"
        self.pending_remote_path = None
        self.on_subject_changed(local_path, (volumes, mask))
        self.selected_record.setText(f"Selected remote: {remote_path}\nCached at: {local_path}")
        self.ssh_status.setText("SSH connected. The selected remote subject is cached locally for this session.")
        self.select_record_button.setEnabled(True)
        self.select_remote_button.setEnabled(True)
        self.ssh_button.setEnabled(True)

    def remote_download_failed(self, message: str) -> None:
        self.remote_download_active = False
        self.pending_remote_path = None
        self.update_ssh_status()
        self.select_record_button.setEnabled(True)
        self.select_remote_button.setEnabled(True)
        self.ssh_button.setEnabled(True)
        QMessageBox.critical(self, "Remote folder download failed", message)

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

    def on_subject_changed(
        self,
        subject_dir: Path,
        preloaded: tuple[dict[str, np.ndarray], np.ndarray | None] | None = None,
    ) -> None:
        subject_dir = subject_dir.expanduser().resolve()
        if subject_files(subject_dir)[1] is None:
            return

        self.selected_record.setText(f"Selected: {subject_dir}")
        volumes, mask = preloaded if preloaded is not None else load_subject(subject_dir)
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
    ssh_session: SSHSession | None = None,
    remote_path: str | None = None,
    initial_data: tuple[dict[str, np.ndarray], np.ndarray | None] | None = None,
) -> None:
    viewer.layers.clear()

    volumes, mask = initial_data if initial_data is not None else load_subject(initial_subject)

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
        ssh_session,
    )
    if remote_path is not None:
        selector.selected_record.setText(
            f"Selected remote: {remote_path}\nCached at: {initial_subject}"
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

    from qtpy.QtWidgets import QApplication

    application = QApplication.instance()
    if application is not None:
        application.aboutToQuit.connect(selector.close_resources)


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
    ssh_session: SSHSession | None = None
    remote_path: str | None = None
    initial_data: tuple[dict[str, np.ndarray], np.ndarray | None] | None = None

    if args.subject:
        requested = Path(args.subject).expanduser()
        initial_subject = requested if requested.is_absolute() else dataset_root / requested
    else:
        selected = select_initial_subject(dataset_root)
        if selected is None:
            viewer.close()
            raise SystemExit("No BraTS record selected.")
        initial_subject, ssh_session, remote_path, preloaded_volumes, preloaded_mask = selected
        if preloaded_volumes is not None:
            initial_data = (preloaded_volumes, preloaded_mask)

    if not initial_subject.is_dir() or subject_files(initial_subject)[1] is None:
        if ssh_session is not None:
            ssh_session.close()
        viewer.close()
        raise SystemExit(
            f"Not a BraTS record folder: {initial_subject}. "
            "Expected a folder containing _seg.nii or _seg.nii.gz."
        )

    add_subject_layers(
        viewer,
        dataset_root,
        initial_subject,
        results_root,
        ssh_session,
        remote_path,
        initial_data,
    )
    napari.run()


if __name__ == "__main__":
    main()
