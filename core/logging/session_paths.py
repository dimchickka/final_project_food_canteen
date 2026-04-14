"""Path contract for artifacts produced by one recognition run."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from config.logging_settings import LOG_ROOT_DIR


class SessionPaths:
    """Defines stable file/directory locations for a single logging session."""

    def __init__(self, log_root: str | Path | None = None, run_name: str | None = None) -> None:
        self._log_root = Path(log_root) if log_root is not None else Path(LOG_ROOT_DIR)
        self._run_name = run_name
        self._run_dir: Path | None = None

    @property
    def run_dir(self) -> Path:
        """Absolute path to the run folder; created lazily by ensure/create."""
        if self._run_dir is None:
            self._run_dir = self._resolve_unique_run_dir()
        return self._run_dir

    @property
    def detections_dir(self) -> Path:
        return self.run_dir / "detections"

    @property
    def cups_dir(self) -> Path:
        return self.detections_dir / "cups"

    @property
    def plates_dir(self) -> Path:
        return self.detections_dir / "plates"

    @property
    def bowls_dir(self) -> Path:
        return self.detections_dir / "bowls"

    @property
    def sauces_dir(self) -> Path:
        return self.detections_dir / "sauces"

    @property
    def meats_dir(self) -> Path:
        return self.detections_dir / "meats"

    @property
    def garnish_dir(self) -> Path:
        return self.detections_dir / "garnish"

    @property
    def source_image_path(self) -> Path:
        return self.run_dir / "source.jpg"

    @property
    def annotated_result_path(self) -> Path:
        return self.run_dir / "annotated_result.jpg"

    @property
    def receipt_json_path(self) -> Path:
        return self.run_dir / "receipt.json"

    @property
    def summary_txt_path(self) -> Path:
        return self.run_dir / "summary.txt"

    @property
    def timings_json_path(self) -> Path:
        return self.run_dir / "timings.json"

    @property
    def pipeline_trace_json_path(self) -> Path:
        return self.run_dir / "pipeline_trace.json"

    @property
    def qwen_validation_txt_path(self) -> Path:
        return self.run_dir / "qwen_validation.txt"

    def _resolve_unique_run_dir(self) -> Path:
        """Builds deterministic run name; appends numeric suffix on collisions."""
        base_name = self._run_name or f"run_{datetime.now().strftime('%Y_%m_%d_%H_%M_%S')}"
        candidate = self._log_root / base_name
        if not candidate.exists():
            return candidate

        suffix = 1
        while True:
            suffixed = self._log_root / f"{base_name}_{suffix:02d}"
            if not suffixed.exists():
                return suffixed
            suffix += 1

    def create(self) -> Path:
        """Creates run directory and standard subdirectories."""
        return self.ensure()

    def ensure(self) -> Path:
        """Ensures that run directory contract exists on disk."""
        self._log_root.mkdir(parents=True, exist_ok=True)
        run_dir = self.run_dir
        run_dir.mkdir(parents=True, exist_ok=True)

        for directory in (
            self.detections_dir,
            self.cups_dir,
            self.plates_dir,
            self.bowls_dir,
            self.sauces_dir,
            self.meats_dir,
            self.garnish_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

        return run_dir
