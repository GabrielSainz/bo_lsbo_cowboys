"""Diagnostics logging helpers for the toy Bayesian-optimization runs.

This module is deliberately instrumentation-only: it stores snapshots that the
experiment scripts already compute, derives post-hoc metrics from those
snapshots, and writes JSON that can be plotted without rerunning BO.
"""

import csv
import json
import math
import os
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

from toy_common import ensure_dir, turtle_path


MetricFn = Callable[[np.ndarray], float]
DecodeFn = Callable[[np.ndarray], np.ndarray]


METRIC_KEYS = [
    "bo_iteration",
    "best_so_far_objective",
    "selected_objective",
    "nearest_previous_selected_latent_distance",
    "mean_top_k_objective",
    "top_k_latent_diversity",
    "normalized_top_k_latent_diversity",
    "adjusted_top_k_objective",
]


def latent_box_diversity_normalizer(z_box: Optional[float], latent_dim: int) -> Optional[float]:
    """Return the diagonal length of [-z_box, z_box]^d for diversity scaling."""
    if z_box is None:
        return None
    z_box = float(z_box)
    if not np.isfinite(z_box) or z_box <= 0.0:
        return None
    return float(2.0 * z_box * math.sqrt(max(1, int(latent_dim))))


def finite_or_none(value: Any) -> Optional[float]:
    """Convert numeric values to JSON-safe floats, mapping nan/inf to null."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def json_safe(value: Any) -> Any:
    """Recursively convert numpy arrays/scalars into JSON-serializable values."""
    if value is None:
        return None
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, (np.floating, float)):
        return finite_or_none(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    return value


def as_2d_array(values: Any, width: Optional[int] = None) -> Optional[np.ndarray]:
    if values is None:
        return None
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        if width is None:
            return np.zeros((0, 0), dtype=np.float64)
        return np.zeros((0, int(width)), dtype=np.float64)
    if arr.ndim == 1:
        if width is None:
            arr = arr.reshape(1, -1)
        else:
            arr = arr.reshape(-1, int(width))
    return arr.astype(np.float64, copy=False)


def safe_mean(values: Iterable[Any]) -> Optional[float]:
    arr = np.asarray([v for v in values if v is not None], dtype=np.float64)
    if arr.size == 0:
        return None
    mask = np.isfinite(arr)
    if not np.any(mask):
        return None
    return float(np.mean(arr[mask]))


def average_pairwise_distance(points: Optional[np.ndarray]) -> Optional[float]:
    if points is None:
        return None
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 2:
        return None
    diffs = points[:, None, :] - points[None, :, :]
    d = np.sqrt(np.sum(diffs * diffs, axis=-1))
    iu = np.triu_indices(points.shape[0], k=1)
    vals = d[iu]
    if vals.size == 0:
        return None
    return float(np.mean(vals))


def decoded_path_list(sequence: Optional[np.ndarray], step_size: Optional[float]) -> Any:
    if sequence is None or step_size is None:
        return None
    return json_safe(turtle_path(np.asarray(sequence, dtype=np.float64), float(step_size)))


def select_top_k_indices(acquisition_values: Optional[np.ndarray], n: int, k: int) -> np.ndarray:
    if n <= 0 or k <= 0:
        return np.zeros((0,), dtype=int)
    take = min(int(k), int(n))
    if acquisition_values is None:
        return np.arange(take, dtype=int)
    acq = np.asarray(acquisition_values, dtype=np.float64).reshape(-1)
    if acq.shape[0] != n:
        return np.arange(take, dtype=int)
    score = np.where(np.isfinite(acq), acq, -np.inf)
    if np.all(~np.isfinite(score)):
        return np.arange(take, dtype=int)
    return np.argsort(score)[-take:][::-1].astype(int)


def subsample_proposals(
    proposal_latents: Optional[np.ndarray],
    proposal_acquisition_values: Optional[np.ndarray],
    max_points: int,
) -> tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """Keep JSON sizes bounded while preserving top-acquisition candidates."""
    if proposal_latents is None:
        return None, None
    latents = np.asarray(proposal_latents, dtype=np.float64)
    if latents.ndim != 2:
        return None, None

    n = latents.shape[0]
    if max_points <= 0 or n <= max_points:
        acq = None
        if proposal_acquisition_values is not None:
            acq = np.asarray(proposal_acquisition_values, dtype=np.float64).reshape(-1)
            if acq.shape[0] != n:
                acq = None
        return latents, acq

    keep = set()
    acq = None
    if proposal_acquisition_values is not None:
        acq = np.asarray(proposal_acquisition_values, dtype=np.float64).reshape(-1)
        if acq.shape[0] == n:
            top_n = max(1, max_points // 2)
            score = np.where(np.isfinite(acq), acq, -np.inf)
            for idx in np.argsort(score)[-top_n:]:
                keep.add(int(idx))
        else:
            acq = None

    remaining = max_points - len(keep)
    if remaining > 0:
        for idx in np.linspace(0, n - 1, num=remaining, dtype=int):
            keep.add(int(idx))

    idx = np.array(sorted(keep), dtype=int)
    if idx.shape[0] > max_points:
        idx = idx[:max_points]
    return latents[idx], acq[idx] if acq is not None else None


class ToyDiagnosticsLogger:
    """Collect per-run diagnostics and export a JSON artifact."""

    def __init__(
        self,
        method: str,
        seed: int,
        problem: str,
        config: Dict[str, Any],
        results_root: str = "results/toy_diagnostics",
        top_k: int = 10,
        max_proposals: int = 2000,
        diversity_normalizer: Optional[float] = None,
        step_size: Optional[float] = None,
        target_path: Optional[np.ndarray] = None,
        decode_latent_fn: Optional[DecodeFn] = None,
        objective_fn: Optional[MetricFn] = None,
    ):
        self.method = str(method)
        self.seed = int(seed)
        self.problem = str(problem)
        self.results_root = os.fspath(results_root)
        self.top_k = int(top_k)
        self.max_proposals = int(max_proposals)
        self.diversity_normalizer = finite_or_none(diversity_normalizer)
        self.step_size = finite_or_none(step_size)
        self.decode_latent_fn = decode_latent_fn
        self.objective_fn = objective_fn

        self.run_dir = os.path.join(
            self.results_root,
            "raw",
            self.method,
            self.problem,
            f"seed_{self.seed}",
        )
        ensure_dir(self.run_dir)

        self.config = dict(config or {})
        self.config.update(
            {
                "diagnostics_top_k": self.top_k,
                "diagnostics_max_proposals": self.max_proposals,
                "top_k_diversity_normalization": (
                    "average pairwise top-k latent distance divided by the "
                    "configured latent-box diagonal, clipped to [0, 1]"
                ),
                "adjusted_top_k_objective_definition": (
                    "(mean_top_k_objective - run_objective_floor) * "
                    "normalized_top_k_latent_diversity; the shift avoids sign "
                    "inversions because the toy oracle is usually non-positive"
                ),
            }
        )

        self.target_path = json_safe(target_path)
        self.objective_background: Optional[Dict[str, Any]] = None
        self.initial_observations: Dict[str, Any] = {}
        self.iterations: List[Dict[str, Any]] = []
        self.metric_series = {key: [] for key in METRIC_KEYS}
        self._selected_latents: List[np.ndarray] = []

    def set_objective_background(self, z_box: float, grid_res: int = 60) -> None:
        """Store a diagnostic-only oracle landscape over a 2D latent grid.

        This evaluates f(decode(z)) on a plotting grid after the run setup. The
        values are never used by the BO algorithms or selection policy.
        """
        if self.decode_latent_fn is None or self.objective_fn is None:
            return
        z_box = float(z_box)
        grid_res = int(grid_res)
        if not np.isfinite(z_box) or z_box <= 0.0 or grid_res <= 1:
            return

        z1 = np.linspace(-z_box, z_box, grid_res)
        z2 = np.linspace(-z_box, z_box, grid_res)
        gx, gy = np.meshgrid(z1, z2)
        z_grid = np.stack([gx.ravel(), gy.ravel()], axis=1)
        y = []
        for z in z_grid:
            x = self.decode_latent_fn(z)
            y.append(float(self.objective_fn(x)))
        self.objective_background = {
            "z1": json_safe(z1),
            "z2": json_safe(z2),
            "objective_grid": json_safe(np.asarray(y, dtype=np.float64).reshape(grid_res, grid_res)),
            "label": "oracle f(decode(z))",
        }

    def set_initial_observations(
        self,
        latents: Any,
        objectives: Any,
        sequences: Optional[Any] = None,
    ) -> None:
        z = as_2d_array(latents)
        y = np.asarray(objectives, dtype=np.float64).reshape(-1)
        seq = None if sequences is None else np.asarray(sequences, dtype=np.float64)

        paths = None
        if seq is not None and self.step_size is not None:
            paths = [decoded_path_list(x, self.step_size) for x in seq]

        self.initial_observations = {
            "latents": json_safe(z),
            "objectives": json_safe(y),
            "decoded_paths": paths,
        }

    def _decode_top_sequences(
        self,
        top_latents: Optional[np.ndarray],
        proposal_sequences: Optional[Any],
        top_indices: np.ndarray,
    ) -> Optional[np.ndarray]:
        if top_latents is None or top_latents.shape[0] == 0:
            return None
        if proposal_sequences is not None:
            seq = np.asarray(proposal_sequences, dtype=np.float64)
            if seq.ndim >= 2 and seq.shape[0] >= np.max(top_indices, initial=0) + 1:
                return seq[top_indices]
        if self.decode_latent_fn is None:
            return None
        decoded = [np.asarray(self.decode_latent_fn(z), dtype=np.float64) for z in top_latents]
        return np.asarray(decoded, dtype=np.float64)

    def _score_sequences(self, sequences: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if sequences is None or self.objective_fn is None:
            return None
        return np.asarray([float(self.objective_fn(x)) for x in sequences], dtype=np.float64)

    def record_iteration(
        self,
        iteration: int,
        selected_latent: Any,
        selected_objective: float,
        best_so_far_objective: float,
        incumbent_best_latent: Optional[Any] = None,
        incumbent_best_objective: Optional[float] = None,
        selected_sequence: Optional[Any] = None,
        incumbent_best_sequence: Optional[Any] = None,
        proposal_latents: Optional[Any] = None,
        proposal_acquisition_values: Optional[Any] = None,
        proposal_sequences: Optional[Any] = None,
        top_k_candidate_objectives: Optional[Any] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        selected_z = np.asarray(selected_latent, dtype=np.float64).reshape(-1)
        proposal_z = as_2d_array(proposal_latents, width=selected_z.shape[0])
        proposal_acq = None
        if proposal_acquisition_values is not None:
            proposal_acq = np.asarray(proposal_acquisition_values, dtype=np.float64).reshape(-1)
            if proposal_z is None or proposal_acq.shape[0] != proposal_z.shape[0]:
                proposal_acq = None

        nearest = None
        if self._selected_latents:
            prev = np.vstack(self._selected_latents)
            nearest = float(np.min(np.linalg.norm(prev - selected_z.reshape(1, -1), axis=1)))

        top_indices = np.zeros((0,), dtype=int)
        top_latents = None
        top_acq = None
        top_sequences = None
        top_objectives = None
        if proposal_z is not None and proposal_z.shape[0] > 0:
            top_indices = select_top_k_indices(proposal_acq, proposal_z.shape[0], self.top_k)
            top_latents = proposal_z[top_indices]
            top_acq = proposal_acq[top_indices] if proposal_acq is not None else None
            top_sequences = self._decode_top_sequences(top_latents, proposal_sequences, top_indices)
            if top_k_candidate_objectives is not None:
                top_objectives = np.asarray(top_k_candidate_objectives, dtype=np.float64).reshape(-1)
            else:
                top_objectives = self._score_sequences(top_sequences)

        mean_top_k_objective = None
        if top_objectives is not None and top_objectives.size > 0:
            mean_top_k_objective = safe_mean(top_objectives)

        diversity = average_pairwise_distance(top_latents)
        normalized_diversity = None
        if diversity is not None and self.diversity_normalizer is not None and self.diversity_normalizer > 0:
            normalized_diversity = float(np.clip(diversity / self.diversity_normalizer, 0.0, 1.0))

        stored_proposals, stored_acq = subsample_proposals(proposal_z, proposal_acq, self.max_proposals)

        row = {
            "iteration": int(iteration),
            "selected_latent": json_safe(selected_z),
            "selected_objective": finite_or_none(selected_objective),
            "best_so_far_objective": finite_or_none(best_so_far_objective),
            "nearest_previous_selected_latent_distance": finite_or_none(nearest),
            "incumbent_best_latent": json_safe(incumbent_best_latent),
            "incumbent_best_objective": finite_or_none(incumbent_best_objective),
            "proposal_latents": json_safe(stored_proposals),
            "proposal_acquisition_values": json_safe(stored_acq),
            "top_k_candidate_latents": json_safe(top_latents),
            "top_k_candidate_acquisition_values": json_safe(top_acq),
            "top_k_candidate_objectives": json_safe(top_objectives),
            "mean_top_k_objective": finite_or_none(mean_top_k_objective),
            "top_k_latent_diversity": finite_or_none(diversity),
            "normalized_top_k_latent_diversity": finite_or_none(normalized_diversity),
            "adjusted_top_k_objective": None,
            "selected_decoded_path": decoded_path_list(
                None if selected_sequence is None else np.asarray(selected_sequence, dtype=np.float64),
                self.step_size,
            ),
            "incumbent_best_decoded_path": decoded_path_list(
                None if incumbent_best_sequence is None else np.asarray(incumbent_best_sequence, dtype=np.float64),
                self.step_size,
            ),
        }
        if extra:
            row["extra"] = json_safe(extra)

        self.iterations.append(row)
        self._selected_latents.append(selected_z.copy())

        self.metric_series["bo_iteration"].append(int(iteration))
        self.metric_series["best_so_far_objective"].append(row["best_so_far_objective"])
        self.metric_series["selected_objective"].append(row["selected_objective"])
        self.metric_series["nearest_previous_selected_latent_distance"].append(
            row["nearest_previous_selected_latent_distance"]
        )
        self.metric_series["mean_top_k_objective"].append(row["mean_top_k_objective"])
        self.metric_series["top_k_latent_diversity"].append(row["top_k_latent_diversity"])
        self.metric_series["normalized_top_k_latent_diversity"].append(
            row["normalized_top_k_latent_diversity"]
        )
        self.metric_series["adjusted_top_k_objective"].append(None)

    def _objective_floor(self) -> float:
        vals: List[float] = []
        for row in self.iterations:
            y = finite_or_none(row.get("selected_objective"))
            if y is not None:
                vals.append(y)
            for top_y in row.get("top_k_candidate_objectives") or []:
                y_top = finite_or_none(top_y)
                if y_top is not None:
                    vals.append(y_top)
        return float(min(vals)) if vals else 0.0

    def finalize(self, extra_summary: Optional[Dict[str, Any]] = None) -> str:
        objective_floor = self._objective_floor()
        adjusted_values = []
        for row in self.iterations:
            mean_y = finite_or_none(row.get("mean_top_k_objective"))
            div = finite_or_none(row.get("normalized_top_k_latent_diversity"))
            adjusted = None
            if mean_y is not None and div is not None:
                adjusted = float((mean_y - objective_floor) * div)
            row["adjusted_top_k_objective"] = finite_or_none(adjusted)
            adjusted_values.append(row["adjusted_top_k_objective"])
        self.metric_series["adjusted_top_k_objective"] = adjusted_values

        payload = {
            "method": self.method,
            "seed": self.seed,
            "problem": self.problem,
            "config": json_safe(self.config),
            "objective_shift_baseline": finite_or_none(objective_floor),
            "target_path": self.target_path,
            "objective_background": self.objective_background,
            "initial_observations": self.initial_observations,
            "metric_series": json_safe(self.metric_series),
            "iterations": self.iterations,
        }
        if extra_summary:
            payload["summary"] = json_safe(extra_summary)

        path = os.path.join(self.run_dir, "metrics.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        self._write_metric_csv(os.path.join(self.run_dir, "metric_series.csv"))
        return path

    def _write_metric_csv(self, path: str) -> None:
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(METRIC_KEYS)
            n = len(self.metric_series["bo_iteration"])
            for i in range(n):
                writer.writerow([self.metric_series[key][i] for key in METRIC_KEYS])
