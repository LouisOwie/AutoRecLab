# /// script
# requires-python = ">=3.11"
# ///
"""Audit generated AutoRecLab result artifacts without re-running generated code.

Run from the AutoRecLab_eval root directory:

    uv run verify_outputs.py

The script performs static and numerical checks on generated CSV/JSON/code/log
artifacts. It does not download data, call an LLM, or execute generated
experiment code outside the recorded AutoRecLab run.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "eval_out"
EXPECTED_SEED_COUNT = 5
TOLERANCE = 1e-9

METRIC_ALIASES = {
    "ndcg10": "NDCG@10",
    "ndcgat10": "NDCG@10",
    "precision10": "Precision@10",
    "precisionat10": "Precision@10",
}

SEED_COLUMNS = (
    "splitseed",
    "seed",
    "randomseed",
    "datasplitseed",
    "datasplitrandomseed",
    "trainingseed",
)

ALGORITHM_COLUMNS = ("algorithm", "scorer", "model")

CODE_FEATURES = {
    "dataset": r"MovieLens100K|DataSet\.MovieLens100K",
    "rating_filter": r"RatingFilter",
    "implicit_conversion": r"MakeImplicit",
    "five_core": r"CorePruning",
    "user_holdout": r"UserHoldout",
    "random_seed": r"set_random_state|random_state|random_seed",
    "test_size_20_percent": r"test_size\s*=\s*0\.2(?:0)?\b",
    "ndcg_metric": r"NDCG",
    "precision_metric": r"Precision",
    "als_algorithm": r"ImplicitMFScorer",
    "popularity_algorithm": r"PopScorer",
}

CODE_SEED_PATTERN = re.compile(
    r"^\s*(?:SEEDS|SPLIT_SEEDS|seeds)\s*=\s*\[([^\]]*)\]",
    re.IGNORECASE | re.MULTILINE,
)


def normalize(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def canonical_metric(value: Any) -> str:
    normalized = normalize(value)
    return METRIC_ALIASES.get(normalized, "")


def canonical_algorithm(value: Any) -> str:
    raw = str(value or "").strip()
    normalized = normalize(raw)
    if not normalized:
        return ""
    if "implicitmf" in normalized or normalized == "als" or "als" in normalized:
        return "ALS"
    if normalized == "pop" or "popscorer" in normalized or "popularity" in normalized:
        return "Pop"
    return raw


def parse_number(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "na", "n/a"}:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def text_is_nonfinite(value: Any) -> bool:
    text = str(value or "").strip()
    if not text:
        return False
    try:
        return not math.isfinite(float(text))
    except ValueError:
        return False


def parse_seed(value: Any) -> int | str | None:
    number = parse_number(value)
    if number is not None and number.is_integer():
        return int(number)
    text = str(value or "").strip()
    return text or None


def seed_key(value: int | str | None) -> str:
    return "" if value is None else str(value)


def relative(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        raw_headers = reader.fieldnames or []
        headers = [normalize(header) for header in raw_headers]
        rows: list[dict[str, str]] = []
        for raw_row in reader:
            row: dict[str, str] = {}
            for raw_key, raw_value in raw_row.items():
                if raw_key is None:
                    continue
                row[normalize(raw_key)] = (raw_value or "").strip()
            if any(value for value in row.values()):
                rows.append(row)
    return headers, rows


def read_json(path: Path) -> Any | None:
    try:
        return json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return None


def metadata_json_paths(generated_dir: Path) -> list[Path]:
    """Return small metadata files, excluding per-seed prediction checkpoints."""
    paths: list[Path] = []
    for path in generated_dir.glob("*.json"):
        stem = normalize(path.stem)
        if any(token in stem for token in ("config", "count", "metadata", "seed", "summary")):
            paths.append(path)
    return sorted(paths)


def walk_dicts(value: Any):
    if isinstance(value, dict):
        yield value
        for child in value.values():
            yield from walk_dicts(child)
    elif isinstance(value, list):
        for child in value:
            yield from walk_dicts(child)


def find_seed_lists(generated_dir: Path) -> list[list[int | str]]:
    found: list[list[int | str]] = []
    for path in metadata_json_paths(generated_dir):
        payload = read_json(path)
        if payload is None:
            continue
        for mapping in walk_dicts(payload):
            for key, value in mapping.items():
                if normalize(key) != "seeds" or not isinstance(value, list):
                    continue
                seeds = [parse_seed(item) for item in value]
                seeds = [item for item in seeds if item is not None]
                if seeds:
                    found.append(seeds)
    return found


def find_dataset_counts(generated_dir: Path) -> tuple[dict[str, int] | None, Path | None]:
    candidates: list[tuple[Path, dict[str, int]]] = []
    for path in metadata_json_paths(generated_dir):
        payload = read_json(path)
        if payload is None:
            continue
        for mapping in walk_dicts(payload):
            normalized = {normalize(key): value for key, value in mapping.items()}
            if not {"users", "items", "interactions"}.issubset(normalized):
                continue
            counts: dict[str, int] = {}
            valid = True
            for key in ("users", "items", "interactions"):
                number = parse_number(normalized[key])
                if number is None or not number.is_integer() or number < 0:
                    valid = False
                    break
                counts[key] = int(number)
            if valid:
                candidates.append((path, counts))
    if not candidates:
        return None, None
    return candidates[-1][1], candidates[-1][0]


def infer_algorithm_from_path(path: Path) -> str:
    text = normalize(path.stem)
    if "pop" in text:
        return "Pop"
    if "als" in text or "implicitmf" in text:
        return "ALS"
    return ""


def find_column(headers: list[str], candidates: tuple[str, ...]) -> str | None:
    for candidate in candidates:
        if candidate in headers:
            return candidate
    return None


def generated_directory(path: Path) -> Path | None:
    for parent in path.parents:
        if parent.name == "generated":
            return parent
    return None


def node_id_for_generated(generated_dir: Path) -> str:
    return generated_dir.parent.name


def load_positions(run_dir: Path) -> dict[str, int]:
    positions: dict[str, int] = {}
    for path in sorted((run_dir / "statistics").glob("*.txt")):
        text = path.read_text(encoding="utf-8", errors="replace")
        id_match = re.search(r"^ID:\s*(.+)$", text, re.MULTILINE)
        position_match = re.search(r"^Position:\s*(\d+)$", text, re.MULTILINE)
        if id_match and position_match:
            positions[id_match.group(1).strip()] = int(position_match.group(1))
    return positions


@dataclass
class ResultArtifact:
    path: Path
    generated_dir: Path
    node_id: str
    position: int | None
    headers: list[str]
    rows: list[dict[str, str]]
    seed_column: str
    seed_source: str
    algorithm_column: str | None
    metric_columns: dict[str, str]
    long_format: bool
    metric_rows: list[dict[str, float | None]] = field(default_factory=list)
    seed_values: list[int | str | None] = field(default_factory=list)
    algorithms: set[str] = field(default_factory=set)
    metrics: set[str] = field(default_factory=set)

    @property
    def label(self) -> str:
        return relative(self.path)

    @property
    def seed_set(self) -> set[str]:
        return {seed_key(value) for value in self.seed_values if value is not None}

    @property
    def protocol_values(self) -> dict[str, tuple[str, ...]]:
        aliases = {
            "train": ("traininteractions", "train"),
            "validation": ("validationinteractions", "validation", "val"),
            "test": ("testinteractions", "test"),
            "train_users": ("trainusers",),
            "test_users": ("testusers",),
        }
        result: dict[str, tuple[str, ...]] = {}
        for name, possible_columns in aliases.items():
            column = find_column(self.headers, possible_columns)
            if column:
                values = sorted({row.get(column, "") for row in self.rows})
                result[name] = tuple(values)
        return result


def load_result_artifact(path: Path, positions: dict[str, int]) -> ResultArtifact | None:
    generated_dir = generated_directory(path)
    if generated_dir is None:
        return None
    headers, rows = read_csv(path)
    seed_column = find_column(headers, SEED_COLUMNS)
    if seed_column is None:
        return None

    metric_columns: dict[str, str] = {}
    for header in headers:
        metric = canonical_metric(header)
        if metric:
            metric_columns[metric] = header

    metric_column = "metric" if "metric" in headers else None
    value_column = find_column(headers, ("value", "metricvalue", "score"))
    long_format = metric_column is not None and value_column is not None
    if not metric_columns and not long_format:
        return None

    algorithm_column = find_column(headers, ALGORITHM_COLUMNS)
    node_id = node_id_for_generated(generated_dir)
    position = positions.get(node_id)
    metric_rows: list[dict[str, float | None]] = []
    seed_values: list[int | str | None] = []
    algorithms: set[str] = set()
    metrics: set[str] = set()
    inferred_algorithm = infer_algorithm_from_path(path)

    for row in rows:
        seed_values.append(parse_seed(row.get(seed_column, "")))
        raw_algorithm = row.get(algorithm_column, "") if algorithm_column else inferred_algorithm
        algorithm = canonical_algorithm(raw_algorithm)
        if algorithm:
            algorithms.add(algorithm)

        values: dict[str, float | None] = {}
        for metric, column in metric_columns.items():
            values[metric] = parse_number(row.get(column, ""))
            metrics.add(metric)
        if long_format:
            metric = canonical_metric(row.get(metric_column or "", ""))
            if metric:
                values[metric] = parse_number(row.get(value_column or "", ""))
                metrics.add(metric)
        metric_rows.append(values)

    if not metrics:
        return None
    seed_source = seed_column
    return ResultArtifact(
        path=path,
        generated_dir=generated_dir,
        node_id=node_id,
        position=position,
        headers=headers,
        rows=rows,
        seed_column=seed_column,
        seed_source=seed_source,
        algorithm_column=algorithm_column,
        metric_columns=metric_columns,
        long_format=long_format,
        metric_rows=metric_rows,
        seed_values=seed_values,
        algorithms=algorithms,
        metrics=metrics,
    )


def discover_artifacts(run_dir: Path, positions: dict[str, int]) -> list[ResultArtifact]:
    artifacts: list[ResultArtifact] = []
    checkpoint = run_dir / "checkpoint"
    if not checkpoint.exists():
        return artifacts
    for path in sorted(checkpoint.rglob("*.csv")):
        artifact = load_result_artifact(path, positions)
        if artifact is not None:
            artifacts.append(artifact)
    return artifacts


def is_summary_csv(path: Path) -> bool:
    try:
        headers, _ = read_csv(path)
    except OSError:
        return False
    return (
        "summary" in normalize(path.stem)
        or ("mean" in headers and ("std" in headers or "samplestd" in headers))
    )


def parse_summary(path: Path) -> list[dict[str, Any]]:
    headers, rows = read_csv(path)
    if "mean" not in headers or not ({"std", "samplestd"} & set(headers)):
        return []
    inferred_algorithm = infer_algorithm_from_path(path)
    records: list[dict[str, Any]] = []
    for row in rows:
        metric = canonical_metric(row.get("metric", ""))
        if not metric:
            continue
        raw_algorithm = row.get("algorithm", "") or inferred_algorithm
        record: dict[str, Any] = {
            "algorithm": canonical_algorithm(raw_algorithm),
            "metric": metric,
        }
        record["mean"] = parse_number(row.get("mean", ""))
        record["std"] = parse_number(row.get("std", row.get("samplestd", "")))
        for field_name in ("min", "max", "range"):
            record[field_name] = parse_number(row.get(field_name, ""))
        for field_name in ("bestseed", "worstseed"):
            record[field_name] = parse_seed(row.get(field_name, ""))
        records.append(record)
    return records


def group_values(artifact: ResultArtifact) -> dict[tuple[str, str], list[tuple[int | str | None, float]]]:
    groups: dict[tuple[str, str], list[tuple[int | str | None, float]]] = {}
    inferred_algorithm = infer_algorithm_from_path(artifact.path)
    for row, metric_row, seed in zip(artifact.rows, artifact.metric_rows, artifact.seed_values):
        raw_algorithm = row.get(artifact.algorithm_column, "") if artifact.algorithm_column else inferred_algorithm
        algorithm = canonical_algorithm(raw_algorithm)
        for metric, value in metric_row.items():
            if value is None:
                continue
            groups.setdefault((algorithm, metric), []).append((seed, value))
    return groups


def calculated_stats(values: list[tuple[int | str | None, float]]) -> dict[str, Any]:
    numbers = [value for _, value in values]
    best = max(values, key=lambda item: item[1])[0] if values else None
    worst = min(values, key=lambda item: item[1])[0] if values else None
    return {
        "mean": statistics.fmean(numbers) if numbers else None,
        "std": statistics.stdev(numbers) if len(numbers) > 1 else 0.0,
        "min": min(numbers) if numbers else None,
        "max": max(numbers) if numbers else None,
        "range": (max(numbers) - min(numbers)) if numbers else None,
        "bestseed": best,
        "worstseed": worst,
    }


def close_enough(left: Any, right: Any) -> bool:
    if left is None or right is None:
        return left is right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return math.isclose(float(left), float(right), rel_tol=TOLERANCE, abs_tol=TOLERANCE)
    return str(left) == str(right)


def summary_checks(artifact: ResultArtifact) -> list[dict[str, str]]:
    summary_paths = [
        path
        for path in sorted(artifact.generated_dir.rglob("*.csv"))
        if path != artifact.path and is_summary_csv(path)
    ]
    if not summary_paths:
        return [{
            "status": "WARN",
            "details": "No summary CSV was found beside this result artifact.",
        }]

    groups = group_values(artifact)
    reports: list[dict[str, str]] = []
    matched_any = False
    for summary_path in summary_paths:
        records = parse_summary(summary_path)
        if not records:
            continue
        mismatches: list[str] = []
        matched = 0
        for record in records:
            candidates = [
                (key, values)
                for key, values in groups.items()
                if key[1] == record["metric"]
                and (not record["algorithm"] or key[0] == record["algorithm"])
            ]
            if len(candidates) != 1:
                continue
            matched += 1
            matched_any = True
            calculated = calculated_stats(candidates[0][1])
            for field_name in ("mean", "std", "min", "max", "range", "bestseed", "worstseed"):
                expected = record.get(field_name)
                actual = calculated.get(field_name)
                if expected is not None and not close_enough(expected, actual):
                    mismatches.append(
                        f"{record['algorithm'] or candidates[0][0][0]} {record['metric']} {field_name}: "
                        f"reported={expected!r}, recomputed={actual!r}"
                    )
        if matched:
            reports.append({
                "status": "FAIL" if mismatches else "PASS",
                "details": (
                    f"Compared {matched} summary row(s) with {relative(summary_path)}. "
                    + ("; ".join(mismatches) if mismatches else "All reported aggregate values match.")
                ),
            })
    if not matched_any:
        return [{
            "status": "WARN",
            "details": "Summary CSVs were present, but no unambiguous metric/algorithm pair could be matched.",
        }]
    return reports


def artifact_checks(artifact: ResultArtifact) -> list[tuple[str, str, str]]:
    checks: list[tuple[str, str, str]] = []
    metrics = sorted(artifact.metrics)
    groups: dict[tuple[str, str], set[str]] = {}
    duplicate_keys: list[str] = []
    missing_values = 0
    non_finite_values = 0
    out_of_range: list[str] = []
    seen: set[tuple[str, str, str]] = set()
    inferred_algorithm = infer_algorithm_from_path(artifact.path)

    for row_index, (row, metric_row, seed) in enumerate(
        zip(artifact.rows, artifact.metric_rows, artifact.seed_values), start=2
    ):
        raw_algorithm = row.get(artifact.algorithm_column, "") if artifact.algorithm_column else inferred_algorithm
        algorithm = canonical_algorithm(raw_algorithm)
        seed_text = seed_key(seed)
        group_key = (seed_text, algorithm)
        row_metrics = set(metric_row) if artifact.long_format else set(metrics)
        value_column = find_column(artifact.headers, ("value", "metricvalue", "score"))
        for metric in row_metrics:
            groups.setdefault(group_key, set())
            if metric in metric_row:
                groups[group_key].add(metric)
            key = (seed_text, algorithm, metric)
            if key in seen:
                duplicate_keys.append(f"row {row_index}: {key}")
            seen.add(key)
            raw_value = metric_row.get(metric)
            if raw_value is None:
                missing_values += 1
                raw_metric_value = (
                    row.get(value_column, "")
                    if artifact.long_format and value_column
                    else row.get(artifact.metric_columns.get(metric, ""), "")
                )
                if text_is_nonfinite(raw_metric_value):
                    non_finite_values += 1
                continue
            if not math.isfinite(raw_value):
                non_finite_values += 1
            if raw_value < 0 or raw_value > 1:
                out_of_range.append(f"{metric}={raw_value}")

    missing_groups = [
        f"{key} missing {sorted(set(metrics) - present)}"
        for key, present in groups.items()
        if set(metrics) - present
    ]
    missing_seed_rows = sum(seed is None for seed in artifact.seed_values)
    if missing_seed_rows:
        checks.append(("missing_seed_values", "FAIL", f"{missing_seed_rows} result row(s) have no usable seed."))
    else:
        checks.append(("missing_seed_values", "PASS", "All result rows have a usable seed."))

    if missing_values or missing_groups:
        checks.append((
            "missing_metric_values",
            "FAIL",
            f"Missing metric cells={missing_values}; incomplete seed/algorithm groups={missing_groups!r}.",
        ))
    else:
        checks.append(("missing_metric_values", "PASS", "All seed/algorithm groups contain every metric."))

    if duplicate_keys:
        checks.append(("duplicate_seed_algorithm_metric", "FAIL", "; ".join(duplicate_keys)))
    else:
        checks.append(("duplicate_seed_algorithm_metric", "PASS", "No duplicate seed/algorithm/metric keys."))

    if non_finite_values:
        checks.append(("finite_metrics", "FAIL", f"Found {non_finite_values} non-finite metric value(s)."))
    else:
        checks.append(("finite_metrics", "PASS", "All parsed metric values are finite."))

    if out_of_range:
        checks.append(("metric_range_0_to_1", "FAIL", "; ".join(out_of_range[:10])))
    else:
        checks.append(("metric_range_0_to_1", "PASS", "All ranking metric values are in [0, 1]."))

    if len(artifact.seed_set) == EXPECTED_SEED_COUNT:
        checks.append(("exactly_five_distinct_seeds", "PASS", f"Seeds={sorted(artifact.seed_set)}."))
    else:
        checks.append((
            "exactly_five_distinct_seeds",
            "WARN",
            f"Found {len(artifact.seed_set)} distinct seed(s): {sorted(artifact.seed_set)}.",
        ))

    checks.append((
        "algorithm_and_metric_coverage",
        "INFO",
        f"Algorithms={sorted(artifact.algorithms)}; metrics={metrics}; seed_source={artifact.seed_source}; "
        f"format={'long' if artifact.long_format else 'wide'}.",
    ))
    return checks


def metadata_for_artifact(artifact: ResultArtifact) -> tuple[list[int | str], str]:
    seed_lists = find_seed_lists(artifact.generated_dir)
    if not seed_lists:
        return [], "No JSON seed list found beside this artifact."
    normalized = [sorted(seed_key(seed) for seed in seeds) for seeds in seed_lists]
    unique_lists = {tuple(items) for items in normalized}
    if len(unique_lists) == 1:
        return seed_lists[0], f"JSON metadata seeds={normalized[0]}."
    return seed_lists[0], f"Multiple JSON seed lists found: {sorted(unique_lists)!r}."


def protocol_signature(artifact: ResultArtifact) -> dict[str, str]:
    seed_signature = json.dumps(sorted(artifact.seed_set))
    algorithm_signature = json.dumps(sorted(artifact.algorithms))
    metric_signature = json.dumps(sorted(artifact.metrics))
    split_signature = json.dumps(artifact.protocol_values, sort_keys=True)
    seed_lists = find_seed_lists(artifact.generated_dir)
    config_files = [
        path
        for path in sorted(artifact.generated_dir.glob("*.json"))
        if "config" in normalize(path.stem)
    ]
    config_payload = None
    if config_files:
        config_payload = read_json(config_files[-1])
    if config_payload is None and seed_lists:
        config_payload = {"seeds": seed_lists[0]}
    config_text = json.dumps(config_payload, sort_keys=True, separators=(",", ":")) if config_payload is not None else "missing"
    config_signature = hashlib.sha256(config_text.encode("utf-8")).hexdigest()[:16]
    return {
        "seed_signature": seed_signature,
        "algorithm_signature": algorithm_signature,
        "metric_signature": metric_signature,
        "split_signature": split_signature,
        "config_signature": config_signature,
    }


def final_rank(artifact: ResultArtifact) -> tuple[int, int, int, int, int, str]:
    complete_algorithms = int({"ALS", "Pop"}.issubset(artifact.algorithms))
    return (
        artifact.position if artifact.position is not None else -1,
        complete_algorithms,
        len(artifact.metrics),
        len(artifact.seed_set),
        1 if "per_run_results" in normalize(artifact.path.stem) else 0,
        relative(artifact.path),
    )


def select_final(artifacts: list[ResultArtifact]) -> ResultArtifact | None:
    return max(artifacts, key=final_rank) if artifacts else None


def final_code_path(
    run_dir: Path,
    positions: dict[str, int],
    preferred_node_id: str | None = None,
) -> Path | None:
    candidates = list((run_dir / "checkpoint").rglob("code.py"))
    if not candidates:
        return None

    if preferred_node_id:
        preferred = [path for path in candidates if path.parent.name == preferred_node_id]
        if preferred:
            candidates = preferred

    def rank(path: Path) -> tuple[int, str]:
        node_id = path.parent.name
        position = positions.get(node_id, -1)
        return position, relative(path)

    return max(candidates, key=rank)


def extract_code_seed_sets(text: str) -> list[set[str]]:
    seed_sets: list[set[str]] = []
    for match in CODE_SEED_PATTERN.finditer(text):
        values = {
            seed_key(parse_seed(value))
            for value in re.findall(r"-?\d+", match.group(1))
        }
        values.discard("")
        if values:
            seed_sets.append(values)
    return seed_sets


def add_report(
    report: list[dict[str, str]],
    run_label: str,
    check: str,
    status: str,
    details: str,
    artifact: Path | None = None,
    position: int | None = None,
):
    report.append({
        "run": run_label,
        "artifact": relative(artifact) if artifact else "",
        "position": "" if position is None else str(position),
        "check": check,
        "status": status,
        "details": details,
    })


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None):
    path.parent.mkdir(exist_ok=True)
    if fieldnames is None:
        fieldnames = []
        for row in rows:
            for key in row:
                if key not in fieldnames:
                    fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def audit_code(
    run_dir: Path,
    positions: dict[str, int],
    run_label: str,
    report: list[dict[str, str]],
    preferred_node_id: str | None = None,
    expected_seeds: set[str] | None = None,
):
    path = final_code_path(run_dir, positions, preferred_node_id)
    if path is None:
        add_report(report, run_label, "generated_code_present", "WARN", "No generated code.py was found.")
        return "", {}
    text = path.read_text(encoding="utf-8", errors="replace")
    features: dict[str, bool] = {}
    for name, pattern in CODE_FEATURES.items():
        present = re.search(pattern, text, re.IGNORECASE) is not None
        features[name] = present
        add_report(
            report,
            run_label,
            f"code_{name}",
            "PASS" if present else "WARN",
            f"{'Found' if present else 'Did not find'} pattern in final code {relative(path)}.",
            path,
            positions.get(path.parent.name),
        )
    code_seed_sets = extract_code_seed_sets(text)
    if not code_seed_sets:
        add_report(
            report,
            run_label,
            "code_seed_list_matches_result",
            "WARN",
            f"No explicit seed list assignment was found in {relative(path)}.",
            path,
            positions.get(path.parent.name),
        )
    elif expected_seeds is None:
        add_report(
            report,
            run_label,
            "code_seed_list_matches_result",
            "INFO",
            f"Code seed list(s)={sorted(code_seed_sets[0])}; no selected result seed set was available.",
            path,
            positions.get(path.parent.name),
        )
    else:
        matching = expected_seeds in code_seed_sets
        add_report(
            report,
            run_label,
            "code_seed_list_matches_result",
            "PASS" if matching else "WARN",
            f"Code seed list(s)={[sorted(values) for values in code_seed_sets]}; "
            f"selected result seeds={sorted(expected_seeds)}.",
            path,
            positions.get(path.parent.name),
        )
    return relative(path), features


def audit_dataset(run_dir: Path, run_label: str, report: list[dict[str, str]]):
    generated_dirs = sorted(
        path for path in (run_dir / "checkpoint").rglob("generated") if path.is_dir()
    )
    counts_candidates: list[tuple[Path, dict[str, int]]] = []
    for generated_dir in generated_dirs:
        counts, path = find_dataset_counts(generated_dir)
        if counts is not None and path is not None:
            counts_candidates.append((path, counts))
    if not counts_candidates:
        add_report(report, run_label, "dataset_counts", "WARN", "No users/items/interactions JSON was found.")
        return None, ""

    path, counts = counts_candidates[-1]
    valid = all(value >= 0 for value in counts.values())
    status = "PASS" if valid else "FAIL"
    add_report(
        report,
        run_label,
        "dataset_counts",
        status,
        f"Reported counts={counts}; source={relative(path)}. This checks recorded metadata, not raw-dataset independence.",
        path,
    )
    return counts, relative(path)


def audit_artifact(
    artifact: ResultArtifact,
    run_label: str,
    report: list[dict[str, str]],
):
    for check, status, details in artifact_checks(artifact):
        add_report(report, run_label, check, status, details, artifact.path, artifact.position)

    metadata_seeds, metadata_details = metadata_for_artifact(artifact)
    if metadata_seeds:
        recorded = sorted(seed_key(seed) for seed in metadata_seeds)
        observed = sorted(artifact.seed_set)
        status = "PASS" if recorded == observed else "WARN"
        details = f"{metadata_details} observed result seeds={observed}."
        add_report(report, run_label, "result_seeds_match_json_metadata", status, details, artifact.path, artifact.position)
    else:
        add_report(report, run_label, "result_seeds_match_json_metadata", "WARN", metadata_details, artifact.path, artifact.position)

    for summary_report in summary_checks(artifact):
        add_report(
            report,
            run_label,
            "summary_recalculation",
            summary_report["status"],
            summary_report["details"],
            artifact.path,
            artifact.position,
        )


def metric_map(artifact: ResultArtifact) -> dict[tuple[str, str, str], float]:
    groups: dict[tuple[str, str, str], float] = {}
    inferred_algorithm = infer_algorithm_from_path(artifact.path)
    for row, metric_row, seed in zip(artifact.rows, artifact.metric_rows, artifact.seed_values):
        raw_algorithm = row.get(artifact.algorithm_column, "") if artifact.algorithm_column else inferred_algorithm
        algorithm = canonical_algorithm(raw_algorithm)
        for metric, value in metric_row.items():
            if seed is not None and value is not None:
                groups[(algorithm, seed_key(seed), metric)] = value
    return groups


def cross_run_report(
    selected: list[tuple[str, ResultArtifact]],
    report: list[dict[str, str]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for index, (left_label, left_artifact) in enumerate(selected):
        left_signature = protocol_signature(left_artifact)
        left_values = metric_map(left_artifact)
        for right_label, right_artifact in selected[index + 1 :]:
            right_signature = protocol_signature(right_artifact)
            same_seed = left_signature["seed_signature"] == right_signature["seed_signature"]
            same_algorithms = left_signature["algorithm_signature"] == right_signature["algorithm_signature"]
            same_metrics = left_signature["metric_signature"] == right_signature["metric_signature"]
            same_split = left_signature["split_signature"] == right_signature["split_signature"]
            same_config = left_signature["config_signature"] == right_signature["config_signature"]
            comparable = same_seed and same_algorithms and same_metrics and same_split and same_config

            right_values = metric_map(right_artifact)
            common = set(left_values) & set(right_values)
            max_difference = max(
                (abs(left_values[key] - right_values[key]) for key in common),
                default=None,
            )
            if comparable:
                matching = max_difference is not None and max_difference <= TOLERANCE
                comparison_status = "MATCH" if matching else "DIFFER"
                details = f"Comparable protocols; common metric keys={len(common)}, max absolute difference={max_difference}."
            else:
                comparison_status = "NOT_COMPARABLE"
                details = (
                    f"Protocol differs: same_seeds={same_seed}, same_algorithms={same_algorithms}, "
                    f"same_metrics={same_metrics}, same_split_sizes={same_split}, same_config={same_config}."
                )

            row = {
                "left_run": left_label,
                "right_run": right_label,
                "left_artifact": relative(left_artifact.path),
                "right_artifact": relative(right_artifact.path),
                "same_seed_set": same_seed,
                "same_algorithms": same_algorithms,
                "same_metrics": same_metrics,
                "same_split_signature": same_split,
                "same_config_signature": same_config,
                "comparable": comparable,
                "comparison": comparison_status,
                "common_metric_keys": len(common),
                "max_absolute_difference": max_difference,
                "details": details,
            }
            rows.append(row)
            add_report(report, f"{left_label} vs {right_label}", "cross_run_protocol", comparison_status, details)
    return rows


def main() -> int:
    run_dirs = sorted(ROOT.glob("out_*"))
    if not run_dirs:
        print("No out_* experiment directories found next to this script.")
        return 1

    OUT_DIR.mkdir(exist_ok=True)
    report: list[dict[str, str]] = []
    selected: list[tuple[str, ResultArtifact]] = []
    summary_rows: list[dict[str, Any]] = []
    recomputed_rows: list[dict[str, Any]] = []

    for index, run_dir in enumerate(run_dirs, start=1):
        run_label = f"R{index} ({run_dir.name})"
        positions = load_positions(run_dir)
        artifacts = discover_artifacts(run_dir, positions)
        final_artifact = select_final(artifacts)
        dataset_counts, dataset_source = audit_dataset(run_dir, run_label, report)
        code_path, code_features = audit_code(
            run_dir,
            positions,
            run_label,
            report,
            final_artifact.node_id if final_artifact else None,
            final_artifact.seed_set if final_artifact else None,
        )

        add_report(
            report,
            run_label,
            "external_reexecution",
            "NOT_PERFORMED",
            "This verifier did not execute generated code outside the AutoRecLab runtime.",
        )
        add_report(
            report,
            run_label,
            "raw_metric_recalculation",
            "NOT_PERFORMED",
            "This verifier recomputes aggregate summaries only; it does not recompute NDCG/Precision from raw recommendations.",
        )

        for artifact in artifacts:
            audit_artifact(artifact, run_label, report)

        if final_artifact is not None:
            selected.append((run_label, final_artifact))
            for (algorithm, metric), values in sorted(group_values(final_artifact).items()):
                stats = calculated_stats(values)
                recomputed_rows.append({
                    "run": run_label,
                    "artifact": relative(final_artifact.path),
                    "algorithm": algorithm,
                    "metric": metric,
                    "n_values": len(values),
                    "seeds": ";".join(sorted(seed_key(seed) for seed, _ in values)),
                    **stats,
                })
            add_report(
                report,
                run_label,
                "selected_final_artifact",
                "INFO",
                f"Selected highest-position, most complete result artifact: {relative(final_artifact.path)}.",
                final_artifact.path,
                final_artifact.position,
            )
            selected_seeds = sorted(final_artifact.seed_set)
            selected_algorithms = sorted(final_artifact.algorithms)
            selected_metrics = sorted(final_artifact.metrics)
            selected_position = final_artifact.position
            selected_path = relative(final_artifact.path)
        else:
            add_report(report, run_label, "result_artifacts", "WARN", "No seed-indexed metric CSV was found.")
            selected_seeds = []
            selected_algorithms = []
            selected_metrics = []
            selected_position = None
            selected_path = ""

        summary_rows.append({
            "run": run_label,
            "directory": run_dir.name,
            "result_artifact_count": len(artifacts),
            "selected_final_artifact": selected_path,
            "selected_position": selected_position if selected_position is not None else "",
            "selected_seeds": ";".join(selected_seeds),
            "selected_algorithms": ";".join(selected_algorithms),
            "selected_metrics": ";".join(selected_metrics),
            "dataset_counts": json.dumps(dataset_counts, sort_keys=True) if dataset_counts else "",
            "dataset_counts_source": dataset_source,
            "final_code": code_path,
            "code_features_present": ";".join(name for name, present in code_features.items() if present),
            "code_features_missing": ";".join(name for name, present in code_features.items() if not present),
        })

    cross_rows = cross_run_report(selected, report)

    write_csv(
        OUT_DIR / "verification_report.csv",
        report,
        ["run", "artifact", "position", "check", "status", "details"],
    )
    write_csv(OUT_DIR / "verification_summary.csv", summary_rows)
    write_csv(OUT_DIR / "verification_cross_run.csv", cross_rows)
    write_csv(OUT_DIR / "verification_recomputed_stats.csv", recomputed_rows)

    print("\n================ Verification Summary ================\n")
    for row in summary_rows:
        print(
            f"{row['run']}: artifacts={row['result_artifact_count']}, "
            f"selected={row['selected_final_artifact'] or 'none'}, "
            f"seeds=[{row['selected_seeds']}], algorithms=[{row['selected_algorithms']}], "
            f"metrics=[{row['selected_metrics']}]"
        )

    print("\nCross-run comparisons:")
    for row in cross_rows:
        print(
            f"{row['left_run']} vs {row['right_run']}: {row['comparison']} "
            f"(same seeds={row['same_seed_set']}, same split={row['same_split_signature']}, "
            f"same config={row['same_config_signature']})"
        )

    status_counts: dict[str, int] = {}
    for row in report:
        status_counts[row["status"]] = status_counts.get(row["status"], 0) + 1
    print(f"\nReport statuses: {status_counts}")
    print(f"Detailed report: {OUT_DIR / 'verification_report.csv'}")
    print(f"Run summary:     {OUT_DIR / 'verification_summary.csv'}")
    print(f"Cross-run file:  {OUT_DIR / 'verification_cross_run.csv'}")
    print(f"Recomputed stats:{OUT_DIR / 'verification_recomputed_stats.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
