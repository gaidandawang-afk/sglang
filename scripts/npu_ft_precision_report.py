#!/usr/bin/env python3

import argparse
import ast
import json
import re
import time
from collections import Counter, defaultdict
from pathlib import Path


STATE_RE = re.compile(
    r"NPU FT precision state stage=(?P<stage>\S+) rank=(?P<rank>\d+) "
    r"layer_id=(?P<layer>\d+) local_slot=(?P<slot>\d+) "
    r"logical_expert_id=(?P<expert>\S+) tensor_index=(?P<tensor>\d+) "
    r"fields=(?P<fields>\{.*\})"
)
PLAN_MARKER = "NPU FT precision step=eplb_transfer_plan"
PLAN_RE = re.compile(
    r"NPU FT precision step=eplb_transfer_plan layer_id=(?P<layer>\d+) "
    r"original_rank=(?P<rank>\d+) "
    r"active_original_ranks=(?P<active>.*?) local_changes=.*? "
    r"movement_plan=(?P<plan>\[.*\]) "
    r"p2p_plan="
)
RECOVERY_FILTER_MARKER = "[Elastic EP] Missing expert recovery filter"
RECOVERY_FILTER_RE = re.compile(
    r"\[Elastic EP\] Missing expert recovery filter rank=(?P<rank>\d+) "
    r"mode=(?P<mode>\S+) "
    r"checked=(?P<checked>\d+) matched=(?P<matched>\d+) "
    r"matched_samples=(?P<samples>\[.*\])"
)
RECOVERY_PLAN_MARKER = "[Elastic EP] Missing expert recovery plan"
RECOVERY_PLAN_RE = re.compile(
    r"\[Elastic EP\] Missing expert recovery plan rank=(?P<rank>\d+) "
    r"missing_by_layer=(?P<missing>\{.*?\}) "
    r"local_target_count=(?P<count>\d+) "
    r"local_target_samples=(?P<samples>\[.*\])"
)
MOVEMENT_KINDS = (
    "same-gpu",
    "free-rider",
    "same-node-p2p",
    "cross-node-p2p",
    "missing",
)


def _iter_recent_files(paths: list[Path], since_minutes: int):
    cutoff = time.time() - since_minutes * 60
    for path in paths:
        candidates = path.rglob("*") if path.is_dir() else (path,)
        for candidate in candidates:
            try:
                if candidate.is_file() and candidate.stat().st_mtime >= cutoff:
                    yield candidate
            except OSError:
                continue


def _sample_fingerprint(fields: dict):
    if "sample_values_raw" in fields:
        return repr(
            (fields.get("sample_coordinates_raw"), fields["sample_values_raw"])
        )
    if "sample_values" not in fields:
        return None
    return repr((fields.get("sample_coordinates"), fields["sample_values"]))


def _raw_field(fields_text: str, name: str):
    marker = f"'{name}': "
    start = fields_text.find(marker)
    if start < 0:
        return None
    start += len(marker)
    end = fields_text.find(", '", start)
    if end < 0:
        end = len(fields_text) - 1 if fields_text.endswith("}") else len(fields_text)
    return fields_text[start:end]


def _parse_fields(fields_text: str):
    try:
        return ast.literal_eval(fields_text), False
    except (SyntaxError, ValueError):
        fields = {
            name: _raw_field(fields_text, name)
            for name in (
                "storage_offset",
                "acl_format",
                "npu_storage_size",
                "sample_error",
            )
        }
        fields = {name: value for name, value in fields.items() if value is not None}

        coordinates_marker = "'sample_coordinates': "
        values_marker = ", 'sample_values': "
        coordinates_start = fields_text.find(coordinates_marker)
        values_start = fields_text.find(values_marker)
        if coordinates_start >= 0 and values_start > coordinates_start:
            coordinates_start += len(coordinates_marker)
            fields["sample_coordinates_raw"] = fields_text[
                coordinates_start:values_start
            ]
            values_start += len(values_marker)
            values_end = (
                len(fields_text) - 1
                if fields_text.endswith("}")
                else len(fields_text)
            )
            fields["sample_values_raw"] = fields_text[values_start:values_end]
        return fields, True


def _slot_key(record: dict, *, include_stage_cycle: bool):
    key = (
        record["rank"],
        record["layer"],
        record["slot"],
        record["tensor"],
    )
    return (record.get("stage_cycle"), *key) if include_stage_cycle else key


def _expert_key(record: dict):
    return (
        record.get("migration_epoch"),
        record["layer"],
        record["expert"],
        record["tensor"],
    )


def _record_summary(record: dict):
    fields = record["fields"]
    movement = record.get("movement")
    if movement is None and record["stage"].startswith("missing_recovery_"):
        movement = {"kind": "missing-recovery"}
    return {
        "rank": record["rank"],
        "layer": record["layer"],
        "slot": record["slot"],
        "expert": record["expert"],
        "tensor": record["tensor"],
        "migration_epoch": record.get("migration_epoch"),
        "stage_cycle": record.get("stage_cycle"),
        "format": fields.get("acl_format"),
        "offset": fields.get("storage_offset"),
        "storage_size": fields.get("npu_storage_size"),
        "samples": fields.get("sample_values", fields.get("sample_values_raw")),
        "sample_error": fields.get("sample_error"),
        "movement": movement or {"kind": "unknown"},
    }


def _compare_same_slot(
    records_by_stage,
    before_stage,
    after_stage,
    *,
    include_stage_cycle=True,
):
    before = {
        _slot_key(record, include_stage_cycle=include_stage_cycle): record
        for record in records_by_stage[before_stage]
    }
    output = []
    for after_record in records_by_stage[after_stage]:
        before_record = before.get(
            _slot_key(after_record, include_stage_cycle=include_stage_cycle)
        )
        if before_record is None:
            continue
        before_fp = _sample_fingerprint(before_record["fields"])
        after_fp = _sample_fingerprint(after_record["fields"])
        if before_fp is not None and after_fp is not None and before_fp != after_fp:
            output.append(
                {
                    "before": _record_summary(before_record),
                    "after": _record_summary(after_record),
                }
            )
    return output


def _compare_migration(records_by_stage):
    before_by_expert = defaultdict(set)
    for record in records_by_stage["migration_before"]:
        fingerprint = _sample_fingerprint(record["fields"])
        if fingerprint is not None:
            before_by_expert[_expert_key(record)].add(fingerprint)

    output = []
    for after_record in records_by_stage["migration_after"]:
        fingerprint = _sample_fingerprint(after_record["fields"])
        expected = before_by_expert.get(_expert_key(after_record), set())
        if fingerprint is not None and expected and fingerprint not in expected:
            output.append(_record_summary(after_record))
    return output


def _analyze_missing_recovery(records_by_stage):
    post_load_by_expert = defaultdict(set)
    for record in records_by_stage["post_load"]:
        fingerprint = _sample_fingerprint(record["fields"])
        if fingerprint is not None:
            post_load_by_expert[
                (record["layer"], record["expert"], record["tensor"])
            ].add(fingerprint)

    before_by_slot = {
        (
            record.get("migration_epoch"),
            record["rank"],
            record["layer"],
            record["slot"],
            record["tensor"],
        ): record
        for record in records_by_stage["missing_recovery_before"]
    }
    counts = Counter()
    mismatches = []
    for after_record in records_by_stage["missing_recovery_after"]:
        after_fp = _sample_fingerprint(after_record["fields"])
        if after_fp is None:
            counts["unsampled"] += 1
            continue

        before_record = before_by_slot.get(
            (
                after_record.get("migration_epoch"),
                after_record["rank"],
                after_record["layer"],
                after_record["slot"],
                after_record["tensor"],
            )
        )
        before_fp = (
            _sample_fingerprint(before_record["fields"])
            if before_record is not None
            else None
        )
        counts["changed" if before_fp != after_fp else "unchanged"] += 1

        expected = post_load_by_expert.get(
            (
                after_record["layer"],
                after_record["expert"],
                after_record["tensor"],
            ),
            set(),
        )
        if not expected:
            counts["no_post_load_reference"] += 1
        elif after_fp in expected:
            counts["matches_post_load"] += 1
        else:
            counts["differs_from_post_load"] += 1
            mismatches.append(
                {
                    "before": (
                        _record_summary(before_record)
                        if before_record is not None
                        else None
                    ),
                    "after": _record_summary(after_record),
                }
            )
    return counts, mismatches


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--since-minutes", type=int, default=180)
    parser.add_argument("--max-mismatches", type=int, default=20)
    args = parser.parse_args()

    records_by_stage = defaultdict(list)
    stage_rank_counts = Counter()
    sampled_stage_rank_counts = Counter()
    movement_counts = Counter()
    movement_entries = []
    movement_plan_parse_errors = 0
    migration_epoch_by_rank = defaultdict(int)
    last_state_stage_by_rank = {}
    stage_cycle_by_rank_and_base = defaultdict(int)
    migration_epoch_contexts = {}
    missing_recovery_filter_stats = {}
    missing_recovery_plan_stats = {}
    files_scanned = 0
    matched_lines = 0
    parse_errors = 0
    field_parse_fallbacks = 0

    for path in _iter_recent_files(args.paths, args.since_minutes):
        files_scanned += 1
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as file:
                for line in file:
                    if not any(
                        marker in line
                        for marker in (
                            "NPU FT precision",
                            RECOVERY_FILTER_MARKER,
                            RECOVERY_PLAN_MARKER,
                        )
                    ):
                        continue
                    if RECOVERY_FILTER_MARKER in line:
                        match = RECOVERY_FILTER_RE.search(line)
                        if match is not None:
                            rank = int(match.group("rank"))
                            try:
                                samples = ast.literal_eval(match.group("samples"))
                            except (SyntaxError, ValueError):
                                samples = match.group("samples")
                            missing_recovery_filter_stats[
                                (migration_epoch_by_rank[rank], rank)
                            ] = {
                                "mode": match.group("mode"),
                                "checked": int(match.group("checked")),
                                "matched": int(match.group("matched")),
                                "matched_samples": samples,
                            }
                        continue
                    if RECOVERY_PLAN_MARKER in line:
                        match = RECOVERY_PLAN_RE.search(line)
                        if match is not None:
                            rank = int(match.group("rank"))
                            try:
                                missing = ast.literal_eval(match.group("missing"))
                            except (SyntaxError, ValueError):
                                missing = match.group("missing")
                            try:
                                samples = ast.literal_eval(match.group("samples"))
                            except (SyntaxError, ValueError):
                                samples = match.group("samples")
                            missing_recovery_plan_stats[
                                (migration_epoch_by_rank[rank], rank)
                            ] = {
                                "missing_by_layer": missing,
                                "local_target_count": int(match.group("count")),
                                "local_target_samples": samples,
                            }
                        continue
                    if PLAN_MARKER in line:
                        matched_lines += 1
                        for kind in MOVEMENT_KINDS:
                            movement_counts[kind] += line.count(f"'{kind}'")
                        plan_match = PLAN_RE.search(line)
                        if plan_match is not None:
                            try:
                                plan_rank = int(plan_match.group("rank"))
                                active_text = plan_match.group("active")
                                try:
                                    active_ranks = ast.literal_eval(active_text)
                                except (SyntaxError, ValueError):
                                    active_ranks = active_text
                                migration_epoch_contexts[
                                    (migration_epoch_by_rank[plan_rank], plan_rank)
                                ] = active_ranks
                                plan = ast.literal_eval(plan_match.group("plan"))
                                for expert, kind, source, destination in plan:
                                    movement_entries.append(
                                        {
                                            "migration_epoch": migration_epoch_by_rank[
                                                plan_rank
                                            ],
                                            "layer": int(plan_match.group("layer")),
                                            "rank": plan_rank,
                                            "active_original_ranks": active_ranks,
                                            "expert": expert,
                                            "kind": kind,
                                            "source": source,
                                            "destination": destination,
                                        }
                                    )
                            except (SyntaxError, ValueError, TypeError):
                                movement_plan_parse_errors += 1
                        continue
                    match = STATE_RE.search(line)
                    if match is None:
                        continue
                    matched_lines += 1
                    fields, used_fallback = _parse_fields(match.group("fields"))
                    field_parse_fallbacks += int(used_fallback)
                    if not fields:
                        parse_errors += 1
                        continue
                    expert_text = match.group("expert")
                    stage = match.group("stage")
                    rank = int(match.group("rank"))
                    if (
                        stage == "migration_before"
                        and last_state_stage_by_rank.get(rank) != "migration_before"
                    ):
                        migration_epoch_by_rank[rank] += 1
                    record = {
                        "stage": stage,
                        "rank": rank,
                        "layer": int(match.group("layer")),
                        "slot": int(match.group("slot")),
                        "expert": (
                            None if expert_text == "None" else int(expert_text)
                        ),
                        "tensor": int(match.group("tensor")),
                        "fields": fields,
                    }
                    if stage in (
                        "migration_before",
                        "migration_after",
                        "missing_recovery_before",
                        "missing_recovery_after",
                    ):
                        record["migration_epoch"] = migration_epoch_by_rank[rank]
                    elif stage.endswith("_before"):
                        stage_base = stage[: -len("_before")]
                        if last_state_stage_by_rank.get(rank) != stage:
                            stage_cycle_by_rank_and_base[(rank, stage_base)] += 1
                        record["stage_cycle"] = stage_cycle_by_rank_and_base[
                            (rank, stage_base)
                        ]
                    elif stage.endswith("_after"):
                        stage_base = stage[: -len("_after")]
                        record["stage_cycle"] = stage_cycle_by_rank_and_base[
                            (rank, stage_base)
                        ]
                    last_state_stage_by_rank[rank] = stage
                    records_by_stage[record["stage"]].append(record)
                    stage_rank_counts[(record["stage"], record["rank"])] += 1
                    if _sample_fingerprint(fields) is not None:
                        sampled_stage_rank_counts[
                            (record["stage"], record["rank"])
                        ] += 1
        except OSError:
            continue

    all_records = [record for records in records_by_stage.values() for record in records]
    num_local_experts = (
        max(record["slot"] for record in all_records) + 1 if all_records else None
    )
    movement_by_destination = {}
    if num_local_experts is not None:
        for entry in movement_entries:
            movement_by_destination[
                (
                    entry["layer"],
                    entry["rank"],
                    entry["migration_epoch"],
                    entry["expert"],
                    entry["destination"] % num_local_experts,
                )
            ] = entry
        for record in records_by_stage["migration_after"]:
            movement = movement_by_destination.get(
                (
                    record["layer"],
                    record["rank"],
                    record.get("migration_epoch"),
                    record["expert"],
                    record["slot"],
                )
            )
            record["movement"] = movement or {"kind": "unchanged"}

    same_slot_checks = {
        "post_load_to_first_forward_before": _compare_same_slot(
            records_by_stage,
            "post_load",
            "first_forward_before",
            include_stage_cycle=False,
        ),
        "first_forward_before_to_after": _compare_same_slot(
            records_by_stage, "first_forward_before", "first_forward_after"
        ),
        "post_scale_down_forward_before_to_after": _compare_same_slot(
            records_by_stage,
            "post_scale_down_forward_before",
            "post_scale_down_forward_after",
        ),
        "first_real_post_scale_down_forward_before_to_after": _compare_same_slot(
            records_by_stage,
            "first_real_post_scale_down_forward_before",
            "first_real_post_scale_down_forward_after",
        ),
    }
    migration_mismatches = _compare_migration(records_by_stage)
    recovery_counts, recovery_mismatches = _analyze_missing_recovery(records_by_stage)
    migration_mismatch_kinds = Counter(
        item.get("movement", {}).get("kind", "unknown")
        for item in migration_mismatches
    )
    migration_mismatch_epochs = Counter(
        f"epoch{item.get('migration_epoch')}"
        for item in migration_mismatches
    )
    migration_epoch_stage_rank_counts = Counter(
        (
            record.get("migration_epoch"),
            record["stage"],
            record["rank"],
        )
        for stage in ("migration_before", "migration_after")
        for record in records_by_stage[stage]
    )
    movement_counts_by_epoch = Counter(
        (entry["migration_epoch"], entry["kind"]) for entry in movement_entries
    )

    report = {
        "files_scanned": files_scanned,
        "matched_lines": matched_lines,
        "parse_errors": parse_errors,
        "field_parse_fallbacks": field_parse_fallbacks,
        "movement_plan_parse_errors": movement_plan_parse_errors,
        "num_local_experts": num_local_experts,
        "stage_rank_counts": {
            f"{stage}/rank{rank}": count
            for (stage, rank), count in sorted(stage_rank_counts.items())
        },
        "sampled_stage_rank_counts": {
            f"{stage}/rank{rank}": count
            for (stage, rank), count in sorted(sampled_stage_rank_counts.items())
        },
        "movement_counts": dict(movement_counts),
        "missing_recovery_counts": dict(sorted(recovery_counts.items())),
        "missing_recovery_filter_stats": {
            f"epoch{epoch}/rank{rank}": stats
            for (epoch, rank), stats in sorted(
                missing_recovery_filter_stats.items()
            )
        },
        "missing_recovery_plan_stats": {
            f"epoch{epoch}/rank{rank}": stats
            for (epoch, rank), stats in sorted(missing_recovery_plan_stats.items())
        },
        "migration_mismatch_counts_by_movement": dict(
            sorted(migration_mismatch_kinds.items())
        ),
        "migration_mismatch_counts_by_epoch": dict(
            sorted(migration_mismatch_epochs.items())
        ),
        "migration_epoch_stage_rank_counts": {
            f"epoch{epoch}/{stage}/rank{rank}": count
            for (epoch, stage, rank), count in sorted(
                migration_epoch_stage_rank_counts.items()
            )
        },
        "movement_counts_by_epoch": {
            f"epoch{epoch}/{kind}": count
            for (epoch, kind), count in sorted(movement_counts_by_epoch.items())
        },
        "migration_epoch_active_ranks": {
            f"epoch{epoch}/rank{rank}": active_ranks
            for (epoch, rank), active_ranks in sorted(
                migration_epoch_contexts.items()
            )
        },
        "mismatch_counts": {
            **{name: len(items) for name, items in same_slot_checks.items()},
            "migration_before_to_after": len(migration_mismatches),
            "missing_recovery_differs_from_post_load": len(recovery_mismatches),
        },
        "mismatches": {
            **{
                name: items[: args.max_mismatches]
                for name, items in same_slot_checks.items()
            },
            "migration_before_to_after": migration_mismatches[
                : args.max_mismatches
            ],
            "missing_recovery_differs_from_post_load": recovery_mismatches[
                : args.max_mismatches
            ],
        },
    }
    print("NPU_FT_PRECISION_REPORT_BEGIN")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print("NPU_FT_PRECISION_REPORT_END")


if __name__ == "__main__":
    main()
