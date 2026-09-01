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
    if "sample_values" not in fields:
        return None
    return repr((fields.get("sample_coordinates"), fields["sample_values"]))


def _slot_key(record: dict):
    return (
        record["rank"],
        record["layer"],
        record["slot"],
        record["tensor"],
    )


def _expert_key(record: dict):
    return record["layer"], record["expert"], record["tensor"]


def _record_summary(record: dict):
    fields = record["fields"]
    return {
        "rank": record["rank"],
        "layer": record["layer"],
        "slot": record["slot"],
        "expert": record["expert"],
        "tensor": record["tensor"],
        "format": fields.get("acl_format"),
        "offset": fields.get("storage_offset"),
        "storage_size": fields.get("npu_storage_size"),
        "samples": fields.get("sample_values"),
        "sample_error": fields.get("sample_error"),
    }


def _compare_same_slot(records_by_stage, before_stage, after_stage):
    before = {_slot_key(record): record for record in records_by_stage[before_stage]}
    output = []
    for after_record in records_by_stage[after_stage]:
        before_record = before.get(_slot_key(after_record))
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
    files_scanned = 0
    matched_lines = 0
    parse_errors = 0

    for path in _iter_recent_files(args.paths, args.since_minutes):
        files_scanned += 1
        try:
            with path.open("r", encoding="utf-8", errors="ignore") as file:
                for line in file:
                    if "NPU FT precision" not in line:
                        continue
                    if PLAN_MARKER in line:
                        matched_lines += 1
                        for kind in MOVEMENT_KINDS:
                            movement_counts[kind] += line.count(f"'{kind}'")
                        continue
                    match = STATE_RE.search(line)
                    if match is None:
                        continue
                    matched_lines += 1
                    try:
                        fields = ast.literal_eval(match.group("fields"))
                    except (SyntaxError, ValueError):
                        parse_errors += 1
                        continue
                    expert_text = match.group("expert")
                    record = {
                        "stage": match.group("stage"),
                        "rank": int(match.group("rank")),
                        "layer": int(match.group("layer")),
                        "slot": int(match.group("slot")),
                        "expert": (
                            None if expert_text == "None" else int(expert_text)
                        ),
                        "tensor": int(match.group("tensor")),
                        "fields": fields,
                    }
                    records_by_stage[record["stage"]].append(record)
                    stage_rank_counts[(record["stage"], record["rank"])] += 1
                    if _sample_fingerprint(fields) is not None:
                        sampled_stage_rank_counts[
                            (record["stage"], record["rank"])
                        ] += 1
        except OSError:
            continue

    same_slot_checks = {
        "post_load_to_first_forward_before": _compare_same_slot(
            records_by_stage, "post_load", "first_forward_before"
        ),
        "first_forward_before_to_after": _compare_same_slot(
            records_by_stage, "first_forward_before", "first_forward_after"
        ),
    }
    migration_mismatches = _compare_migration(records_by_stage)

    report = {
        "files_scanned": files_scanned,
        "matched_lines": matched_lines,
        "parse_errors": parse_errors,
        "stage_rank_counts": {
            f"{stage}/rank{rank}": count
            for (stage, rank), count in sorted(stage_rank_counts.items())
        },
        "sampled_stage_rank_counts": {
            f"{stage}/rank{rank}": count
            for (stage, rank), count in sorted(sampled_stage_rank_counts.items())
        },
        "movement_counts": dict(movement_counts),
        "mismatch_counts": {
            **{name: len(items) for name, items in same_slot_checks.items()},
            "migration_before_to_after": len(migration_mismatches),
        },
        "mismatches": {
            **{
                name: items[: args.max_mismatches]
                for name, items in same_slot_checks.items()
            },
            "migration_before_to_after": migration_mismatches[
                : args.max_mismatches
            ],
        },
    }
    print("NPU_FT_PRECISION_REPORT_BEGIN")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    print("NPU_FT_PRECISION_REPORT_END")


if __name__ == "__main__":
    main()
