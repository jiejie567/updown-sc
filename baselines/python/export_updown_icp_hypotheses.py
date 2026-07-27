#!/usr/bin/env python3
import csv
from pathlib import Path


ROOT = Path("${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/baseline_20260715")
UPDOWN = Path(
    "${UPDOWN_SC_ROOT}/icra2027_runtime/experiments/two_bag_pilot_20260715/"
    "results/loc_2_floor_sector_local_w04_w06")
OUTPUT = ROOT / "results/updown_icp_hypotheses.csv"
RETRIEVAL_OUTPUT = ROOT / "results/updown_sector_local_w04_w06_per_query.csv"


def main():
    with (ROOT / "queries/loc_2_floor/metadata.csv").open(newline="") as handle:
        metadata = list(csv.DictReader(handle))
    query_by_window = {
        int(row["window"]): int(row["query_id"])
        for row in metadata if row["truth_valid"].lower() == "true"
    }
    with (UPDOWN / "relocalization_windows.csv").open(newline="") as handle:
        retrieval_ms = {
            int(row["window"]): float(row["scan_context_ms"])
            for row in csv.DictReader(handle)
        }

    rows = []
    with (UPDOWN / "scan_context_candidate_hypotheses.csv").open(newline="") as handle:
        for source in csv.DictReader(handle):
            window = int(source["window"])
            if window not in query_by_window or int(source["hypothesis_rank"]) != 1:
                continue
            rows.append({
                "algorithm": "UpDown-SC",
                "query_id": query_by_window[window],
                "window": window,
                "candidate_rank": source["candidate_rank"],
                "candidate_index": source["candidate_index"],
                "distance": source["distance"],
                "sector_shift": source["sector_shift"],
                "yaw_shift_rad": source["yaw_shift_rad"],
                "root_shift_y": 0.0,
                "vertical_shift": source["vertical_shift"],
                "retrieval_ms": retrieval_ms[window],
            })

    rows.sort(key=lambda row: (int(row["query_id"]), int(row["candidate_rank"])))
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} hypotheses to {OUTPUT}")

    truth_by_window = {
        int(row["window"]): row
        for row in metadata if row["truth_valid"].lower() == "true"
    }
    source_by_window_rank = {}
    with (UPDOWN / "scan_context_candidate_hypotheses.csv").open(newline="") as handle:
        for source in csv.DictReader(handle):
            if int(source["hypothesis_rank"]) == 1:
                source_by_window_rank[(int(source["window"]), int(source["candidate_rank"]))] = source
    grouped = {}
    for row in rows:
        grouped.setdefault(int(row["window"]), []).append(row)
    retrieval_rows = []
    for window in sorted(grouped):
        truth = truth_by_window[window]
        truth_x = float(truth["truth_x"])
        truth_y = float(truth["truth_y"])
        candidates = sorted(grouped[window], key=lambda row: int(row["candidate_rank"]))
        sources = [
            source_by_window_rank[(window, int(candidate["candidate_rank"]))]
            for candidate in candidates
        ]
        errors = [
            ((float(source["candidate_x"]) - truth_x) ** 2 +
             (float(source["candidate_y"]) - truth_y) ** 2) ** 0.5
            for source in sources
        ]
        first_correct = next((rank + 1 for rank, error in enumerate(errors) if error <= 2.0), None)
        top = sources[0]
        retrieval_rows.append({
            "algorithm": "UpDown-SC",
            "query_id": query_by_window[window],
            "window": window,
            "start_s": truth["start_s"],
            "truth_x": truth_x,
            "truth_y": truth_y,
            "top1_index": top["candidate_index"],
            "top1_x": top["candidate_x"],
            "top1_y": top["candidate_y"],
            "top1_error_m": errors[0],
            "first_correct_rank": first_correct if first_correct is not None else "",
            "recall_at_1": first_correct is not None and first_correct <= 1,
            "recall_at_5": first_correct is not None and first_correct <= 5,
            "recall_at_10": first_correct is not None and first_correct <= 10,
            "recall_at_100": first_correct is not None and first_correct <= 100,
            "retrieval_ms": retrieval_ms[window],
        })
    with RETRIEVAL_OUTPUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(retrieval_rows[0]))
        writer.writeheader()
        writer.writerows(retrieval_rows)
    print(f"wrote {len(retrieval_rows)} retrieval rows to {RETRIEVAL_OUTPUT}")


if __name__ == "__main__":
    main()
