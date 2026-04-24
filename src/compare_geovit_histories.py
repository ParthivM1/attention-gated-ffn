from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_history(path: str | Path) -> list[dict[str, Any]]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def summarize(old_history: list[dict[str, Any]], new_history: list[dict[str, Any]]) -> dict[str, Any]:
    paired = list(zip(old_history, new_history))
    divergence_epoch = None
    worst_residual_gap = None
    worst_accuracy_gap = None
    rows: list[dict[str, Any]] = []
    for old_row, new_row in paired:
        epoch = int(old_row["epoch"])
        old_acc = float(old_row["test_acc"])
        new_acc = float(new_row["test_acc"])
        old_res = float(old_row.get("fc1_spectral_residual_norm", 0.0))
        new_res = float(new_row.get("fc1_spectral_residual_norm", 0.0))
        acc_gap = new_acc - old_acc
        res_gap = new_res - old_res
        rows.append(
            {
                "epoch": epoch,
                "old_test_acc": old_acc,
                "new_test_acc": new_acc,
                "acc_gap": acc_gap,
                "old_spectral_residual_norm": old_res,
                "new_spectral_residual_norm": new_res,
                "residual_gap": res_gap,
            }
        )
        if divergence_epoch is None and res_gap > 0.05 and acc_gap < -1.0:
            divergence_epoch = epoch
        if worst_residual_gap is None or res_gap > worst_residual_gap["residual_gap"]:
            worst_residual_gap = rows[-1]
        if worst_accuracy_gap is None or acc_gap < worst_accuracy_gap["acc_gap"]:
            worst_accuracy_gap = rows[-1]
    return {
        "old_best_acc": max(float(row["test_acc"]) for row in old_history),
        "new_best_acc": max(float(row["test_acc"]) for row in new_history),
        "divergence_epoch": divergence_epoch,
        "worst_residual_gap": worst_residual_gap,
        "worst_accuracy_gap": worst_accuracy_gap,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare two GeoViT history.json files.")
    parser.add_argument("--old-history", required=True)
    parser.add_argument("--new-history", required=True)
    parser.add_argument("--output", type=str, default="")
    args = parser.parse_args()

    payload = summarize(load_history(args.old_history), load_history(args.new_history))
    text = json.dumps(payload, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
