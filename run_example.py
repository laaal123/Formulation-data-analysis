"""Command-line demonstration on the synthetic dataset with known ground truth.

    python run_example.py                 # the confounded example, q30
    python run_example.py --response mdt  # a dissolution profile metric
"""
from __future__ import annotations

import argparse

from config import RunSettings
from pipeline import run_attribution
from record import to_markdown
from synthetic import GROUND_TRUTH, example_dataset


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--response", default="q30_pct")
    ap.add_argument("--profile-metric", default=None)
    ap.add_argument("--medium", default="0.1N HCl")
    ap.add_argument("--model", default="elastic_net")
    ap.add_argument("--scheme", default="leave_one_formulation_out")
    ap.add_argument("--permutations", type=int, default=1000)
    ap.add_argument("--bootstrap", type=int, default=500)
    ap.add_argument("--record", default=None, help="write the method record to this path")
    args = ap.parse_args()

    ds = example_dataset()
    print(ds.report.text())
    print()
    res = run_attribution(
        ds,
        None if args.profile_metric else args.response,
        profile_metric=args.profile_metric,
        medium=args.medium if args.profile_metric else None,
        model_type=args.model,
        scheme=args.scheme,
        settings=RunSettings(n_permutations=args.permutations, n_bootstrap=args.bootstrap),
    )
    print(res.text())
    print()
    print("Ground truth of this synthetic dataset:")
    for k, v in GROUND_TRUTH.items():
        print(f"  {k}: {v}")
    if args.record:
        with open(args.record, "w") as fh:
            fh.write(to_markdown(res))
        print(f"\nMethod record written to {args.record}")


if __name__ == "__main__":
    main()
