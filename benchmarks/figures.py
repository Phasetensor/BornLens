"""Render validation panels from retained metrics and fixed-seed illustrative draws."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).parent / "acceptance"


def render():
    plt.rcParams.update(
        {
            "svg.fonttype": "none",
            "svg.hashsalt": "bornlens",
            "font.size": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    rows = list(csv.DictReader((ROOT / "metrics.csv").open()))

    def values(area, metric, keep=lambda s: True):
        return [
            (r["scenario"], float(r["estimate"]))
            for r in rows
            if r["area"] == area and r["metric"] == metric and keep(r["scenario"])
        ]

    fig, axes = plt.subplots(4, 3, figsize=(13, 13), constrained_layout=True)
    panels = iter(axes.flat)

    def panel(title, ylabel):
        ax = next(panels)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_ylabel(ylabel)
        return ax

    def bars(ax, rows):
        labels, y = zip(*rows)
        ax.bar(range(len(y)), y, color="#345775")
        ax.set_xticks(range(len(y)), labels, rotation=30, ha="right")

    ax = panel("Multiple testing", "Empirical FDR")
    bars(
        ax,
        [
            (s.split("/")[0], v)
            for s, v in values(
                "multiplicity", "FDR", lambda s: s.endswith("/12/0.65/True")
            )
        ],
    )
    ax = panel("Sequential monitoring", "Half-width")
    ns = [1, 16, 64, 256]
    ax.plot(
        ns, [values("sequence-path", f"halfwidth-{n}")[0][1] for n in ns], marker="o"
    )
    ax.set_xlabel("Cumulative shots per arm")
    ax = panel("Coverage over 256 looks", "Probability")
    for metric in ["sequence", "naive", "fixed_at_horizon"]:
        pairs = values("sequential", metric)
        ax.plot(range(len(pairs)), [v for _, v in pairs], marker=".", label=metric)
    ax.legend()
    ax.set_xticks(range(len(pairs)), [s for s, _ in pairs], rotation=30)
    ax = panel("Beta posterior change", "Density")
    rng = np.random.default_rng(4703)
    for x, y in [(0.5, 0.5), (10, 2)]:
        delta = 2 * (rng.beta(3 + x, 7 + y, 50000) - rng.beta(2 + x, 5 + y, 50000))
        ax.hist(delta, bins=35, density=True, histtype="step", label=f"Beta({x},{y})")
    ax.legend()
    ax.set_xlabel("Pauli change")
    ax = panel("Paired percentile bootstrap", "Empirical coverage")
    for m in ["paired_boot_coverage", "unpaired_boot_coverage"]:
        ax.plot(
            range(5),
            [v for _, v in values("paired", m, lambda s: s.endswith("/12"))],
            marker="o",
            label=m.split("_")[0],
        )
    ax.axhline(0.95, ls=":", color="gray")
    ax.set_xticks(
        range(5), ["positive", "negative", "indep.", "matched", "rare"], rotation=30
    )
    ax.legend()
    ax = panel("Categorical permutation test", "Rejection rate")
    bars(
        ax,
        list(
            zip(
                ["null", "sparse", "degenerate"],
                [v for _, v in values("distribution", "rejection_rate")],
            )
        ),
    )
    ax = panel("Drift detection", "Alarm probability")
    bars(
        ax,
        [
            (s.split("/")[-1], v)
            for s, v in values(
                "drift", "alarm_probability", lambda s: s.startswith("128/32/")
            )
        ],
    )
    ax = panel("Allocation n : 2n", "Correct classification")
    for pair in [".5/.5", "0.01/0.02", "0.2/0.6"]:
        prefix = pair.replace(".5/.5", "0.5/0.5") + "/"
        vals = values(
            "design", "correct", lambda s, prefix=prefix: s.startswith(prefix)
        )
        ax.plot(
            [int(s.split("/")[-1]) for s, _ in vals],
            [v for _, v in vals],
            marker="o",
            label=prefix[:-1],
        )
    ax.legend()
    ax.set_xlabel("Shots in A")
    ax = panel("Hierarchical shrinkage", "Mean squared error")
    for m in ["shrinkage_mse", "raw_mse"]:
        vals = values(
            "hierarchy", m, lambda s: s.startswith("3/") and s.endswith("False")
        )
        ax.plot(
            [float(s.split("/")[1]) for s, _ in vals],
            [v for _, v in vals],
            marker="o",
            label=m,
        )
    ax.legend()
    ax.set_xlabel("True between-group SD")
    ax = panel("Circuit regression examples", "Decision")
    codes = {"fail": -1, "inconclusive": 0, "pass": 1}
    reg = [r for r in rows if r["area"] == "regression"]
    bars(ax, [(r["scenario"], codes[r["metric"]]) for r in reg])
    ax.set_yticks([-1, 0, 1], ["fail", "inconclusive", "pass"])
    ax.set_xlabel("Shots / angle (rad)")
    ax = panel("Validation runtime", "Seconds")
    timing = json.loads((ROOT / "metadata.json").read_text())[
        "runtime_seconds_and_peak_traced_bytes"
    ]
    bars(ax, [(k, v[0]) for k, v in timing.items()])
    ax = panel("Peak traced allocations", "MiB")
    bars(ax, [(k, v[1] / 2**20) for k, v in timing.items()])
    fig.savefig(ROOT / "validation.svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    render()
