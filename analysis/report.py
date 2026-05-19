from __future__ import annotations

from pathlib import Path

from analysis.json_util import dumps_pretty
from analysis.plotting import bar_with_errors, fold_line_plot
from analysis.stats_utils import comparison_table
from app.data import DATA_SOURCE, load_student_math
from app.train import run_comparison


def generate_report(out_dir: Path | None = None, cv_splits: int = 3) -> dict:
    """Execute the generate report routine."""
    out = Path(out_dir or "reports")
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)

    df, y = load_student_math()
    results = run_comparison(df, y, cv_splits=cv_splits)
    table = comparison_table(results)

    summary = {
        "data_source": DATA_SOURCE,
        "task": "binary_classification_pass_fail_with_native_categoricals",
        "n_samples": int(len(y)),
        "cv_splits": cv_splits,
        "models": table,
    }
    (out / "summary.json").write_text(dumps_pretty(summary), encoding="utf-8")

    labels = [r["model"] for r in table]
    means = [r["roc_auc_mean"] for r in table]
    stds = [r["roc_auc_std"] for r in table]
    bar_with_errors(
        labels,
        means,
        stds,
        fig_dir / "roc_auc_encoding_comparison.png",
        title="UCI student math — encoding x model family (ROC-AUC)",
        ylabel="ROC-AUC",
    )

    for name, m in results.items():
        safe = name.replace("/", "_").replace(" ", "_")
        fold_line_plot(
            name,
            m["folds"],
            fig_dir / f"folds_{safe}.png",
            metric_name="ROC-AUC",
        )

    md = "\n".join(
        [
            "# Categorical encoding benchmark — statistics",
            "",
            f"**Data:** {DATA_SOURCE}",
            "",
            "## ROC-AUC (mean ± std over CV)",
            "",
            "| Model stack | Mean | Std |",
            "|---|---:|---:|",
        ]
    )
    for r in table:
        md += f"\n| {r['model']} | {r['roc_auc_mean']:.4f} | {r['roc_auc_std']:.4f} |"
    md += "\n\n## Figures\n\n- `figures/roc_auc_encoding_comparison.png`\n"
    (out / "REPORT.md").write_text(md, encoding="utf-8")
    return {"output_dir": str(out.resolve()), "n_models": len(table)}


def main() -> None:
    """Execute the main routine."""
    print(dumps_pretty(generate_report()))


if __name__ == "__main__":
    main()
