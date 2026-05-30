"""
P5 Reproducibility Report Generator.

Compares reproduced evaluation results against paper benchmark values
(P5 RecSys 2022, Tables 2-7) and generates a formatted report.

Usage:
    python generate_report.py --results eval_results_beauty_BEST_EVAL_LOSS.json \
                              --dataset beauty --backbone t5-small
"""
import json
import argparse
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, Tuple


# ── Paper Benchmark Values ────────────────────────────────
# P5-Small on Amazon Beauty (from P5 RecSys 2022 Tables 2-7)
PAPER_BENCHMARKS = {
    # Table 2: Rating Prediction
    "rating_1-6": {"RMSE": 1.3128, "MAE": 0.8428},
    "rating_1-10": {"RMSE": 1.2989, "MAE": 0.8473},
    # Table 3: Sequential Recommendation
    "seq_2-3": {"HR@1": 0.0307, "HR@5": 0.0503, "HR@10": 0.0659,
                "NDCG@5": 0.0370, "NDCG@10": 0.0421},
    "seq_2-13": {"HR@1": 0.0313, "HR@5": 0.0490, "HR@10": 0.0646,
                 "NDCG@5": 0.0358, "NDCG@10": 0.0409},
    # Table 4: Explanation Generation
    "exp_3-3": {"BLEU-4": 1.2237, "ROUGE-1": 17.6938, "ROUGE-2": 4.2313, "ROUGE-L": 12.8606},
    "exp_3-9": {"BLEU-4": 1.9788, "ROUGE-1": 25.6253, "ROUGE-2": 7.1652, "ROUGE-L": 17.1536},
    "exp_3-12": {"BLEU-4": 1.9425, "ROUGE-1": 25.1474, "ROUGE-2": 7.0301, "ROUGE-L": 16.8232},
    # Table 5: Review Rating Prediction
    "review_rating_4-2": {"RMSE": 0.6233, "MAE": 0.3051},
    "review_rating_4-4": {"RMSE": None, "MAE": None},  # 4-4 not in table for Beauty
    # Table 6: Review Summarization
    "review_summ_4-1": {"BLEU-4": None, "BLEU-2": 2.1225, "ROUGE-1": 8.4205, "ROUGE-L": 7.8426},
    # Table 7: Direct Recommendation
    "direct_5-1": {"HR@1": 0.0513, "HR@5": 0.1446, "HR@10": 0.2406,
                   "NDCG@5": 0.0975, "NDCG@10": 0.1289},
    "direct_5-4": {"HR@1": 0.0862, "HR@5": 0.2448, "HR@10": 0.3441,
                   "NDCG@5": 0.1673, "NDCG@10": 0.1993},
    "direct_5-5": {"HR@1": 0.0601, "HR@5": 0.1611, "HR@10": 0.2370,
                   "NDCG@5": 0.1117, "NDCG@10": 0.1367},
    "direct_5-8": {"HR@1": 0.0571, "HR@5": 0.1566, "HR@10": 0.2317,
                   "NDCG@5": 0.1078, "NDCG@10": 0.1326},
}


def load_results(path: str) -> Dict:
    with open(path, 'r') as f:
        return json.load(f)


def compare_metric(key: str, metric: str, reproduced: Optional[float],
                   paper: Optional[float]) -> Tuple[str, float]:
    """Compare one metric and return (status, deviation_pct)."""
    if reproduced is None or paper is None or paper == 0:
        return ("N/A", 0.0)

    deviation = abs(reproduced - paper) / abs(paper) * 100
    if deviation <= 2.0:
        status = "PASS"
    elif deviation <= 5.0:
        status = "WARN"
    else:
        status = "FAIL"
    return (status, deviation)


def generate_markdown_report(results: Dict, dataset: str, backbone: str,
                             checkpoint: str, output_path: str = None) -> str:
    """Generate a full markdown reproducibility report."""
    lines = []
    lines.append(f"# P5 Reproducibility Report")
    lines.append(f"")
    lines.append(f"**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Dataset**: {dataset}")
    lines.append(f"**Backbone**: {backbone}")
    lines.append(f"**Checkpoint**: {checkpoint}")
    lines.append(f"")
    lines.append(f"---")
    lines.append(f"")

    # Summary stats
    total_metrics = 0
    pass_metrics = 0
    warn_metrics = 0
    fail_metrics = 0
    deviation_sum = 0.0

    def add_task_section(title, table_ref, task_keys):
        nonlocal total_metrics, pass_metrics, warn_metrics, fail_metrics, deviation_sum

        lines.append(f"## {title}")
        lines.append(f"")
        lines.append(f"*Paper reference: {table_ref}*")
        lines.append(f"")

        for tk in task_keys:
            reproduced = results.get(tk, {})
            paper = PAPER_BENCHMARKS.get(tk, {})
            if not paper:
                continue

            lines.append(f"### {tk}")
            lines.append(f"")
            lines.append(f"| Metric | Paper | Reproduced | Deviation | Status |")
            lines.append(f"|--------|-------|------------|-----------|--------|")

            for metric, paper_val in paper.items():
                if paper_val is None:
                    continue
                repro_val = reproduced.get(metric)
                status, dev = compare_metric(tk, metric, repro_val, paper_val)

                total_metrics += 1
                if status == "PASS":
                    pass_metrics += 1
                elif status == "WARN":
                    warn_metrics += 1
                elif status == "FAIL":
                    fail_metrics += 1
                deviation_sum += dev

                paper_str = f"{paper_val:.4f}"
                repro_str = f"{repro_val:.4f}" if repro_val is not None else "N/A"
                status_emoji = {"PASS": "✓", "WARN": "⚠", "FAIL": "✗", "N/A": "-"}[status]
                lines.append(f"| {metric} | {paper_str} | {repro_str} | {dev:.1f}% | {status_emoji} {status} |")

            lines.append(f"")

    # Table 2: Rating
    add_task_section("1. Rating Prediction", "Table 2",
                     ["rating_1-6", "rating_1-10"])

    # Table 3: Sequential
    add_task_section("2. Sequential Recommendation", "Table 3",
                     ["seq_2-3", "seq_2-13"])

    # Table 4: Explanation
    add_task_section("3. Explanation Generation", "Table 4",
                     ["exp_3-3", "exp_3-9", "exp_3-12"])

    # Tables 5 & 6: Review
    add_task_section("4. Review Related", "Tables 5 & 6",
                     ["review_rating_4-2", "review_summ_4-1"])

    # Table 7: Direct
    add_task_section("5. Direct Recommendation", "Table 7",
                     ["direct_5-1", "direct_5-4", "direct_5-5", "direct_5-8"])

    # Overall summary
    lines.append(f"---")
    lines.append(f"")
    lines.append(f"## Overall Summary")
    lines.append(f"")
    pass_rate = pass_metrics / total_metrics * 100 if total_metrics > 0 else 0
    avg_dev = deviation_sum / total_metrics if total_metrics > 0 else 0

    lines.append(f"| | Count | Percentage |")
    lines.append(f"|---|-------|------------|")
    lines.append(f"| **PASS** (±2%) | {pass_metrics} | {pass_metrics/total_metrics*100:.1f}% |")
    lines.append(f"| **WARN** (±5%) | {warn_metrics} | {warn_metrics/total_metrics*100:.1f}% |")
    lines.append(f"| **FAIL** (>5%) | {fail_metrics} | {fail_metrics/total_metrics*100:.1f}% |")
    lines.append(f"| **Total** | {total_metrics} | 100% |")
    lines.append(f"")
    lines.append(f"**Average deviation**: {avg_dev:.2f}%")
    lines.append(f"")

    # Acceptance criteria
    criteria_pass = pass_metrics >= 25  # ≥25 out of 30 metrics within 2% (DOE.md Section 1.8)
    lines.append(f"## Acceptance Criteria (DOE Section 1.8)")
    lines.append(f"")
    lines.append(f"- **Primary**: ≥25/30 metrics within ±2% deviation → **{'PASSED' if criteria_pass else 'NOT MET'}** ({pass_metrics}/30 within ±2%)")
    lines.append(f"- **Secondary**: All metrics show same trend direction as paper → Check manually")
    lines.append(f"- **Overall verdict**: {'✅ REPRODUCTION SUCCESSFUL' if criteria_pass and avg_dev < 3.0 else '⚠ NEEDS INVESTIGATION'}")
    lines.append(f"")

    lines.append(f"---")
    lines.append(f"*Report generated by reproduce/generate_report.py*")

    report = "\n".join(lines)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"Report saved to: {output_path}")

    return report


def main():
    p = argparse.ArgumentParser(description="Generate P5 reproducibility report")
    p.add_argument('--results', type=str, required=True,
                   help='Path to eval results JSON')
    p.add_argument('--dataset', type=str, default='beauty')
    p.add_argument('--backbone', type=str, default='t5-small')
    p.add_argument('--checkpoint', type=str, default='',
                   help='Checkpoint name for report header')
    p.add_argument('--output', type=str, default=None,
                   help='Output markdown file (default: auto-generated)')
    args = p.parse_args()

    results = load_results(args.results)

    if not args.output:
        name = Path(args.results).stem
        args.output = f"report_{name}.md"

    report = generate_markdown_report(
        results, args.dataset, args.backbone,
        args.checkpoint or args.results, args.output
    )

    print("\n" + report)


if __name__ == '__main__':
    main()
