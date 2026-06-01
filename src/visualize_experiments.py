"""
Comprehensive visualization of P5 vs RL+Memory experiments.

Reads Phase 2/3/4 result JSONs + improved post-fix data from README,
generates 6 SVG figures for the experiment report.

Sources:
  - outputs/p5_eval_results.json          (Phase 2: P5 14-task eval)
  - outputs/rl_memory_results/comparison_results.json (Phase 3: P5 vs RL)
  - outputs/ablation_results.json         (Phase 4: 5-variant ablation)
  - README.md embedded tables             (improved post-cooc-fix results)
"""

import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load real data
with open(PROJECT_ROOT / "outputs" / "p5_eval_results.json") as f:
    p5_eval = json.load(f)

with open(PROJECT_ROOT / "outputs" / "rl_memory_results" / "comparison_results.json") as f:
    phase3 = json.load(f)

with open(PROJECT_ROOT / "outputs" / "ablation_results.json") as f:
    phase4 = json.load(f)

# ---------------------------------------------------------------------------
# Style config
# ---------------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

C = {
    "p5":       "#5B9BD5",
    "rl":       "#ED7D31",
    "ablation": ["#4472C4", "#ED7D31", "#A5A5A5", "#FFC000", "#5B9BD5", "#70AD47"],
    "exploit":  "#4472C4",
    "explore":  "#ED7D31",
    "memory":   "#70AD47",
    "diverse":  "#9B59B6",
    "grid":     "#D9D9D9",
}


def save(path_stem: str):
    p = str(OUTPUT_DIR / f"{path_stem}.svg")
    plt.savefig(p)
    print(f"  Saved {path_stem}.svg")
    plt.close()


# ===========================================================================
# Figure 1: P5 vs RL+Memory — HR & NDCG Comparison (Phase 3 improved)
# ===========================================================================
def fig1_p5_vs_rl_comparison():
    """Bar chart: P5 vs RL+Memory on HR@k and NDCG@k (post-cooc-fix data)."""
    # Improved data from README (post co-occurrence embeddings fix)
    metrics = ["HR@1", "HR@5", "HR@10", "NDCG@5", "NDCG@10"]
    p5_vals   = [0.000, 0.000, 0.000, 0.0000, 0.0000]
    rl_vals   = [0.002, 0.016, 0.024, 0.0090, 0.0117]
    # Old collapsed data (from JSON) for contrast
    p5_old    = [phase3["p5_baseline"]["HR@1"],
                 phase3["p5_baseline"]["HR@5"],
                 phase3["p5_baseline"]["HR@10"],
                 phase3["p5_baseline"]["NDCG@5"],
                 phase3["p5_baseline"]["NDCG@10"]]
    rl_old    = [phase3["rl_memory"]["HR@1"],
                 phase3["rl_memory"]["HR@5"],
                 phase3["rl_memory"]["HR@10"],
                 phase3["rl_memory"]["NDCG@5"],
                 phase3["rl_memory"]["NDCG@10"]]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # --- Left: Before fix (5% data, policy collapsed) ---
    x = np.arange(len(metrics))
    w = 0.30
    ax = axes[0]
    b1 = ax.bar(x - w/2, [v*100 for v in p5_old], w, color=C["p5"], label="P5 (Beam=20)")
    b2 = ax.bar(x + w/2, [v*100 for v in rl_old], w, color=C["rl"], label="RL+Memory", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Value (%)")
    ax.set_title("Before Fix (5% data, policy collapsed)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, color=C["grid"])
    # Annotate policy collapse
    ax.annotate("RL: 99.9%\nexploit_similar",
                xy=(2, rl_old[2]*100), xytext=(2, rl_old[2]*100 + 0.02),
                ha="center", fontsize=8, color=C["rl"],
                arrowprops=dict(arrowstyle="->", color=C["rl"], lw=0.8))

    # --- Right: After fix (co-occurrence embeddings) ---
    ax = axes[1]
    b3 = ax.bar(x - w/2, [v*100 for v in p5_vals], w, color=C["p5"],
                label="P5 (Beam=20)", hatch="//")
    b4 = ax.bar(x + w/2, [v*100 for v in rl_vals], w, color=C["rl"],
                label="RL+Memory", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylabel("Value (%)")
    ax.set_title("After Fix (co-occurrence embeddings)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, color=C["grid"])
    # Annotate
    for i, (pv, rv) in enumerate(zip(p5_vals, rl_vals)):
        if rv > 0:
            ax.text(x[i] + w/2, rv*100 + 0.001, f"{rv*100:.1f}%",
                    ha="center", va="bottom", fontsize=8, fontweight="bold", color=C["rl"])

    fig.suptitle("P5 vs RL+Memory: Before & After Co-occurrence Embeddings Fix",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    save("fig1_p5_vs_rl_comparison")


# ===========================================================================
# Figure 2: Ablation Experiment — HR & NDCG Side-by-Side
# ===========================================================================
def fig2_ablation_metrics():
    """Grouped bar chart: all 5 ablation variants vs P5 on HR/NDCG."""
    # Improved data from README Phase 4
    variants = ["P5\nBaseline", "Full\nSystem", "No\nMemory",
                "Short-term\nOnly", "No Time\nDecay", "Fixed\nε=0.1"]
    hr5  = [0.000, 0.016, 0.016, 0.016, 0.016, 0.016]
    hr10 = [0.000, 0.024, 0.024, 0.024, 0.024, 0.024]
    ndcg5  = [0.0000, 0.0090, 0.0090, 0.0090, 0.0090, 0.0090]
    ndcg10 = [0.0000, 0.0117, 0.0117, 0.0117, 0.0117, 0.0117]

    x = np.arange(len(variants))
    w = 0.18

    fig, ax = plt.subplots(figsize=(12, 5.5))

    ax.bar(x - 1.5*w, [v*100 for v in hr5],  w, color=C["ablation"][0], label="HR@5")
    ax.bar(x - 0.5*w, [v*100 for v in hr10], w, color=C["ablation"][1], label="HR@10")
    ax.bar(x + 0.5*w, [v*100 for v in ndcg5], w, color=C["ablation"][3], label="NDCG@5")
    ax.bar(x + 1.5*w, [v*100 for v in ndcg10],w, color=C["ablation"][5], label="NDCG@10")

    ax.set_xticks(x)
    ax.set_xticklabels(variants, fontsize=9)
    ax.set_ylabel("Value (%)")
    ax.set_title("Phase 4 Ablation: All RL Variants vs P5 Baseline (500 test users)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3, color=C["grid"])

    # Annotate HR@10 values
    for i, v in enumerate(hr10):
        if v > 0:
            ax.text(x[i] - 0.5*w, v*100 + 0.002, f"{v*100:.1f}%",
                    ha="center", fontsize=7, fontweight="bold", rotation=90, va="bottom")

    fig.tight_layout()
    save("fig2_ablation_metrics")


# ===========================================================================
# Figure 3: Action Distribution Comparison (Policy Collapse vs Healthy)
# ===========================================================================
def fig3_action_distribution():
    """Donut charts: action distribution before vs after fix, + ablation variant."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    # --- Left: Collapsed policy (Phase 3 old) ---
    ax = axes[0]
    old_actions = {"exploit_similar": 99.9, "explore_new_category": 0.1}
    colors = [C["exploit"], C["explore"]]
    wedges, texts, autotexts = ax.pie(
        list(old_actions.values()),
        labels=list(old_actions.keys()),
        autopct="%1.1f%%",
        colors=colors,
        startangle=90,
        pctdistance=0.6,
    )
    for at in autotexts:
        at.set_fontsize(7)
    ax.set_title("Phase 3 (Collapsed)\n99.9% exploit_similar", fontweight="bold")

    # --- Center: Healthy policy (Phase 3 improved) ---
    ax = axes[1]
    new_actions = {
        "serendipity": 45.2,
        "exploit_high_ctr": 30.0,
        "explore_new_brand": 18.0,
        "explore_trending_global": 5.4,
        "remind_abandoned": 1.4,
    }
    colors2 = [C["diverse"], C["exploit"], C["explore"], C["memory"], C["ablation"][2]]
    wedges, texts, autotexts = ax.pie(
        list(new_actions.values()),
        labels=list(new_actions.keys()),
        autopct="%1.1f%%",
        colors=colors2,
        startangle=90,
        pctdistance=0.6,
    )
    for at in autotexts:
        at.set_fontsize(6.5)
    ax.set_title("Phase 3 (Fixed)\nDiverse action distribution", fontweight="bold")

    # --- Right: Ablation action distributions (Phase 4 improved) ---
    ax = axes[2]
    ablation_actions = {
        "full_system":       {"remind_abandoned": 68.4, "explore_new_brand": 20.2, "explore_trending_global": 7.4, "exploit_high_ctr": 2.4, "serendipity": 1.6},
        "no_memory":         {"exploit_high_ctr": 38.4, "explore_new_brand": 27.8, "remind_abandoned": 14.2, "explore_trending_global": 10.4, "serendipity": 8.2},
        "short_term_only":   {"serendipity": 53.0, "explore_new_brand": 34.6, "remind_abandoned": 5.2, "explore_trending_global": 5.0, "exploit_high_ctr": 0.0},
        "no_time_decay":     {"explore_new_brand": 36.2, "serendipity": 32.0, "explore_trending_global": 15.4, "remind_abandoned": 9.0, "exploit_high_ctr": 7.4},
        "fixed_epsilon":     {"explore_trending_global": 37.4, "remind_abandoned": 30.4, "exploit_high_ctr": 25.6, "explore_new_brand": 4.8, "serendipity": 0.0},
    }

    variant_names_short = ["Full\nSystem", "No\nMemory", "Short-term\nOnly", "No Time\nDecay", "Fixed\nε=0.1"]
    # Top 3 actions to show
    action_names = ["remind_abandoned", "exploit_high_ctr", "explore_new_brand",
                    "explore_trending_global", "serendipity"]
    action_colors = {"remind_abandoned": C["memory"],
                     "exploit_high_ctr": C["exploit"],
                     "explore_new_brand": C["explore"],
                     "explore_trending_global": C["ablation"][4],
                     "serendipity": C["diverse"]}

    bar_data = {an: [] for an in action_names}
    for vn in ["full_system", "no_memory", "short_term_only", "no_time_decay", "fixed_epsilon"]:
        for an in action_names:
            bar_data[an].append(ablation_actions[vn].get(an, 0))

    x = np.arange(len(variant_names_short))
    w = 0.15
    for i, an in enumerate(action_names):
        offset = (i - 2) * w
        ax.bar(x + offset, bar_data[an], w, label=an, color=action_colors[an])

    ax.set_xticks(x)
    ax.set_xticklabels(variant_names_short, fontsize=8)
    ax.set_ylabel("Action Share (%)")
    ax.set_title("Phase 4 Ablation: Action Distributions\n(memory config drives strategy)", fontweight="bold")
    ax.legend(fontsize=6.5, ncol=2, loc="upper right")
    ax.grid(axis="y", alpha=0.3, color=C["grid"])

    fig.tight_layout()
    save("fig3_action_distribution")


# ===========================================================================
# Figure 4: P5 14-Task Evaluation Radar
# ===========================================================================
def fig4_p5_eval_radar():
    """Radar chart of P5 performance across all 14 evaluation tasks."""
    # Normalize each task type to [0,1] for radar
    tasks = []
    hr_vals = []
    ndcg_vals = []
    rmse_vals = []
    bleu_vals = []
    rouge_vals = []

    for task_name, metrics in p5_eval.items():
        tasks.append(task_name)
        if "HR@10" in metrics:
            hr_vals.append(metrics.get("HR@10", 0))
        else:
            hr_vals.append(0)

        if "NDCG@10" in metrics:
            ndcg_vals.append(metrics.get("NDCG@10", 0))
        else:
            ndcg_vals.append(0)

        if "RMSE" in metrics:
            rmse_vals.append(metrics.get("RMSE", 0))
        else:
            rmse_vals.append(0)

        if "BLEU-4" in metrics:
            bleu_vals.append(metrics.get("BLEU-4", 0))
        else:
            bleu_vals.append(0)

        if "ROUGE-L" in metrics:
            rouge_vals.append(metrics.get("ROUGE-L", 0))
        else:
            rouge_vals.append(0)

    # Normalize RMSE (lower is better, so invert)
    rmse_arr = np.array(rmse_vals)
    rmse_norm = 1.0 / (1.0 + rmse_arr / 100.0)  # squash large RMSE

    # Normalize others to [0,1]
    def norm(arr):
        a = np.array(arr)
        if a.max() > 0:
            return a / (a.max() + 1e-8)
        return a

    hr_norm = norm(hr_vals)
    ndcg_norm = norm(ndcg_vals)
    bleu_norm = norm(bleu_vals)
    rouge_norm = norm(rouge_vals)

    # Build category -> averaged metric
    categories = ["Rating", "Rating", "Sequential", "Sequential",
                  "Explanation", "Explanation", "Explanation",
                  "Review", "Review", "Review",
                  "Direct", "Direct", "Direct", "Direct"]
    # Manually map tasks to categories
    task_cat_map = {
        "rating_1-6": "Rating",
        "rating_1-10": "Rating",
        "seq_2-3": "Sequential",
        "seq_2-13": "Sequential",
        "exp_3-3": "Explanation",
        "exp_3-9": "Explanation",
        "exp_3-12": "Explanation",
        "review_summ_4-1": "Review",
        "review_rating_4-2": "Review",
        "review_rating_4-4": "Review",
        "direct_5-1": "Direct",
        "direct_5-4": "Direct",
        "direct_5-5": "Direct (Unseen)",
        "direct_5-8": "Direct (Unseen)",
    }

    # Use a grouped bar chart instead of radar (more readable for 14 tasks)
    task_labels = [task_cat_map[t] for t in tasks]
    short_names = ["R1-6", "R1-10", "S2-3", "S2-13", "E3-3", "E3-9", "E3-12",
                   "Rv4-1", "Rv4-2", "Rv4-4", "D5-1", "D5-4", "D5-5", "D5-8"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # Subplot 1: Sequential + Direct HR@10
    ax = axes[0, 0]
    seq_direct = ["seq_2-3", "seq_2-13", "direct_5-1", "direct_5-4", "direct_5-5", "direct_5-8"]
    sd_labels = ["Seq\n2-3", "Seq\n2-13", "Direct\n5-1", "Direct\n5-4", "Direct\n5-5\n(unseen)", "Direct\n5-8\n(unseen)"]
    sd_hr10 = [p5_eval[t].get("HR@10", 0)*100 for t in seq_direct]
    sd_ndcg10 = [p5_eval[t].get("NDCG@10", 0)*100 for t in seq_direct]
    x = np.arange(len(sd_labels))
    w = 0.30
    ax.bar(x - w/2, sd_hr10, w, color=C["p5"], label="HR@10")
    ax.bar(x + w/2, sd_ndcg10, w, color=C["rl"], label="NDCG@10")
    ax.set_xticks(x)
    ax.set_xticklabels(sd_labels, fontsize=8)
    ax.set_ylabel("Value (%)")
    ax.set_title("Sequential & Direct Recommendation")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, color=C["grid"])

    # Subplot 2: Rating tasks RMSE
    ax = axes[0, 1]
    rating_tasks = ["rating_1-6", "rating_1-10", "review_rating_4-2", "review_rating_4-4"]
    rat_labels = ["Rating\n1-6", "Rating\n1-10", "Review\nRating 4-2", "Review\nRating 4-4"]
    rat_rmse = [p5_eval[t].get("RMSE", 0) for t in rating_tasks]
    rat_mae = [p5_eval[t].get("MAE", 0) for t in rating_tasks]
    x = np.arange(len(rat_labels))
    w = 0.30
    b1 = ax.bar(x - w/2, rat_rmse, w, color=C["explore"], label="RMSE")
    b2 = ax.bar(x + w/2, rat_mae, w, color=C["exploit"], label="MAE")
    ax.set_xticks(x)
    ax.set_xticklabels(rat_labels, fontsize=8)
    ax.set_ylabel("Error")
    ax.set_title("Rating Prediction (lower is better)")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, color=C["grid"])
    # Note the scale issue
    ax.annotate("RMSE=1332 for 1-10\n(5% data, undertrained)",
                xy=(1, 1332), xytext=(1.5, 800),
                ha="center", fontsize=8, color=C["explore"],
                arrowprops=dict(arrowstyle="->", color=C["explore"], lw=0.8))

    # Subplot 3: Explanation + Review text generation
    ax = axes[1, 0]
    text_tasks = ["exp_3-3", "exp_3-9", "exp_3-12", "review_summ_4-1"]
    text_labels = ["Exp\n3-3", "Exp\n3-9", "Exp\n3-12", "Review\nSumm 4-1"]
    text_bleu = [p5_eval[t].get("BLEU-4", 0)*100 for t in text_tasks]
    text_rouge = [p5_eval[t].get("ROUGE-L", 0)*100 for t in text_tasks]
    x = np.arange(len(text_labels))
    w = 0.30
    ax.bar(x - w/2, text_bleu, w, color=C["ablation"][3], label="BLEU-4")
    ax.bar(x + w/2, text_rouge, w, color=C["ablation"][4], label="ROUGE-L")
    ax.set_xticks(x)
    ax.set_xticklabels(text_labels, fontsize=8)
    ax.set_ylabel("Value (%)")
    ax.set_title("Explanation & Review Generation")
    ax.legend()
    ax.grid(axis="y", alpha=0.3, color=C["grid"])

    # Subplot 4: Summary table / key takeaways
    ax = axes[1, 1]
    ax.axis("off")
    summary_text = (
        "P5 Evaluation Summary (14 tasks, 5% Beauty data, 10 epochs)\n"
        "──────────────────────────────────────────────\n"
        "Rating: RMSE 266 (1-6) / 1332 (1-10) — undertrained\n"
        "Sequential: HR@10 ≈ 0.5% — near random\n"
        "Explanation: BLEU-4 < 0.06 — weak generation\n"
        "Review: good ROUGE-L (1.44) on exp_3-12\n"
        "Direct (seen): HR@10 = 8.2% — strongest task\n"
        "Direct (unseen): HR@10 = 0.0% — total failure\n\n"
        "Key limitation: P5 trained only 10 epochs on 5%\n"
        "data. Full 100% data training needed for\n"
        "competitive results (RTX PRO 6000 96GB available)."
    )
    ax.text(0, 0.95, summary_text, transform=ax.transAxes,
            fontsize=9.5, fontfamily="monospace", va="top",
            bbox=dict(boxstyle="round,pad=0.8", facecolor="#F5F5F5", alpha=0.8))
    ax.set_title("P5 Baseline Analysis")

    fig.suptitle("Phase 2: P5 Baseline Evaluation — 14 Tasks Overview",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    save("fig4_p5_eval_dashboard")


# ===========================================================================
# Figure 5: Experiment Progress Timeline — Policy Collapse → Recovery
# ===========================================================================
def fig5_progress_timeline():
    """Visual timeline showing metric improvements across experiment phases."""
    phases = ["Phase 1\nP5 Train\n(5% data)",
              "Phase 2\nP5 Eval\n(14 tasks)",
              "Phase 3\nRL v1\n(collapsed)",
              "Phase 3\nRL v2\n(cooc-fix)",
              "Phase 4\nAblation\n(fixed)",
              "E-E Sim\nOnline RL\n(this work)"]

    hr10_vals = [None, 0.0052, 0.0016, 0.024, 0.024, None]
    ndcg10_vals = [None, 0.0023, 0.0005, 0.0117, 0.0117, None]
    num_actions = [None, None, 3, 5, 14, 14]
    action_entropy = [None, None, 0.02, 1.75, 2.2, 2.14]

    x = np.arange(len(phases))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    # Left: HR@10 / NDCG@10 progression
    ax = axes[0]
    valid_hr = [(i, v) for i, v in enumerate(hr10_vals) if v is not None]
    valid_nd = [(i, v) for i, v in enumerate(ndcg10_vals) if v is not None]

    ax.plot([i for i, _ in valid_hr], [v*100 for _, v in valid_hr],
            "o-", color=C["rl"], linewidth=2, markersize=10, label="HR@10")
    ax.plot([i for i, _ in valid_nd], [v*100 for _, v in valid_nd],
            "s-", color=C["p5"], linewidth=2, markersize=10, label="NDCG@10")

    # Highlight policy collapse and recovery
    ax.axvspan(2.3, 3.3, alpha=0.12, color=C["explore"], label="Cooc-Fix")
    ax.annotate("+15×\nimprovement",
                xy=(3, hr10_vals[3]*100), xytext=(3.8, hr10_vals[3]*100 + 0.5),
                fontsize=9, color=C["rl"], fontweight="bold",
                arrowprops=dict(arrowstyle="->", color=C["rl"], lw=1.2))

    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels(phases, fontsize=8)
    ax.set_ylabel("Value (%)")
    ax.set_title("HR@10 & NDCG@10 Progression")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.3, color=C["grid"])

    # Right: Action diversity
    ax = axes[1]
    valid_act = [(i, v) for i, v in enumerate(num_actions) if v is not None]
    valid_ent = [(i, v) for i, v in enumerate(action_entropy) if v is not None]

    ax2 = ax.twinx()
    bars = ax.bar([i for i, _ in valid_act], [v for _, v in valid_act],
                  color=C["exploit"], alpha=0.5, label="# Actions Used", width=0.4)
    line = ax2.plot([i for i, _ in valid_ent], [v for _, v in valid_ent],
                    "D-", color=C["diverse"], linewidth=2, markersize=10, label="Action Entropy")

    ax.axhline(y=16, color=C["grid"], linestyle="--", linewidth=0.8, alpha=0.5)
    ax.annotate("Total actions = 16", xy=(5, 16), ha="right", fontsize=7, color="gray")

    ax.set_xticks(range(len(phases)))
    ax.set_xticklabels(phases, fontsize=8)
    ax.set_ylabel("# Actions", color=C["exploit"])
    ax2.set_ylabel("Action Entropy", color=C["diverse"])
    ax2.set_ylim(0, 3.0)
    ax.set_title("Policy Action Diversity Progression")
    # Merge legends
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
    ax.grid(axis="y", alpha=0.3, color=C["grid"])

    fig.suptitle("Experiment Progress: Policy Collapse → Recovery → Analysis",
                 fontweight="bold", y=1.01)
    fig.tight_layout()
    save("fig5_progress_timeline")


# ===========================================================================
# Figure 6: P5 14-Task Full Heatmap
# ===========================================================================
def fig6_p5_task_heatmap():
    """Heatmap showing P5 performance across all 14 tasks with normalized scores."""
    tasks = list(p5_eval.keys())
    task_short = ["R1-6", "R1-10", "S2-3", "S2-13", "E3-3", "E3-9", "E3-12",
                  "Rv4-1", "Rv4-2", "Rv4-4", "D5-1", "D5-4", "D5-5", "D5-8"]

    # All metric keys
    all_metrics = ["HR@1", "HR@5", "HR@10", "NDCG@5", "NDCG@10",
                   "RMSE", "MAE", "BLEU-4", "ROUGE-1", "ROUGE-2", "ROUGE-L"]

    matrix = np.zeros((len(all_metrics), len(tasks)))
    mask = np.zeros((len(all_metrics), len(tasks)), dtype=bool)

    for j, task in enumerate(tasks):
        data = p5_eval[task]
        for i, metric in enumerate(all_metrics):
            if metric in data:
                matrix[i, j] = data[metric]
                mask[i, j] = True

    # Normalize each row
    for i in range(len(all_metrics)):
        row = matrix[i]
        if row[mask[i]].max() > 0:
            max_v = row[mask[i]].max()
            if max_v > 0:
                row[mask[i]] = row[mask[i]] / max_v

    fig, ax = plt.subplots(figsize=(14, 5.5))

    # Use masked array for display
    matrix_display = np.where(mask, matrix, np.nan)
    im = ax.imshow(matrix_display, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(task_short, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(all_metrics)))
    ax.set_yticklabels(all_metrics, fontsize=8)

    # Add text annotations
    for i in range(len(all_metrics)):
        for j in range(len(tasks)):
            if mask[i, j]:
                val = matrix[i, j]
                if val > 0.01:
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6)
                else:
                    ax.text(j, i, "0", ha="center", va="center", fontsize=6, color="gray")

    ax.set_title("P5 Baseline: 14-Task Performance Heatmap (normalized per metric)")
    plt.colorbar(im, ax=ax, label="Normalized Score (1.0 = best within metric)")

    fig.tight_layout()
    save("fig6_p5_task_heatmap")


# ===========================================================================
# Figure 7: Ablation Component Analysis — Deep Dive
# ===========================================================================
def fig7_ablation_deep_dive():
    """Comprehensive ablation figure: component matrix + action strategy divergence."""
    fig = plt.figure(figsize=(16, 8))

    # --- Top: Component configuration matrix ---
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.3], hspace=0.35, wspace=0.35)

    # Subplot A: Component enable/disable matrix
    ax = fig.add_subplot(gs[0, 0])
    variants = ["Full System", "No Memory", "Short-term\nOnly",
                "No Time\nDecay", "Fixed\nε=0.1"]
    components = ["Memory\nSystem", "Long-term\n(FAISS)", "Time\nDecay", "Adaptive\nε"]
    config_matrix = np.array([
        [1, 1, 1, 1],   # full_system: all enabled
        [0, 0, 0, 1],   # no_memory: all mem disabled
        [1, 0, 0, 1],   # short_term_only: only ST buffer
        [1, 1, 0, 1],   # no_time_decay: no decay
        [1, 1, 1, 0],   # fixed_epsilon: no adaptive
    ])

    ax.imshow(config_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)
    for i in range(len(variants)):
        for j in range(len(components)):
            val = config_matrix[i, j]
            ax.text(j, i, "✓" if val else "✗",
                    ha="center", va="center", fontsize=14,
                    color="black" if val else "#888",
                    fontweight="bold" if val else "normal")
    ax.set_xticks(range(len(components)))
    ax.set_xticklabels(components, fontsize=9)
    ax.set_yticks(range(len(variants)))
    ax.set_yticklabels(variants, fontsize=9)
    ax.set_title("Component Configuration Matrix\n(✓ = enabled, ✗ = disabled)", fontweight="bold")

    # Subplot B: Key insight — same metrics, different strategies
    ax = fig.add_subplot(gs[0, 1])
    ax.axis("off")
    insight_text = (
        "Ablation Key Finding\n"
        "════════════════════\n\n"
        "All 5 variants produce IDENTICAL\n"
        "recommendation metrics:\n"
        "  HR@5  = 1.6%\n"
        "  HR@10 = 2.4%\n"
        "  NDCG@10 = 1.17%\n\n"
        "But the POLICY learns DIFFERENT\n"
        "action strategies for each memory\n"
        "configuration:\n\n"
        "  Full System → 68% remind_abandoned\n"
        "  No Memory   → 38% exploit_high_ctr\n"
        "  Short-term  → 53% serendipity\n"
        "  No Decay    → 36% explore_new_brand\n"
        "  Fixed ε     → 37% explore_trending\n\n"
        "➜ POMDP Policy adapts to memory\n"
        "   context availability.\n"
        "➜ Retriever is the bottleneck —\n"
        "   all strategies return same\n"
        "   candidates via similarity rank."
    )
    ax.text(0, 0.95, insight_text, transform=ax.transAxes,
            fontsize=8.5, fontfamily="monospace", va="top",
            bbox=dict(boxstyle="round,pad=1.0", facecolor="#FFF8E1",
                      edgecolor=C["ablation"][3], alpha=0.9))

    # Subplot C: Action strategy divergence (radar-like parallel coordinates)
    ax = fig.add_subplot(gs[1, :])

    ablation_actions = {
        "Full System":      {"remind_abandoned": 68.4, "explore_new_brand": 20.2, "explore_trending_global": 7.4, "exploit_high_ctr": 2.4, "serendipity": 1.6},
        "No Memory":        {"exploit_high_ctr": 38.4, "explore_new_brand": 27.8, "remind_abandoned": 14.2, "explore_trending_global": 10.4, "serendipity": 8.2},
        "Short-term Only":  {"serendipity": 53.0, "explore_new_brand": 34.6, "remind_abandoned": 5.2, "explore_trending_global": 5.0, "exploit_high_ctr": 0.0},
        "No Time Decay":    {"explore_new_brand": 36.2, "serendipity": 32.0, "explore_trending_global": 15.4, "remind_abandoned": 9.0, "exploit_high_ctr": 7.4},
        "Fixed ε=0.1":     {"explore_trending_global": 37.4, "remind_abandoned": 30.4, "exploit_high_ctr": 25.6, "explore_new_brand": 4.8, "serendipity": 0.0},
    }

    action_names = ["remind_abandoned", "exploit_high_ctr", "explore_new_brand",
                    "explore_trending_global", "serendipity"]
    action_labels = ["Remind\nAbandoned", "Exploit\nHigh CTR", "Explore\nNew Brand",
                     "Explore\nTrending", "Serendipity"]
    action_colors_map = {
        "remind_abandoned": C["memory"],
        "exploit_high_ctr": C["exploit"],
        "explore_new_brand": C["explore"],
        "explore_trending_global": C["ablation"][4],
        "serendipity": C["diverse"],
    }

    x = np.arange(len(action_names))
    n_variants = len(ablation_actions)
    w = 0.15
    variant_names = list(ablation_actions.keys())

    for i, (vname, act_dist) in enumerate(ablation_actions.items()):
        vals = [act_dist.get(a, 0) for a in action_names]
        offset = (i - (n_variants - 1) / 2) * w
        bars = ax.bar(x + offset, vals, w, label=vname,
                      color=C["ablation"][i], alpha=0.85, edgecolor="white", linewidth=0.5)
        # Mark dominant action
        max_idx = np.argmax(vals)
        ax.text(x[max_idx] + offset, vals[max_idx] + 1.5, f"{vals[max_idx]:.0f}%",
                ha="center", fontsize=7, fontweight="bold",
                color=C["ablation"][i])

    ax.set_xticks(x)
    ax.set_xticklabels(action_labels, fontsize=9)
    ax.set_ylabel("Action Share (%)")
    ax.set_title("Ablation Action Distribution: Same Metrics, Different Strategies",
                 fontweight="bold")
    ax.legend(fontsize=8, ncol=5, loc="upper right")
    ax.grid(axis="y", alpha=0.3, color=C["grid"])
    ax.set_ylim(0, 80)

    # Add annotation: metrics are identical
    ax.annotate("All variants: HR@10=2.4%\nNDCG@10=1.17%",
                xy=(0.5, 0.92), xycoords="axes fraction",
                ha="center", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#E8F5E9",
                          edgecolor= C["ablation"][5], alpha=0.9))

    save("fig7_ablation_deep_dive")


# ===========================================================================
# Main
# ===========================================================================
def main():
    print(f"Generating figures to {OUTPUT_DIR}/ ...")
    fig1_p5_vs_rl_comparison()
    fig2_ablation_metrics()
    fig3_action_distribution()
    fig4_p5_eval_radar()
    fig5_progress_timeline()
    fig6_p5_task_heatmap()
    fig7_ablation_deep_dive()
    print(f"\nDone. {len(os.listdir(OUTPUT_DIR))} SVGs generated in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
