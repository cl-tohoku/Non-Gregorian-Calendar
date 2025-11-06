import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_avg_counts(file_path):
    count_g_list, count_w_list = [], []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            count_g_list.append(data["count_g"])
            count_w_list.append(data["count_w"])
    avg_g = sum(count_g_list) / len(count_g_list) if count_g_list else 0
    avg_w = sum(count_w_list) / len(count_w_list) if count_w_list else 0
    ratio = avg_w / avg_g if avg_g != 0 else 0
    return avg_g, avg_w, ratio


def load_full_match_rate(result_file):
    with open(result_file, encoding="utf-8") as f:
        data = json.load(f)
    return data["average_scores"]["full_match_rate"]


def main():
    parser = argparse.ArgumentParser(description="Plot corpus frequency ratio vs Full Match Rate per era.")
    parser.add_argument("--corpus_dir", type=str, required=True,
                        help="Directory containing {era}.jsonl corpus frequency files.")
    parser.add_argument("--results_dir", type=str, required=True,
                        help="Directory containing summary_prompt_{era}.json files.")
    parser.add_argument("--output", type=str, default="era_ratio_plot",
                        help="Base filename for output (without extension).")
    parser.add_argument("--font_size", type=int, default=12,
                        help="Font size for labels and ticks.")
    args = parser.parse_args()

    plt.rcParams.update({
        "font.size": args.font_size,
        "axes.titlesize": args.font_size + 2,
        "axes.labelsize": args.font_size + 2,
        "xtick.labelsize": args.font_size,
        "ytick.labelsize": args.font_size,
        "legend.fontsize": args.font_size - 1
    })

    eras = ["meiji", "taisho", "showa", "heisei"]
    corpus_dir = Path(args.corpus_dir)
    results_dir = Path(args.results_dir)

    # --- データ収集 ---
    data = []
    for era in eras:
        corpus_path = corpus_dir / f"{era}.jsonl"
        result_path = results_dir / f"summary_prompt_{era}.json"

        if not corpus_path.exists():
            print(f"Warning: missing corpus file: {corpus_path}")
            continue
        if not result_path.exists():
            print(f"Warning: missing result file: {result_path}")
            continue

        avg_g, avg_w, ratio = load_avg_counts(corpus_path)
        fmr = load_full_match_rate(result_path)
        data.append({
            "era": era.capitalize(),
            "avg_g": avg_g,
            "avg_w": avg_w,
            "ratio": ratio,
            "full_match_rate": fmr
        })

    df = pd.DataFrame(data)
    if df.empty:
        print("No valid data found.")
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))
    width = 0.6

    x = range(len(df))
    ax1.bar(x, df["ratio"], color="skyblue", width=width, label="Corpus freq ratio (Ja/Gre)")
    ax1.set_ylabel("Corpus frequency ratio")
    ax1.set_xlabel("Era")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["era"])
    ax1.set_ylim(0, 0.35)

    ax2 = ax1.twinx()
    ax2.plot(x, df["full_match_rate"], marker="o", color="Red", label="Acc (Ja)")
    ax2.set_ylabel("Full Match Rate")

    # max_fmr = df["full_match_rate"].max()
    # min_fmr = df["full_match_rate"].min()
    ax2.set_ylim(0, 0.25)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper right", frameon=True)

    plt.tight_layout()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    png_path = output_dir / "era_ratio_plot.png"
    pdf_path = output_dir / "era_ratio_plot.pdf"

    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    print(f"Saved:\n - {png_path}\n - {pdf_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
