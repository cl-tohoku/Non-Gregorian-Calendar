import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_corpus_freq(file_path):
    counts = []
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            data = json.loads(line)
            counts.append(data["count_w"])
    return sum(counts) / len(counts) if counts else 0


def main():
    parser = argparse.ArgumentParser(description="Plot corpus frequency vs full match rate per era transition.")
    parser.add_argument("--corpus_dir", type=str, required=True,
                        help="Directory path for corpus frequency JSONL files (e.g. /home/.../infini-gram/)")
    parser.add_argument("--results_file", type=str, required=True,
                        help="Path to results_summary.json file")
    parser.add_argument("--output", type=str, default="era_correlation",
                        help="Output file or directory (e.g., /path/to/output or /path/to/output/)")
    parser.add_argument("--font_size", type=int, default=12,
                        help="Base font size for labels, ticks, and title")
    args = parser.parse_args()

    plt.rcParams.update({
        "font.size": args.font_size,
        "axes.titlesize": args.font_size + 2,
        "axes.labelsize": args.font_size + 2,
        "xtick.labelsize": args.font_size,
        "ytick.labelsize": args.font_size,
        "legend.fontsize": args.font_size - 1
    })

    eras = ["meiji", "taisho", "showa", "heisei", "reiwa"]
    era_pairs = [("meiji", "taisho"), ("taisho", "showa"), ("showa", "heisei"), ("heisei", "reiwa")]

    corpus_freq = {}
    for era in eras:
        file_path = Path(args.corpus_dir) / f"{era}.jsonl"
        if file_path.exists():
            corpus_freq[era] = load_corpus_freq(file_path)
        else:
            corpus_freq[era] = None
            print(f"⚠️ Warning: {file_path} not found")

    with open(args.results_file, encoding="utf-8") as f:
        results = json.load(f)

    best_prompt = results["best_prompt_per_metric"]
    year_match = {era: best_prompt[era]["year_match"][1] for era in best_prompt}

    data = []
    for pre, post in era_pairs:
        data.append({
            "pair": f"{pre.capitalize()}→{post.capitalize()}",
            "freq_pre": corpus_freq.get(pre),
            "freq_post": corpus_freq.get(post),
            "year_match": year_match.get(pre, 0)
        })
    df = pd.DataFrame(data)

    output_path = Path(args.output)
    if output_path.suffix == "": 
        output_path.mkdir(parents=True, exist_ok=True)
        full_png_path = output_path / "era_correlation.png"
        full_pdf_path = output_path / "era_correlation.pdf"
        post_png_path = output_path / "era_correlation_post_only.png"
        post_pdf_path = output_path / "era_correlation_post_only.pdf"
    else:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        full_png_path = output_path.with_suffix(".png")
        full_pdf_path = output_path.with_suffix(".pdf")
        post_png_path = output_path.with_name(output_path.stem + "_post_only.png")
        post_pdf_path = output_path.with_name(output_path.stem + "_post_only.pdf")

    fig, ax1 = plt.subplots(figsize=(8, 5))
    width = 0.4
    x = range(len(df))

    ax1.bar([i - width/2 for i in x], df["freq_pre"], color="limegreen", width=width, label="Pre-era freq")
    ax1.bar([i + width/2 for i in x], df["freq_post"],color="skyblue", width=width, label="Post-era freq")
    ax1.set_ylabel("Average corpus freq")
    ax1.set_xlabel("Era transition")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["pair"])

    ax2 = ax1.twinx()
    ax2.plot(x, df["year_match"], marker="o", color="Red", label="Acc (Ja)")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(full_png_path, dpi=300)
    plt.savefig(full_pdf_path)
    print(f"Saved full figure:\n - {full_png_path}\n - {full_pdf_path}")
    plt.close(fig)

    fig, ax1 = plt.subplots(figsize=(6, 5))
    width = 0.5
    ax1.bar(x, df["freq_post"], width=width, color="skyblue", label="Corpus freq (J)")
    ax1.set_ylabel("Average corpus freq (Japanese Calendar)")
    ax1.set_xlabel("Era transition")
    ax1.set_xticks(x)
    ax1.set_xticklabels(df["pair"])

    ax2 = ax1.twinx()
    ax2.plot(x, df["year_match"], marker="o", color="lightcoral", label="Full Match (J)")
    ax2.set_ylabel("Full Match Rate (Japanese Calendar)")
    ax2.set_ylim(0, 1)

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left", frameon=True)

    plt.tight_layout()
    plt.savefig(post_png_path, dpi=300)
    plt.savefig(post_pdf_path)
    print(f"Saved post-era only figure:\n - {post_png_path}\n - {post_pdf_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()