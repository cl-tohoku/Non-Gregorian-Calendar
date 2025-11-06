import argparse
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_error_rates(models, model_name_map, eras, setting, base_dir):
    error_rates = {}
    for model in models:
        lang = "ja" if "Swallow" in model or "sarashina" in model or "llm-jp" in model else "en"
        model_display = model_name_map.get(model, model) + f" ({lang})"

        summary_path = Path(base_dir) / model / f"{setting}10" / f"{lang}-prompt" / "results_summary.json"
        error_path = Path(base_dir) / model / f"{setting}10" / f"{lang}-prompt" / "error_add10.json"

        try:
            with open(summary_path) as f:
                summary = json.load(f)
            with open(error_path) as f:
                error_data = json.load(f)

            per_prompt = error_data["per_prompt"]
            model_era_errors = {}

            for era in eras:
                prompt_id = summary["best_prompt_per_metric"][era]["year_match"][0]
                era_error_rate = per_prompt[str(prompt_id)]["per_era"][era]["error_rate"]
                model_era_errors[era] = era_error_rate

            error_rates[model_display] = model_era_errors

        except Exception as e:
            print(f"[Error] {model_display}: {e}")

    return error_rates

def plot_heatmap(error_rates, eras, output_dir, font_size,
                 annot_font_size=None, cbar_font_size=None, cbar_label_font_size=None):
    df = pd.DataFrame(error_rates).T[eras]

    if annot_font_size is None:
        annot_font_size = font_size
    if cbar_font_size is None:
        cbar_font_size = font_size
    if cbar_label_font_size is None:
        cbar_label_font_size = font_size

    plt.figure(figsize=(30, 5))
    ax = sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0, vmax=1,
        annot_kws={"size": annot_font_size},
        cbar_kws={"label": "Out-of-range Error"}
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=cbar_font_size)
    cbar.set_label("Out-of-range Error", fontsize=cbar_label_font_size)

    plt.ylabel("Model", fontsize=font_size)
    plt.xlabel("Era Transition", fontsize=font_size)

    ax.set_xticklabels([era.capitalize() for era in eras], fontsize=font_size)

    plt.yticks(fontsize=font_size)

    output_path_png = Path(output_dir) / "add10_error_heatmap.png"
    output_path_pdf = Path(output_dir) / "add10_error_heatmap.pdf"
    plt.tight_layout()
    plt.savefig(output_path_png, dpi=300)
    plt.savefig(output_path_pdf)
    print(f"Heatmap saved to:\n - {output_path_png}\n - {output_path_pdf}")
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Generate heatmap of +10 year error rates in Japanese calendar task.")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Base directory containing model results (e.g. /home/.../Japanese_calendar_arithmetic/)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the output heatmap")
    parser.add_argument("--setting", type=str, default="after",
                        help="Setting name (e.g., 'after')")
    parser.add_argument("--font_size", type=int, default=40,
                        help="Font size for heatmap labels and title")
    parser.add_argument("--models", nargs="+", required=True,
                        help="List of model names to include (e.g., llm-jp-3-13b sarashina2-13b Llama-2-7b-hf)")
    args = parser.parse_args()

    model_name_map = {
        "llm-jp-3-13b": "LLM-JP-3-13B",
        "sarashina2-13b": "Sarashina2-13B",
        "Swallow-13b-hf": "Swallow-13B",
        "Swallow-MS-7b-v0.1": "Swallow-MS-7B",
        "Llama-3-Swallow-8B-v0.1": "LLaMA3-Swallow-8B",
        "Llama-2-7b-hf": "LLaMA2-7B",
        "Llama-2-13b-hf": "LLaMA2-13B",
        "Mistral-7B-v0.1": "Mistral-7B",
        "Llama-3.1-8B": "LLaMA3.1-8B"
    }

    eras = ["meiji", "taisho", "showa", "heisei"]

    os.makedirs(args.output_dir, exist_ok=True)
    error_rates = load_error_rates(args.models, model_name_map, eras, args.setting, args.base_dir)
    plot_heatmap(error_rates, eras, args.output_dir, args.font_size)


if __name__ == "__main__":
    main()
