import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

DEFAULT_MODEL_NAME_MAP = {
    "llm-jp-3-13b": "LLM-JP-3-13B",
    "sarashina2-13b": "Sarashina2-13B",
    "Swallow-13b-hf": "Swallow-13B",
    "Swallow-MS-7b-v0.1": "Swallow-MS-7B",
    "Llama-3-Swallow-8B-v0.1": "LLaMA3-Swallow-8B",
}

ERAS = ["meiji", "taisho", "showa", "heisei"]
ERA_LABELS = [e.capitalize() for e in ERAS]
METRIC_KEY = "Year Format Match"


def collect_year_match(base_dir, models, model_name_map, setting="japrompt", name="ja-ja-name"):
    results = {}

    for model in models:
        model_dir = Path(base_dir) / model / "birthyear_wareki" / setting / name
        summary_path = model_dir / "summary_all.json"
        model_display = model_name_map.get(model, model) + " (ja)"

        if not summary_path.exists():
            print(f"Missing summary_all.json for {model}")
            continue

        try:
            era_values = {}
            for era in ERAS:
                era_path = model_dir / f"summary_{era}.json"
                if not era_path.exists():
                    print(f"Missing: {era_path}")
                    continue

                with open(era_path, encoding="utf-8") as ef:
                    era_summary = json.load(ef)

                # "Year Format Match" の平均を算出
                avg, total = 0.0, 0
                for prompt_id, data in era_summary.items():
                    if isinstance(data, dict) and METRIC_KEY in data:
                        avg += data[METRIC_KEY] * data.get("total", 1)
                        total += data.get("total", 1)
                era_values[era] = round(avg / total, 2) if total > 0 else 0.0

            results[model_display] = era_values

        except Exception as e:
            print(f"[Error] {model}: {e}")

    return results


def plot_heatmap(data_dict, output_dir, font_size=18):
    """ヒートマップを描画して保存"""
    df = pd.DataFrame(data_dict).T[ERAS]  # ← .T を戻す！
    df.columns = ERA_LABELS  # 先頭大文字化

    plt.figure(figsize=(40, 20))
    ax = sns.heatmap(
        df,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        vmin=0,
        vmax=1,
        annot_kws={"size": font_size},
        cbar_kws={"label": METRIC_KEY},
    )

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=font_size)
    cbar.set_label("Gregorian Bias Error", fontsize=font_size)

    plt.xlabel("Era", fontsize=font_size)
    plt.ylabel("Model", fontsize=font_size)
    plt.xticks(fontsize=font_size)
    plt.yticks(fontsize=font_size, rotation=0)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    png_path = Path(output_dir) / "year_format_match_heatmap.png"
    pdf_path = Path(output_dir) / "year_format_match_heatmap.pdf"

    plt.tight_layout()
    plt.savefig(png_path, dpi=300)
    plt.savefig(pdf_path)
    plt.close()
    print(f"Heatmap saved to:\n - {png_path}\n - {pdf_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate heatmap of Year Format Match for Japanese models.")
    parser.add_argument("--base_dir", type=str, required=True,
                        help="Base directory containing model results (e.g. /home/.../human_recall_eval/)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the output heatmap")
    parser.add_argument("--models", nargs="+", required=True,
                        help="List of model names to include (e.g., llm-jp-3-13b sarashina2-13b)")
    parser.add_argument("--setting", type=str, default="japrompt",
                        help="Prompt setting name (default: japrompt)")
    parser.add_argument("--name", type=str, default="ja-ja-name",
                        help="Subdirectory name for summaries (default: ja-ja-name)")
    parser.add_argument("--font_size", type=int, default=60,
                        help="Font size for labels and annotations")
    args = parser.parse_args()

    data = collect_year_match(args.base_dir, args.models, DEFAULT_MODEL_NAME_MAP, args.setting, args.name)
    plot_heatmap(data, args.output_dir, font_size=args.font_size)


if __name__ == "__main__":
    main()
