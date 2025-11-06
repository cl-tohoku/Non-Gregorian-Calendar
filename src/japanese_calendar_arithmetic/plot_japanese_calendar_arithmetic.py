import argparse
import json
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

MODEL_NAME_MAP = {
    "llm-jp-3-13b":        "LLM-JP-3-13B",
    "sarashina2-13b":      "Sarashina2-13B",
    "Swallow-13b-hf":      "Swallow-13B",
    "Swallow-MS-7b-v0.1":  "Swallow-MS-7B",
    "Llama-3-Swallow-8B-v0.1": "LLaMA3-Swallow-8B",
    "Llama-2-7b-hf":       "LLaMA2-7B",
    "Llama-2-13b-hf":      "LLaMA2-13B",
    "Mistral-7B-v0.1":     "Mistral-7B",
    "Llama-3.1-8B":        "LLaMA3.1-8B",
    "gpt-4o":              "GPT-4o",
    "deepseek-chat":       "DeepSeek-V3",
}

JAPANESE_MODELS = {
    "llm-jp-3-13b", "sarashina2-13b",
    "Swallow-13b-hf", "Swallow-MS-7b-v0.1",
    "Llama-3-Swallow-8B-v0.1"
}

PURPLE_MODELS = {"gpt-4o", "deepseek-chat"}

ERA_LABEL_MAP_AFTER = {
    "meiji": "Meiji→Taisho",
    "taisho": "Taisho→Showa",
    "showa": "Showa→Heisei",
    "heisei": "Heisei→Reiwa"
}
ERA_LABEL_MAP_BEFORE = {
    "taisho": "Taisho→Meiji",
    "showa": "Showa→Taisho",
    "heisei": "Heisei→Showa",
    "reiwa": "Reiwa→Heisei"
}

METRICS = {
    "gengo_match": "Era Match Rate",
    "almost_match": "Nearly Match Rate",
    "year_match": "Full Match Rate"
}


def load_results(model, model_display, base_dir, task_type, prompt_lan):
    if task_type == "before":
        eras = ["taisho", "showa", "heisei", "reiwa"]
        era_label_map = ERA_LABEL_MAP_BEFORE
    else:
        eras = ["meiji", "taisho", "showa", "heisei"]
        era_label_map = ERA_LABEL_MAP_AFTER

    if prompt_lan == "natural-prompt":
        actual_prompt_lan = "ja-prompt" if model in JAPANESE_MODELS else "en-prompt"
    else:
        actual_prompt_lan = prompt_lan

    path = os.path.join(base_dir, model, f"{task_type}10", actual_prompt_lan, "results_summary.json")
    if not os.path.exists(path):
        print(f"[WARN] Missing: {path}")
        return None

    with open(path, encoding="utf-8") as f:
        data = json.load(f)

    best_prompt = data.get("best_prompt_per_metric", {})
    record = {metric_name: {} for metric_name in METRICS.values()}

    for era in eras:
        if era not in best_prompt:
            continue
        for metric, metric_name in METRICS.items():
            score = best_prompt.get(era, {}).get(metric, None)
            if score is None:
                continue

            if isinstance(score, list) and len(score) > 0:
                try:
                    score = float(score[-1])
                except Exception:
                    continue
            elif not isinstance(score, (int, float)):
                continue

            record[metric_name][era_label_map[era]] = round(score, 3)

    return record

def draw_heatmap(data_dict, metric_display, task_type, prompt_lan, output_dir):
    df = pd.DataFrame.from_dict(data_dict, orient="index")

    ordered_rows = []
    for k, v in MODEL_NAME_MAP.items():
        if v in df.index:
            ordered_rows.append(v)
        if f"{v} (ja)" in df.index:
            ordered_rows.append(f"{v} (ja)")
        if f"{v} (en)" in df.index:
            ordered_rows.append(f"{v} (en)")
    df = df.reindex(index=ordered_rows)

    plt.figure(figsize=(10, 8))
    ax = sns.heatmap(df, annot=True, cmap="Reds", vmin=0, vmax=1, fmt=".2f",
                     cbar_kws={"label": "Accuracy"}, annot_kws={"fontsize": 18})
    cbar = ax.collections[0].colorbar
    cbar.ax.yaxis.label.set_size(20)
    cbar.ax.tick_params(labelsize=20)

    title_prefix = "add" if task_type == "after" else "subtract"
    ax.set_title(f"{title_prefix}: {metric_display}", pad=12, fontsize=20)
    ax.set_xlabel("Era", labelpad=10, fontsize=20)
    ax.set_ylabel("Model", labelpad=10, fontsize=20)
    plt.xticks(rotation=30, ha="center", fontsize=18)
    plt.yticks(rotation=0, fontsize=18)

    for tick_label in ax.get_yticklabels():
        label_text = tick_label.get_text()
        base_label = label_text.split(" ")[0]
        dir_name = next((k for k, v in MODEL_NAME_MAP.items() if v == base_label), None)
        if dir_name in PURPLE_MODELS:
            tick_label.set_color("#9467bd")  # 紫
        elif dir_name in JAPANESE_MODELS or "(ja)" in label_text:
            tick_label.set_color("#d62728")  # 赤
        else:
            tick_label.set_color("#1f77b4")  # 青

    os.makedirs(output_dir, exist_ok=True)
    filename = f"{prompt_lan}_{metric_display.replace(' ', '_')}_heatmap"
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename}.png"), dpi=300)
    plt.savefig(os.path.join(output_dir, f"{filename}.pdf"))
    plt.close()
    print(f"[INFO] Saved: {filename}.png/.pdf")


def main():
    parser = argparse.ArgumentParser(description="Generate Japanese calendar arithmetic heatmaps.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing model results")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for heatmaps")
    parser.add_argument("--task_type", choices=["before", "after"], required=True)
    parser.add_argument("--prompt_lan", default="natural-prompt")
    parser.add_argument("--models", nargs="+", default=list(MODEL_NAME_MAP.keys()))
    args = parser.parse_args()

    tables = {metric_name: {} for metric_name in METRICS.values()}

    for model in args.models:
        model_display = MODEL_NAME_MAP[model]
        record = load_results(model, model_display, args.base_dir, args.task_type, args.prompt_lan)
        if not record:
            continue

        if model in PURPLE_MODELS:
            for lang_suffix in ["ja", "en"]:
                display_label = f"{model_display} ({lang_suffix})"
                results = load_results(model, model_display, args.base_dir, args.task_type, f"{lang_suffix}-prompt")
                if results:
                    for metric_name, scores in results.items():
                        tables[metric_name][display_label] = scores
        else:
            for metric_name, scores in record.items():
                tables[metric_name][model_display] = scores

    for metric_name, data_dict in tables.items():
        if not data_dict:
            print(f"[WARN] No data for {metric_name}")
            continue
        draw_heatmap(data_dict, metric_name, args.task_type, args.prompt_lan, args.output_dir)


if __name__ == "__main__":
    main()
