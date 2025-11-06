import argparse
import json
import os

import matplotlib.pyplot as plt

model_name_map = {
    "Llama-2-7b-hf": "LLaMA2-7B",
    "Llama-2-13b-hf": "LLaMA2-13B",
    "Mistral-7B-v0.1": "Mistral-7B",
    "Llama-3.1-8B": "LLaMA3.1-8B",
    "llm-jp-3-13b": "LLM-JP-3-13B",
    "sarashina2-13b": "Sarashina2-13B",
    "Swallow-13b-hf": "Swallow-13B",
    "Swallow-MS-7b-v0.1": "Swallow-MS-7B",
    "Llama-3-Swallow-8B-v0.1": "LLaMA3-Swallow-8B",
    "gpt-4o": "GPT-4o",
    "deepseek-chat": "DeepSeek-V3"
}

model_colors = {
    "Llama-2-7b-hf": "#1f77b4",
    "Llama-2-13b-hf": "#000080",
    "Mistral-7B-v0.1": "#17becf",
    "Llama-3.1-8B": "#1c1cf0",
    "llm-jp-3-13b": "#d62728",
    "sarashina2-13b": "#ff7f0e",
    "Swallow-13b-hf": "#ff6f61",
    "Swallow-MS-7b-v0.1": "#8B0000",
    "Llama-3-Swallow-8B-v0.1": "#e6550d",
    "gpt-4o (en)": "#FF00FF",
    "gpt-4o (ja)": "#8B008B",
    "deepseek-chat (en)": "#EE82EE",
    "deepseek-chat (ja)": "#9400D3",
}

japanese_models = {
    "llm-jp-3-13b", "sarashina2-13b",
    "Swallow-13b-hf", "Swallow-MS-7b-v0.1",
    "Llama-3-Swallow-8B-v0.1"
}
english_models = {
    "Llama-2-7b-hf", "Llama-2-13b-hf",
    "Mistral-7B-v0.1", "Llama-3.1-8B"
}
dual_models = {"gpt-4o", "deepseek-chat"}

def get_prompt_setting(prompt_lan, model, default_notation):
    if prompt_lan == "natural-prompt":
        if model in japanese_models or model.endswith("(ja)"):
            return "japrompt", "ja-ja-name"
        elif model in english_models or model.endswith("(en)"):
            return "enprompt", "ja-en-name"
        elif model in dual_models:
            raise ValueError(f"Specify explicit language (ja/en) for {model}")
    elif prompt_lan == "ja-prompt":
        return "japrompt", "ja-ja-name"
    elif prompt_lan == "en-prompt":
        return "enprompt", "ja-en-name"
    else:
        raise ValueError("Plese set an approptiate prompt setting")


def load_summary(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f).get("average_scores", {})


def plot_metric_scatter(base_dir, out_dir, models, prompt_lan, notation, metric, ylabel):
    data = []
    for model in models:
        if model in dual_models:
            for lang in ["ja", "en"]:
                prompt_lan_model = f"{lang}prompt"
                notation_model = "ja-ja-name" if lang == "ja" else "ja-en-name"
                label = f"{model} ({lang})"
                color = model_colors.get(label, "#000000")

                seireki_path = f"{base_dir}/{model}/birthyear_seireki/{prompt_lan_model}/{notation_model}/summary_prompt_seireki.json"
                wareki_path = f"{base_dir}/{model}/birthyear_wareki/{prompt_lan_model}/{notation_model}/summary_prompt_wareki.json"
                seireki_scores = load_summary(seireki_path)
                wareki_scores = load_summary(wareki_path)
                if not seireki_scores or not wareki_scores:
                    continue

                x = seireki_scores.get(metric)
                y = wareki_scores.get(metric)
                if x is not None and y is not None:
                    data.append((label, x, y, color))
        else:
            prompt_lan_model, notation_model = get_prompt_setting(prompt_lan, model, notation)
            seireki_path = f"{base_dir}/{model}/birthyear_seireki/{prompt_lan_model}/{notation_model}/summary_prompt_seireki.json"
            wareki_path = f"{base_dir}/{model}/birthyear_wareki/{prompt_lan_model}/{notation_model}/summary_prompt_wareki.json"
            seireki_scores = load_summary(seireki_path)
            wareki_scores = load_summary(wareki_path)
            if not seireki_scores or not wareki_scores:
                continue

            x = seireki_scores.get(metric)
            y = wareki_scores.get(metric)
            if x is not None and y is not None:
                label = model_name_map.get(model, model)
                data.append((label, x, y, model_colors.get(model, "#000000")))

    if not data:
        print(f"[WARN] No data for metric '{metric}'")
        return

    plt.figure(figsize=(8, 8))
    for label, x, y, color in data:
        plt.scatter(x, y, s=300, color=color, label=label)

    min_val = min([min(x, y) for _, x, y, _ in data])
    max_val = max([max(x, y) for _, x, y, _ in data])
    pad = (max_val - min_val) * 0.05
    plt.plot([min_val - pad, max_val + pad], [min_val - pad, max_val + pad], "--", color="gray")

    plt.xlabel(f"{ylabel} (Gregorian)", fontsize=14)
    plt.ylabel(f"{ylabel} (Japanese)", fontsize=14)
    plt.title(f"{ylabel} Comparison", fontsize=15)
    plt.grid(True)
    # plt.legend(fontsize=9)
    plt.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    plt.savefig(f"{out_dir}/{metric}_scatter.png", dpi=300)
    plt.savefig(f"{out_dir}/{metric}_scatter.pdf", dpi=300)
    plt.close()
    print(f"[INFO] Saved scatter for {metric}")

def plot_full_and_tolerance_scatter(base_dir, out_dir, models, prompt_lan, notation):
    data = []

    for model in models:
        if model in dual_models:
            for lang in ["ja", "en"]:
                prompt_lan_model = f"{lang}prompt"
                notation_model = "ja-ja-name" if lang == "ja" else "ja-en-name"
                label = f"{model} ({lang})"
                color = model_colors.get(label, "#000000")

                seireki_path = f"{base_dir}/{model}/birthyear_seireki/{prompt_lan_model}/{notation_model}/summary_prompt_seireki.json"
                wareki_path = f"{base_dir}/{model}/birthyear_wareki/{prompt_lan_model}/{notation_model}/summary_prompt_wareki.json"
                seireki_scores = load_summary(seireki_path)
                wareki_scores = load_summary(wareki_path)
                if not seireki_scores or not wareki_scores:
                    continue

                data.append({
                    "label": label,
                    "color": color,
                    "full": (seireki_scores.get("full_match_rate"), wareki_scores.get("full_match_rate")),
                    "tol": (seireki_scores.get("tolerance_match_rate"), wareki_scores.get("tolerance_match_rate"))
                })
        else:
            prompt_lan_model, notation_model = get_prompt_setting(prompt_lan, model, notation)
            seireki_path = f"{base_dir}/{model}/birthyear_seireki/{prompt_lan_model}/{notation_model}/summary_prompt_seireki.json"
            wareki_path = f"{base_dir}/{model}/birthyear_wareki/{prompt_lan_model}/{notation_model}/summary_prompt_wareki.json"
            seireki_scores = load_summary(seireki_path)
            wareki_scores = load_summary(wareki_path)
            if not seireki_scores or not wareki_scores:
                continue

            data.append({
                "label": model_name_map.get(model, model),
                "color": model_colors.get(model, "#000000"),
                "full": (seireki_scores.get("full_match_rate"), wareki_scores.get("full_match_rate")),
                "tol": (seireki_scores.get("tolerance_match_rate"), wareki_scores.get("tolerance_match_rate"))
            })

    if not data:
        print("[WARN] No data found for full/tolerance comparison.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    for d in data:
        fx, fy = d["full"]
        tx, ty = d["tol"]
        if fx is None or fy is None or tx is None or ty is None:
            continue
        ax.scatter(fx, fy, s=300, color=d["color"], marker='o', label=f"{d['label']} (Full)")
        ax.scatter(tx, ty, s=300, facecolors='none', edgecolors=d["color"], marker='s', linewidths=2,
                   label=f"{d['label']} (±3 years)")

    all_vals = [v for d in data for v in d["full"] + d["tol"] if v is not None]
    vmin, vmax = min(all_vals), max(all_vals)
    pad = (vmax - vmin) * 0.05
    ax.plot([vmin - pad, vmax + pad], [vmin - pad, vmax + pad], "--", color="gray")

    ax.set_xlabel("Match Rate (Gregorian calendar)", fontsize=14)
    ax.set_ylabel("Match Rate (Japanese calendar)", fontsize=14)
    ax.set_title("Full vs Tolerance Match Rates", fontsize=15)
    ax.grid(True)

    handles, labels = ax.get_legend_handles_labels()

    base_labels = []
    for label in labels:
        clean = label.split(" (Full)")[0].split(" (±3 years)")[0]
        if clean not in base_labels:
            base_labels.append(clean)

    if prompt_lan == "natural-prompt":
        display_name_map = {
            "gpt-4o (ja)": "GPT-4o (ja)",
            "gpt-4o (en)": "GPT-4o (en)",
            "deepseek-chat (ja)": "DeepSeek-V3 (ja)",
            "deepseek-chat (en)": "DeepSeek-V3 (en)",
        }
    elif prompt_lan == "ja-prompt":
        display_name_map = {
            "gpt-4o (ja)": "GPT-4o (ja)",
            "deepseek-chat (ja)": "DeepSeek-V3 (ja)",
        }
    elif prompt_lan == "en-prompt":
        display_name_map = {
            "gpt-4o (en)": "GPT-4o (en)",
            "deepseek-chat (en)": "DeepSeek-V3 (en)",
        }
    else:
        raise ValueError("Plese set an approptiate prompt setting")

    ordered_labels = [m for m in model_colors.keys() if m in base_labels or model_name_map.get(m, m) in base_labels]

    model_handles = []
    display_names = []

    for m in ordered_labels:
        if m in display_name_map:
            display_name = display_name_map[m]
        else:
            base_key = m.split(" (")[0]
            display_name = model_name_map.get(base_key, m)

        color = model_colors.get(m, next((d["color"] for d in data if d["label"] == m), "black"))
        model_handles.append(plt.Line2D(
            [0], [0], marker='o', color='w',
            markerfacecolor=color, markersize=10, label=display_name
        ))
        display_names.append(display_name)

    match_type_handles = [
        plt.Line2D([0], [0], marker='o', color='black', markerfacecolor='black', markersize=10, label='Full match'),
        plt.Line2D([0], [0], marker='s', color='black', markerfacecolor='none', markersize=10, label='Within ±3 years')
    ]

    os.makedirs(out_dir, exist_ok=True)

    ax.legend().remove()
    plt.tight_layout()
    fig.savefig(f"{out_dir}/full_and_tolerance_scatter_main.pdf", dpi=300)
    fig.savefig(f"{out_dir}/full_and_tolerance_scatter_main.png", dpi=300)

    fig_model_legend = plt.figure(figsize=(3, len(display_names) * 0.35))
    fig_model_legend.legend(handles=model_handles, labels=display_names, loc='center', frameon=True,
                            fontsize=10, title="Models")
    fig_model_legend.tight_layout()
    fig_model_legend.savefig(f"{out_dir}/legend_models.pdf", bbox_inches='tight')
    plt.close(fig_model_legend)

    fig_match_legend = plt.figure(figsize=(3, 1.5))
    fig_match_legend.legend(handles=match_type_handles, loc='center', frameon=True,
                            fontsize=10, title="Match type")
    fig_match_legend.tight_layout()
    fig_match_legend.savefig(f"{out_dir}/legend_match_type.pdf", bbox_inches='tight')
    plt.close(fig_match_legend)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Japanese calendar recall plotter with GPT/DeepSeek dual support.")
    parser.add_argument("--base_dir", type=str, required=True, help="Base directory containing results")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save plots")
    parser.add_argument("--prompt_lan", type=str, default="natural-prompt", help="Prompt language type")
    parser.add_argument("--notation", type=str, default=None, help="Notation (for backward compatibility)")
    parser.add_argument("--models", nargs="+", default=list(model_name_map.keys()), help="List of models to include")
    args = parser.parse_args()

    metrics = {
        "full_match_rate": "Full Match Rate",
        "tolerance_match_rate": "Tolerance Match Rate",
        "format_match_rate": "Format Match Rate",
    }

    for metric, label in metrics.items():
        plot_metric_scatter(args.base_dir, args.output_dir, args.models, args.prompt_lan, args.notation, metric, label)
        plot_full_and_tolerance_scatter(args.base_dir, args.output_dir, args.models, args.prompt_lan, args.notation)
