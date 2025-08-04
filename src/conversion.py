import glob
import json
import os
import re

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# モデルの準備
model_name = "Llama-2-7b-hf"
prompt_ln = "ja-prompt"
model_path = f"/work00/share/hf_models/{model_name}"
tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.padding_side = "left"
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype="auto",
    device_map="auto",
    output_hidden_states=True,
)

# 元号境界の許容
BOUNDARY_MAP = {
    ("明治", 45): ("大正", 1),
    ("大正", 1): ("明治", 45),
    ("大正", 15): ("昭和", 1),
    ("昭和", 1): ("大正", 15),
    ("昭和", 64): ("平成", 1),
    ("平成", 1): ("昭和", 64),
    ("平成", 31): ("令和", 1),
    ("令和", 1): ("平成", 31),
}

def is_valid_wareki_format(text):
    return bool(re.search(r"(明治|大正|昭和|平成|令和|安政|文久|慶応|元治|万延|文政|天保|嘉永|文化)[元\d]+年", text))

def is_valid_seireki_format(text):
    return bool(re.search(r"\d{4}年", text))

def is_correct(response, answer, alt_answer=None):
    answer_strs = [str(answer)]
    if answer == 1:
        answer_strs.append("元")
    if alt_answer is not None:
        answer_strs.append(str(alt_answer))
        if alt_answer == 1:
            answer_strs.append("元")
    return any(ans in response for ans in answer_strs)

def generate_batch(prompts, max_new_tokens=32, batch_size=64):
    results = []
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i:i+batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(model.device)

        # token_type_ids を削除（decoder-onlyモデルで不要）
        generate_inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}

        with torch.no_grad():
            outputs = model.generate(
                **generate_inputs,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id  # 警告抑制
            )

        for input_ids, output_ids in zip(inputs['input_ids'], outputs):
            decoded = tokenizer.decode(output_ids[len(input_ids):], skip_special_tokens=True)
            results.append(decoded.split("\n")[0].strip())
    return results

def main():
    input_dir = "data/birthperiod/no_name"
    output_dir = f"out/transform_task/{prompt_ln}/{model_name}"
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "wareki": {"correct": 0, "total": 0, "format_match": 0},
        "seireki": {"correct": 0, "total": 0, "format_match": 0},
        "per_era": {}  # 各元号の結果をここに追加
    }

    for path in sorted(glob.glob(f"{input_dir}/*.jsonl")):
        era = os.path.splitext(os.path.basename(path))[0]
        if era not in {"heisei", "meiji", "taisho", "showa"}:
            continue

        with open(path) as f:
            lines = [json.loads(l) for l in f]  # noqa: E741
        print(f"{era}: {len(lines)}件読み込み")

        # 各元号ごとの初期化
        summary["per_era"][era] = {
            "wareki": {"correct": 0, "total": 0, "format_match": 0},
            "seireki": {"correct": 0, "total": 0, "format_match": 0},
        }

        # プロンプトの準備
        prompts_wareki = []
        prompts_seireki = []
        meta_entries = []

        for entry in lines:
            seireki = entry["seireki_value"]
            wareki = entry["wareki_value"]
            period = entry["period_label"]
            alt_period, alt_wareki = BOUNDARY_MAP.get((period, wareki), (None, None))
            alt_seireki = seireki if alt_period else None

            one_shot_wareki = f"1804年を和暦に変換すると、文化元年です。{seireki}年を和暦に変換すると、"
            one_shot_seireki = f"文化1年を西暦に変換すると、1804年です。{period}{wareki}年を西暦に変換すると、"

            prompts_wareki.append(one_shot_wareki)
            prompts_seireki.append(one_shot_seireki)

            meta_entries.append({
                **entry,
                "prompt_wareki": one_shot_wareki,
                "prompt_seireki": one_shot_seireki,
                "alt_wareki": alt_wareki,
                "alt_seireki": alt_seireki,
            })

        # 推論
        responses_wareki = generate_batch(prompts_wareki)
        responses_seireki = generate_batch(prompts_seireki)

        # 結果保存
        output_path = os.path.join(output_dir, f"{era}.jsonl")
        with open(output_path, "w") as fout:
            for i, meta in enumerate(meta_entries):
                seireki = meta["seireki_value"]
                wareki = meta["wareki_value"]
                alt_wareki = meta["alt_wareki"]
                alt_seireki = meta["alt_seireki"]

                res_wareki = responses_wareki[i]
                res_seireki = responses_seireki[i]

                is_format_wareki = is_valid_wareki_format(res_wareki)
                is_correct_wareki = is_correct(res_wareki, wareki, alt_wareki) if is_format_wareki else False

                is_format_seireki = is_valid_seireki_format(res_seireki)
                is_correct_seireki = is_correct(res_seireki, seireki, alt_seireki) if is_format_seireki else False

                # 全体summary
                summary["wareki"]["total"] += 1
                summary["seireki"]["total"] += 1
                if is_format_wareki:
                    summary["wareki"]["format_match"] += 1
                    if is_correct_wareki:
                        summary["wareki"]["correct"] += 1
                if is_format_seireki:
                    summary["seireki"]["format_match"] += 1
                    if is_correct_seireki:
                        summary["seireki"]["correct"] += 1

                # era別summary
                summary["per_era"][era]["wareki"]["total"] += 1
                summary["per_era"][era]["seireki"]["total"] += 1
                if is_format_wareki:
                    summary["per_era"][era]["wareki"]["format_match"] += 1
                    if is_correct_wareki:
                        summary["per_era"][era]["wareki"]["correct"] += 1
                if is_format_seireki:
                    summary["per_era"][era]["seireki"]["format_match"] += 1
                    if is_correct_seireki:
                        summary["per_era"][era]["seireki"]["correct"] += 1

                fout.write(json.dumps({
                    **meta,
                    "response_wareki": res_wareki,
                    "format_wareki": is_format_wareki,
                    "correct_wareki": is_correct_wareki,
                    "response_seireki": res_seireki,
                    "format_seireki": is_format_seireki,
                    "correct_seireki": is_correct_seireki,
                }, ensure_ascii=False) + "\n")

    # summary 出力：割合の追加
    for mode in ["wareki", "seireki"]:
        fmt = summary[mode]
        fmt["accuracy"] = round(fmt["correct"] / fmt["format_match"], 4) if fmt["format_match"] > 0 else 0.0
        fmt["format_rate"] = round(fmt["format_match"] / fmt["total"], 4) if fmt["total"] > 0 else 0.0

    for era in summary["per_era"]:
        for mode in ["wareki", "seireki"]:
            fmt = summary["per_era"][era][mode]
            fmt["accuracy"] = round(fmt["correct"] / fmt["format_match"], 4) if fmt["format_match"] > 0 else 0.0
            fmt["format_rate"] = round(fmt["format_match"] / fmt["total"], 4) if fmt["total"] > 0 else 0.0

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
