import gc
import json
import os
import random
import re
from datetime import datetime
from random import shuffle

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

# ====== 設定 ======
model_names = [
    "Llama-2-7b-hf",
    "Llama-2-13b-hf",
    "Mistral-7B-v0.1",
    "llm-jp-3-13b",
    "sarashina2-13b",
    "Swallow-13b-hf",
    "Swallow-MS-7b-v0.1",
    "Llama-3.1-8B",
    "Llama-3-Swallow-8B-v0.1",
]

era_list = ["meiji", "taisho", "showa", "heisei"]
input_dir = "data/birthperiod/no_name"
prompt_number = "2"
batch_size = 64
max_tokens = 50
num_samples_per_era = 500  # 各時代から選ぶ件数

# ====== 和暦の境界定義 ======
ERA_BOUNDARIES = {
    "meiji":   {"start": datetime(1868, 1, 25),  "end": datetime(1912, 7, 29)},
    "taisho":  {"start": datetime(1912, 7, 30), "end": datetime(1926, 12, 24)},
    "showa":   {"start": datetime(1926, 12, 25), "end": datetime(1989, 1, 7)},
    "heisei":  {"start": datetime(1989, 1, 8),  "end": datetime(2019, 4, 30)},
    "reiwa":   {"start": datetime(2019, 5, 1),  "end": None},
}

ERA_JAPANESE_NAMES = {
    "meiji": "明治",
    "taisho": "大正",
    "showa": "昭和",
    "heisei": "平成",
    "reiwa": "令和"
}

# ====== 和暦変換関数（元年→1年） ======
def seireki_to_wareki(date: datetime):
    for era, bounds in reversed(ERA_BOUNDARIES.items()):
        start = bounds["start"]
        end = bounds["end"] if bounds["end"] else datetime.max
        if start <= date <= end:
            year = date.year - start.year + 1
            era_jp = ERA_JAPANESE_NAMES[era]
            return f"{era_jp}{year}年{date.month}月{date.day}日"
    return "不明"

# ====== 和暦 → datetime 変換 ======
def wareki_to_datetime(wareki_str):
    match = re.match(r"(明治|大正|昭和|平成|令和)(\d+)年(\d+)月(\d+)日", wareki_str)
    if not match:
        raise ValueError(f"和暦の形式が不正です: {wareki_str}")
    era_jp, year_str, month_str, day_str = match.groups()
    year = int(year_str)
    month = int(month_str)
    day = int(day_str)
    for era_key, era_name in ERA_JAPANESE_NAMES.items():
        if era_name == era_jp:
            start = ERA_BOUNDARIES[era_key]["start"]
            seireki_year = start.year + year - 1
            return datetime(seireki_year, month, day)
    raise ValueError(f"元号が不明です: {wareki_str}")

# ====== モデルごとにループ ======
for model_name in model_names:
    print(f"\n=== Starting model: {model_name} ===")

    model_path = f"/work00/share/hf_models/{model_name}/"
    output_base_dir = f"out/response/{model_name}/after10/ja-prompt/prompt_number{prompt_number}"
    os.makedirs(output_base_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    def generate_response(model, tokenizer, prompts, max_tokens=50):
        encoding = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        )

        input_ids = encoding.input_ids.to(model.device)
        attention_mask = encoding.attention_mask.to(model.device)

        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_tokens,
            pad_token_id=tokenizer.pad_token_id,
            do_sample=False,
        )

        responses = []
        for _i, output in enumerate(output_ids):
            input_length = input_ids.shape[1]
            generated_tokens = output[input_length:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            responses.append(response)
        return responses

    # ====== 時代ごとに処理 ======
    for era in era_list:
        file_path = os.path.join(input_dir, f"{era}_add_day.jsonl")
        if not os.path.exists(file_path):
            print(f"[警告] ファイルが見つかりません: {file_path}")
            continue

        with open(file_path, encoding="utf-8") as f:
            lines = [json.loads(line.strip()) for line in f]

        sorted_data = sorted(lines, key=lambda x: x["seireki_value"], reverse=True)
        last_5_years = sorted_data[:1825]

        shuffle(last_5_years)
        selected_data = last_5_years[:num_samples_per_era]

        prompts = []
        metadata = []

        for entry in selected_data:
            wareki = entry["entity_label"]
            prompt = f"天保14年3月8日から10年経つと弘化3年3月8日になります。{wareki}から10年経つと"
            try:
                base_date = wareki_to_datetime(wareki)
                seireki_str = f"{base_date.year}年{base_date.month}月{base_date.day}日"
            except Exception as e:
                print(f"[ERROR] 和暦変換に失敗: {wareki} → {e}")
                base_date = None
                seireki_str = "Error"

            metadata.append({
                "wareki": wareki,
                "seireki": seireki_str,
                "period_label": era.capitalize(),
                "prompt": prompt,
                "base_date": base_date
            })
            prompts.append(prompt)

        # ====== LLM に問い合わせ ======
        results = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_metadata = metadata[i:i + batch_size]

            responses = generate_response(model=model, tokenizer=tokenizer, prompts=batch_prompts)

            for meta, response in zip(batch_metadata, responses):
                try:
                    ten_years_later = meta["base_date"].replace(year=meta["base_date"].year + 10)
                    correct_seireki = f"{ten_years_later.year}年{ten_years_later.month}月{ten_years_later.day}日"
                    correct_wareki = seireki_to_wareki(ten_years_later)
                except Exception as e:
                    correct_seireki = "Error"
                    correct_wareki = "Error"
                    print(f"[ERROR] {meta['wareki']} の10年後処理失敗: {e}")

                result = {
                    "wareki": meta["wareki"],
                    "seireki": meta["seireki"],
                    "period_label": meta["period_label"],
                    "prompt": meta["prompt"],
                    "llm_response": response,
                    "correct_seireki": correct_seireki,
                    "correct_wareki": correct_wareki
                }
                results.append(result)

            print(f"[{model_name}][{era}] バッチ {i // batch_size + 1} / {(len(prompts) + batch_size - 1) // batch_size} 完了")

        # ====== 保存 ======
        era_output_file = os.path.join(output_base_dir, f"{era}_results.jsonl")
        with open(era_output_file, "w", encoding="utf-8") as f:
            for entry in results:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"[{model_name}][{era}] {len(results)}件の結果を {era_output_file} に保存しました。")

    # ====== モデルのメモリ解放 ======
    print(f"=== Finished model: {model_name}. Releasing memory ===")
    del model
    del tokenizer
    torch.cuda.empty_cache()
    gc.collect()
