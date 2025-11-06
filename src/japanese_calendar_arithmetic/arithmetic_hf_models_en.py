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

prompt_lan = "en-prompt"
prompt_number = "3"
era_list = ["meiji", "taisho", "showa", "heisei"]
input_dir = "data/birthperiod/no_name"
batch_size = 64
max_tokens = 50
num_samples_per_era = 500

ERA_BOUNDARIES = {
    "meiji": {"start": datetime(1868, 1, 25), "end": datetime(1912, 7, 29)},
    "taisho": {"start": datetime(1912, 7, 30), "end": datetime(1926, 12, 24)},
    "showa": {"start": datetime(1926, 12, 25), "end": datetime(1989, 1, 7)},
    "heisei": {"start": datetime(1989, 1, 8), "end": datetime(2019, 4, 30)},
    "reiwa": {"start": datetime(2019, 5, 1), "end": None},
}

ERA_JAPANESE_NAMES = {
    "meiji": "明治",
    "taisho": "大正",
    "showa": "昭和",
    "heisei": "平成",
    "reiwa": "令和"
}

ERA_ENGLISH_NAMES = {
    "meiji": "Meiji",
    "taisho": "Taisho",
    "showa": "Showa",
    "heisei": "Heisei",
    "reiwa": "Reiwa"
}

ERA_MACRON_NAMES = {
    "meiji": "Meiji",
    "taisho": "Taishō",
    "showa": "Shōwa",
    "heisei": "Heisei",
    "reiwa": "Reiwa"
}


def wareki_to_datetime(wareki_str):
    match = re.match(r"(明治|大正|昭和|平成)(\d+)年(\d+)月(\d+)日", wareki_str)
    if not match:
        raise ValueError(f"和暦の形式が不正です: {wareki_str}")
    era_jp, year_str, month_str, day_str = match.groups()
    year, month, day = int(year_str), int(month_str), int(day_str)
    for era_key, era_name in ERA_JAPANESE_NAMES.items():
        if era_name == era_jp:
            start = ERA_BOUNDARIES[era_key]["start"]
            seireki_year = start.year + year - 1
            return datetime(seireki_year, month, day)
    raise ValueError(f"元号が不明です: {wareki_str}")


def wareki_to_english_label(date: datetime):
    for era, bounds in reversed(ERA_BOUNDARIES.items()):
        start = bounds["start"]
        end = bounds["end"] if bounds["end"] else datetime.max
        if start <= date <= end:
            year = date.year - start.year + 1
            era_en = ERA_ENGLISH_NAMES[era]
            month_str = date.strftime("%B")
            return f"{month_str} {date.day}, {era_en} {year}"
    return "Unknown"


def wareki_to_english_label_with_macron(date: datetime):
    for era, bounds in reversed(ERA_BOUNDARIES.items()):
        start = bounds["start"]
        end = bounds["end"] if bounds["end"] else datetime.max
        if start <= date <= end:
            year = date.year - start.year + 1
            era_en_macron = ERA_MACRON_NAMES[era]
            month_str = date.strftime("%B")
            return f"{month_str} {date.day}, {era_en_macron} {year}"
    return "Unknown"


for model_name in model_names:
    print(f"\n=== Starting model: {model_name} ===")

    model_path = f"/work00/share/hf_models/{model_name}/"
    output_base_dir = f"out/response/{model_name}/after10/{prompt_lan}/prompt_number{prompt_number}"
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
        for _, output in enumerate(output_ids):
            input_length = input_ids.shape[1]
            generated_tokens = output[input_length:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            responses.append(response)

        return responses

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
            example_prompt = "The date 10 years after March 8, Tenpō 14 is March 8, Kōka 3."

            try:
                base_date = wareki_to_datetime(wareki)
                seireki_str = f"{base_date.year}年{base_date.month}月{base_date.day}日"
                wareki_en_macron = wareki_to_english_label_with_macron(base_date)
                prompt = f"{example_prompt} The date 10 years after {wareki_en_macron} is"
            except Exception as e:
                print(f"[ERROR] fail in wareki conversion: {wareki} → {e}")
                base_date = None
                seireki_str = "Error"
                prompt = f"{example_prompt} (Invalid date)"

            metadata.append({
                "wareki": wareki,
                "seireki": seireki_str,
                "period_label": era.capitalize(),
                "prompt": prompt,
                "base_date": base_date
            })
            prompts.append(prompt)

        results = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            batch_metadata = metadata[i:i + batch_size]

            responses = generate_response(model=model, tokenizer=tokenizer, prompts=batch_prompts)

            for meta, response in zip(batch_metadata, responses):
                try:
                    ten_years_later = meta["base_date"].replace(year=meta["base_date"].year + 10)
                    correct_seireki = f"{ten_years_later.year}年{ten_years_later.month}月{ten_years_later.day}日"
                    correct_wareki = wareki_to_english_label(ten_years_later)
                except Exception as e:
                    correct_seireki = "Error"
                    correct_wareki = "Error"
                    print(f"[ERROR] {e}")

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

            print(f"[{era}] バッチ {i // batch_size + 1} / {(len(prompts) + batch_size - 1) // batch_size} 完了")

        era_output_file = os.path.join(output_base_dir, f"{era}_results.jsonl")
        with open(era_output_file, "w", encoding="utf-8") as f:
            for entry in results:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"[{era}] {len(results)} -> {era_output_file}")

    print(f"=== Finished model: {model_name}. Releasing memory ===")
    del model
    del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
