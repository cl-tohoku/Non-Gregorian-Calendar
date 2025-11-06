import json
import os
import random
import re
from datetime import datetime
from random import shuffle
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

import numpy as np
from openai import OpenAI

API_KEY = "Your API KEY"

client = OpenAI(api_key=API_KEY)
MODEL_NAME = "gpt-4o"

random.seed(42)
np.random.seed(42)

era_list = ["meiji", "taisho", "showa", "heisei"]
input_dir = "data/birthperiod/no_name"
prompt_number = "3"
max_tokens = 50
num_samples_per_era = 500
max_workers = 16

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

def seireki_to_wareki(date: datetime):
    for era, bounds in reversed(ERA_BOUNDARIES.items()):
        start = bounds["start"]
        end = bounds["end"] if bounds["end"] else datetime.max
        if start <= date <= end:
            year = date.year - start.year + 1
            era_jp = ERA_JAPANESE_NAMES[era]
            return f"{era_jp}{year}年{date.month}月{date.day}日"
    return "不明"

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

def call_gpt4o(prompt, max_tokens=400):
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "あなたは和暦とグレゴリオ暦の専門家です。以下に続くように文章を答えのみ生成してください。"},
                {"role": "user", "content": prompt}
            ],
            max_completion_tokens=max_tokens,
            temperature=0
        )
        return completion.choices[0].message.content.strip()
    except Exception as e:
        return f"[ERROR] {e}"

def generate_parallel(prompts, max_workers=16):
    results = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(call_gpt4o, p, max_tokens): idx
                   for idx, p in enumerate(prompts)}
        for future in tqdm(as_completed(futures), total=len(prompts)):
            idx = futures[future]
            results[idx] = future.result()
    return results

def run():
    print(f"\n=== Starting model: {MODEL_NAME} ===")

    output_base_dir = f"out/response/{MODEL_NAME}/after10/ja-prompt/prompt_number{prompt_number}"
    os.makedirs(output_base_dir, exist_ok=True)

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

        prompts, metadata = [], []

        for entry in selected_data:
            wareki = entry["entity_label"]
            prompt = f"天保14年3月8日に対する10年後の日付は弘化3年3月8日。\n{wareki}に対する10年後の日付は"
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

        responses = generate_parallel(prompts, max_workers=max_workers)
        results = []
        for meta, response in zip(metadata, responses):
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

        era_output_file = os.path.join(output_base_dir, f"{era}_results.jsonl")
        with open(era_output_file, "w", encoding="utf-8") as f:
            for entry in results:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        print(f"[{MODEL_NAME}][{era}] {len(results)}件の結果を {era_output_file} に保存しました。")

    print(f"=== Finished model: {MODEL_NAME} ===")

if __name__ == "__main__":
    run()