import json
import os
import random
import re
import requests
from datetime import datetime
from random import shuffle
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import numpy as np

API_URL = "https://api.deepseek.com/v1/chat/completions"
API_KEY = "Your API KEY"
MODEL_NAME = "deepseek-chat"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

random.seed(42)
np.random.seed(42)

prompt_lan = "en-prompt"
prompt_number = "3"
era_list = ["meiji", "taisho", "showa", "heisei"]
input_dir = "data/birthperiod/no_name"
max_tokens = 50
num_samples_per_era = 500
max_workers = 16

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

def call_deepseek(prompt, max_tokens=200):
    try:
        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": "You are an expert in the Japanese and Gregorian calendars. Please generate only the answer that continues from the text below."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": max_tokens,
            "temperature": 0
        }
        response = requests.post(API_URL, headers=headers, json=payload)
        response.raise_for_status()
        data = response.json()
        return data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[ERROR] {e}"


def generate_parallel(prompts, max_workers=16):
    results = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(call_deepseek, p, max_tokens): i for i, p in enumerate(prompts)}
        for future in tqdm(as_completed(futures), total=len(prompts)):
            idx = futures[future]
            results[idx] = future.result()
    return results


def run():
    output_base_dir = f"out/response/{MODEL_NAME}/after10/{prompt_lan}/prompt_number{prompt_number}"
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
        example_prompt = "The date 10 years after March 8, Tenpō 14 is March 8, Kōka 3."

        for entry in selected_data:
            wareki = entry["entity_label"]
            try:
                base_date = wareki_to_datetime(wareki)
                seireki_str = f"{base_date.year}年{base_date.month}月{base_date.day}日"
                wareki_en_macron = wareki_to_english_label_with_macron(base_date)
                prompt = f"{example_prompt} The date 10 years after {wareki_en_macron} is"
            except Exception as e:
                print(f"[ERROR] 和暦変換に失敗: {wareki} → {e}")
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

        responses = generate_parallel(prompts, max_workers=max_workers)

        results = []
        for meta, response in zip(metadata, responses):
            try:
                ten_years_later = meta["base_date"].replace(year=meta["base_date"].year + 10)
                correct_seireki = f"{ten_years_later.year}年{ten_years_later.month}月{ten_years_later.day}日"
                correct_wareki = wareki_to_english_label(ten_years_later)
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

        print(f"[{era}] {len(results)}件の結果を {era_output_file} に保存しました。")

    print(f"=== Finished {MODEL_NAME} ===")


if __name__ == "__main__":
    run()