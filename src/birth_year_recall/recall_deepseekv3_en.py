import json
import os
import random
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

API_KEY = "Your API Key"
API_URL = "https://api.deepseek.com/v1/chat/completions"
MODEL_NAME = "deepseek-chat"

def sample_few_shot_examples(seirekiwareki: str, seed: int = 42):
    if seirekiwareki == "wareki":
        return [
            ("Ieyasu Tokugawa", "Tenmon 11"),
            ("Takamori Saigō", "Bunsei 10"),
            ("Masako Hōjō", "Hōgen 2")
        ]
    else:
        return [
            ("Ieyasu Tokugawa", "1542"),
            ("Takamori Saigō", "1827"),
            ("Masako Hōjō", "1135")
        ]

def build_base_model_fewshot_prompt(target_name: str, seirekiwareki: str) -> str:
    shots = sample_few_shot_examples(seirekiwareki)
    if seirekiwareki == "wareki":
        lines = [f"The Japanese calendar dates {name}'s birth to {year}." for name, year in shots]
        lines.append(f"The Japanese calendar dates {target_name}'s birth to")
    else:
        lines = [f"The Gregorian calendar dates {name}'s birth to {year}." for name, year in shots]
        lines.append(f"The Gregorian calendar dates {target_name}'s birth to")
    return "\n".join(lines)

def get_prompt_config(seirekiwareki: str):
    return {
        "name": f"3-shot prompt ({seirekiwareki})",
        "description": f"予測対象: {'和暦' if seirekiwareki == 'wareki' else '西暦'}",
        "make_prompt": lambda name: build_base_model_fewshot_prompt(name, seirekiwareki)
    }

def call_deepseek(idx: int, prompt: str, max_tokens=50):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are an expert in the Japanese and Gregorian calendars. Please generate only the answer that continues from the text below."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0,
        "max_tokens": max_tokens
    }

    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=30)
        res = response.json()
        answer = res["choices"][0]["message"]["content"].strip()
    except Exception as e:
        answer = f"[ERROR] {str(e)}"

    return idx, answer

def generate_all(prompts, max_tokens=50, max_workers=10):
    results = [None] * len(prompts)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {
            executor.submit(call_deepseek, i, p, max_tokens): i
            for i, p in enumerate(prompts)
        }
        for future in tqdm(as_completed(future_to_idx), total=len(prompts)):
            idx, ans = future.result()
            results[idx] = ans
    return results

def main():
    prompt_lan = "enprompt"
    isen = "ja-en-name"
    seirekiwareki = "wareki"
    degree = 20
    sample_size = 300
    seed = 42
    prompt_number = 3
    is_label_en = "_en"
    en = ""
    input_dir = f"data/birthperiod/degree{degree}/merged/{en}"
    seirekiwareki_file = f"birthyear_{seirekiwareki}"
    target_eras = ["meiji", "taisho", "showa", "heisei"]

    config = get_prompt_config(seirekiwareki)
    make_prompt = config["make_prompt"]

    output_dir = os.path.join("out/human_recall", MODEL_NAME, seirekiwareki_file, prompt_lan, isen, f"prompt{prompt_number}")
    os.makedirs(output_dir, exist_ok=True)

    for era_name in target_eras:
        jsonl_path = os.path.join(input_dir, f"{era_name}_merged.jsonl")
        print(f"[{MODEL_NAME}] [prompt {prompt_number}] {config['name']} - 処理中: {era_name}")

        with open(jsonl_path, encoding="utf-8") as f:
            data = [json.loads(line) for line in f]

        random.seed(seed)
        data = random.sample(data, min(sample_size, len(data)))

        prompts, records = [], []
        for entry in data:
            prompt = make_prompt(entry[f"entity_label{is_label_en}"])
            prompts.append(prompt)
            records.append({
                "entity_id": entry["entity_id"],
                "entity_label": entry[f"entity_label{is_label_en}"],
                "period_label": entry["period_label"],
                "seireki_value": entry["seireki_value"],
                "wareki_value": entry["wareki_value"],
                "prompt_number": prompt_number,
                "prompt_name": config["name"],
                "prompt": prompt
            })

        responses = generate_all(prompts, max_tokens=50, max_workers=10)

        output_path = os.path.join(output_dir, f"{era_name}_llm_response.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for rec, res in zip(records, responses):
                rec["llm_response"] = res
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"[{MODEL_NAME}] [prompt {prompt_number}] {era_name} の結果を保存しました: {output_path}")

if __name__ == "__main__":
    main()