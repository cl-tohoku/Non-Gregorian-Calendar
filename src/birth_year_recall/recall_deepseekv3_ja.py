import json
import os
import random
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import httpx

API_URL = "https://api.deepseek.com/v1/chat/completions"
API_KEY = "Your API Key"
MODEL_NAME = "deepseek-chat"

def sample_few_shot_examples(seirekiwareki: str, seed: int = 42):
    if seirekiwareki == "wareki":
        return [
            ("徳川家康", "天文11年"),
            ("西郷隆盛", "文政10年"),
            ("北条政子", "保元2年")
        ]
    else:
        return [
            ("徳川家康", "1542年"),
            ("西郷隆盛", "1827年"),
            ("北条政子", "1157年")
        ]

def build_base_model_fewshot_prompt(target_name: str, seirekiwareki: str) -> str:
    shots = sample_few_shot_examples(seirekiwareki)
    lines = [f"{name}の生まれ年は和暦で{year}です。" for name, year in shots]
    lines.append(f"{target_name}の生まれ年は和暦で")
    return "\n".join(lines)

def get_prompt_config(seirekiwareki: str):
    return {
        "name": f"ベースモデル向け3-shot形式（{seirekiwareki}）",
        "description": f"人物と{'和暦' if seirekiwareki == 'wareki' else '西暦'}の例を提示し、次の人物の生年を予測させる。",
        "make_prompt": lambda name: build_base_model_fewshot_prompt(name, seirekiwareki)
    }

def call_deepseek_sync(idx, prompt, max_tokens=50):
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "あなたは和暦とグレゴリオ暦の専門家です。以下に続くように文章を答えのみ生成してください。"},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens,
        "temperature": 0
    }

    try:
        response = httpx.post(API_URL, headers=headers, json=payload, timeout=30)
        response.raise_for_status()
        content = response.json()
        answer = content["choices"][0]["message"]["content"].strip()
    except Exception as e:
        answer = f"[ERROR] {e}"

    return idx, answer

def main():
    prompt_lan = "japrompt"
    isen = "ja-ja-name"
    en = ""
    is_label_en = ""
    seirekiwareki = "wareki"
    degree = 20
    seirekiwareki_file = f"birthyear_{seirekiwareki}"
    input_dir = f"data/birthperiod/degree{degree}/merged/{en}"
    sample_size = 300
    seed = 42

    target_eras = ["meiji", "taisho", "showa", "heisei"]
    prompt_number = 3
    config = get_prompt_config(seirekiwareki)
    make_prompt = config["make_prompt"]

    output_dir = os.path.join("out/human_recall", "deepseek-chat", seirekiwareki_file, prompt_lan, isen, f"prompt{prompt_number}")
    os.makedirs(output_dir, exist_ok=True)

    for era_name in target_eras:
        jsonl_path = os.path.join(input_dir, f"{era_name}_merged.jsonl")
        print(f"[deepseek-chat] [prompt {prompt_number}] {config['name']} - 処理中: {era_name}")

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

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(call_deepseek_sync, idx, prompt) for idx, prompt in enumerate(prompts)]
            results = [future.result() for future in tqdm(futures)]

        results_sorted = [res for idx, res in sorted(results, key=lambda x: x[0])]

        output_path = os.path.join(output_dir, f"{era_name}_llm_response.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for rec, res in zip(records, results_sorted):
                rec["llm_response"] = res
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"[deepseek-chat] [prompt {prompt_number}] {era_name} の結果を保存しました: {output_path}")

if __name__ == "__main__":
    main()