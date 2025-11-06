import asyncio
import json
import os
import random
from openai import AsyncOpenAI

API_KEY = "Your API Key"
client = AsyncOpenAI(api_key=API_KEY)

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
        lines = [f"According to the Japanese calendar, {name} was born in {year}." for name, year in shots]
        lines.append(f"According to the Japanese calendar dates {target_name} was born in")
    else:
        lines = [f"According to the Gregorian calendar, {name} was born in {year}." for name, year in shots]
        lines.append(f"According to the Gregorian calendar, {target_name} was born in")
    return "\n".join(lines)

def get_prompt_config(seirekiwareki: str):
    return {
        "name": f"3-shot prompt ({seirekiwareki})",
        "description": f"予測対象: {'和暦' if seirekiwareki == 'wareki' else '西暦'}",
        "make_prompt": lambda name: build_base_model_fewshot_prompt(name, seirekiwareki)
    }

# ==== gpt-4o 応答生成 ====
async def generate_response_gpt4o(prompts: list, max_tokens=50):
    tasks = []
    for idx, prompt in enumerate(prompts):
        tasks.append(_call_gpt4o(idx, prompt, max_tokens))
    results = await asyncio.gather(*tasks)
    # idxでソートして順序を保証
    results_sorted = [res for idx, res in sorted(results, key=lambda x: x[0])]
    return results_sorted

async def _call_gpt4o(idx, prompt, max_tokens):
    resp = await client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are an expert in the Japanese and Gregorian calendars. Please generate only the answer that continues from the text below."},
            {"role": "user", "content": prompt}
        ],
        max_tokens=max_tokens,
        temperature=0,
    )
    answer = resp.choices[0].message.content.strip()
    return idx, answer

async def main():
    prompt_lan = "enprompt"
    isen = "ja-en-name"
    en = ""
    is_label_en = "_en"
    seirekiwareki = "seireki"
    degree = 20
    seirekiwareki_file = f"birthyear_{seirekiwareki}"
    input_dir = f"data/birthperiod/degree{degree}/merged/{en}"
    sample_size = 300
    seed = 42

    target_eras = ["meiji", "taisho", "showa", "heisei"]

    prompt_number = 1
    config = get_prompt_config(seirekiwareki)
    make_prompt = config["make_prompt"]

    output_dir = os.path.join("out/human_recall", "gpt-4o", seirekiwareki_file, prompt_lan, isen, f"prompt{prompt_number}")
    os.makedirs(output_dir, exist_ok=True)

    for era_name in target_eras:
        jsonl_path = os.path.join(input_dir, f"{era_name}_merged.jsonl")
        print(f"[gpt-4o] [prompt {prompt_number}] {config['name']} - 処理中: {era_name}")

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

        responses = await generate_response_gpt4o(prompts)

        output_path = os.path.join(output_dir, f"{era_name}_llm_response.jsonl")
        with open(output_path, "w", encoding="utf-8") as f:
            for rec, res in zip(records, responses):
                rec["llm_response"] = res
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"[gpt-4o] [prompt {prompt_number}] {era_name} の結果を保存しました: {output_path}")

if __name__ == "__main__":
    asyncio.run(main())