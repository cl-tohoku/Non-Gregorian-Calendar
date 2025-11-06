import gc
import glob
import json
import os
import re

import requests
from tqdm import tqdm

ERA_TRANSLATION = {
    "明治": "Meiji", "大正": "Taisho", "昭和": "Showa", "平成": "Heisei", "令和": "Reiwa",
}
ERA_LATIN_MACRON = {
    "明治": "Meiji", "大正": "Taishō", "昭和": "Shōwa", "平成": "Heisei", "令和": "Reiwa",
}

BOUNDARY_MAP = {
    ("明治", 45): ("大正", 1), ("大正", 1): ("明治", 45),
    ("大正", 15): ("昭和", 1), ("昭和", 1): ("大正", 15),
    ("昭和", 64): ("平成", 1), ("平成", 1): ("昭和", 64),
    ("平成", 31): ("令和", 1), ("令和", 1): ("平成", 31),
}

def is_valid_wareki_format(text):
    return bool(re.search(
        r"(Meiji|Taisho|Taishō|Showa|Shōwa|Heisei|Reiwa|"
        r"Ansei|Bunkyū|Bunkyu|Keiō|Keio|Genji|Man'en|Manen|"
        r"Bunsei|Tenpō|Tenpo|Kaei|Bunka)"
        r"\s(Gannen|\d{1,2})", text, re.IGNORECASE
    ))

def is_valid_seireki_format(text):
    return bool(re.search(r"\b\d{4}\b", text))

def call_deepseek(prompt, api_key, api_url, model, max_tokens=32):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert in the Japanese and Gregorian calendars. Please generate only the answer that continues from the text below."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
    }
    response = requests.post(api_url, headers=headers, json=payload)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()

def generate_batch(prompts, api_key, api_url, model, max_new_tokens=32):
    results = []
    for prompt in tqdm(prompts):
        try:
            res = call_deepseek(prompt, api_key, api_url, model, max_new_tokens)
            results.append(res.split("\n")[0])
        except Exception as e:
            print(f"error: {e}")
            results.append("")
    return results

def run(args):
    api_key = args.api_key
    api_url = "https://api.deepseek.com/v1/chat/completions"
    model = "deepseek-chat"
    input_dir = args.input_dir
    output_dir = args.output_dir
    output_dir = output_dir + f"/en-prompt/{model}"
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "wareki": {"correct": 0, "total": 0, "format_match": 0},
        "seireki": {"correct": 0, "total": 0, "format_match": 0},
        "per_era": {}
    }

    def is_correct_wareki(response, answer, alt_answer=None):
        answer_strs = []
        for era in [
            "Meiji", "Taisho", "Taishō", "Showa", "Shōwa", "Heisei", "Reiwa",
            "Ansei", "Bunkyū", "Bunkyu", "Keiō", "Keio", "Genji",
            "Man'en", "Manen", "Bunsei", "Tenpō", "Tenpo", "Kaei", "Bunka"
        ]:
            if answer == 1:
                answer_strs.append(f"{era} Gannen")
            answer_strs.append(f"{era} {answer}")
            if alt_answer is not None:
                if alt_answer == 1:
                    answer_strs.append(f"{era} Gannen")
                answer_strs.append(f"{era} {alt_answer}")
        return any(ans in response for ans in answer_strs)

    def is_correct_seireki(response, answer, alt_answer=None):
        patterns = [fr"\b{re.escape(str(answer))}\b"]
        if alt_answer is not None:
            patterns.append(fr"\b{re.escape(str(alt_answer))}\b")
        return any(re.search(p, response) for p in patterns)

    for path in sorted(glob.glob(f"{input_dir}/en/*.jsonl")):
        era = os.path.splitext(os.path.basename(path))[0]
        if era not in {"meiji", "taisho", "showa", "heisei"}:
            continue

        with open(path) as f:
            lines = [json.loads(line) for line in f]

        print(f" {era}: {len(lines)} data")
        summary["per_era"][era] = {
            "wareki": {"correct": 0, "total": 0, "format_match": 0},
            "seireki": {"correct": 0, "total": 0, "format_match": 0},
        }

        prompts_wareki, prompts_seireki, meta_entries = [], [], []
        for entry in lines:
            jp_period = entry["Era"]
            seireki = entry["Gregorian"]
            wareki_str = entry["Japanese"]

            wareki_match = re.search(r"\d+", wareki_str)
            wareki = int(wareki_match.group()) if wareki_match else 1

            eng_period = jp_period
            roman_period = jp_period

            alt_period, alt_wareki = BOUNDARY_MAP.get((jp_period, wareki), (None, None))
            alt_seireki = seireki if alt_period else None

            prompt_wareki = f"In the Japanese calendar, The year 1804 corresponds to Bunka 1. In the Japanese calendar, The year {seireki} corresponds to"
            prompt_seireki = f"In the Gregorian calendar, Bunka 1 corresponds to the year 1804. In the Gregorian calendar, {roman_period} {wareki} corresponds to the year"

            prompts_wareki.append(prompt_wareki)
            prompts_seireki.append(prompt_seireki)

            meta_entries.append({
                **entry,
                "seireki_value": seireki,
                "wareki_value": wareki,
                "period_label": jp_period,
                "prompt_wareki": prompt_wareki,
                "prompt_seireki": prompt_seireki,
                "alt_wareki": alt_wareki,
                "alt_seireki": alt_seireki,
                "eng_period": eng_period,
                "roman_period": roman_period
            })

        responses_wareki = generate_batch(prompts_wareki, api_key, api_url, model)
        responses_seireki = generate_batch(prompts_seireki, api_key, api_url, model)

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
                is_correct_wareki_flag = is_correct_wareki(res_wareki, wareki, alt_wareki) if is_format_wareki else False

                is_format_seireki = is_valid_seireki_format(res_seireki)
                is_correct_seireki_flag = is_correct_seireki(res_seireki, seireki, alt_seireki) if is_format_seireki else False

                for mode, is_format, is_correct_flag in [
                    ("wareki", is_format_wareki, is_correct_wareki_flag),
                    ("seireki", is_format_seireki, is_correct_seireki_flag)
                ]:
                    summary[mode]["total"] += 1
                    summary["per_era"][era][mode]["total"] += 1
                    if is_format:
                        summary[mode]["format_match"] += 1
                        summary["per_era"][era][mode]["format_match"] += 1
                        if is_correct_flag:
                            summary[mode]["correct"] += 1
                            summary["per_era"][era][mode]["correct"] += 1

                fout.write(json.dumps({
                    **meta,
                    "response_wareki": res_wareki,
                    "format_wareki": is_format_wareki,
                    "correct_wareki": is_correct_wareki_flag,
                    "response_seireki": res_seireki,
                    "format_seireki": is_format_seireki,
                    "correct_seireki": is_correct_seireki_flag,
                }, ensure_ascii=False) + "\n")

    for mode in ["wareki", "seireki"]:
        fmt = summary[mode]
        fmt["accuracy"] = round(fmt["correct"] / fmt["format_match"], 4) if fmt["format_match"] > 0 else 0.0
        fmt["format_rate"] = round(fmt["format_match"] / fmt["total"], 4)

    for era in summary["per_era"]:
        for mode in ["wareki", "seireki"]:
            fmt = summary["per_era"][era][mode]
            fmt["accuracy"] = round(fmt["correct"] / fmt["format_match"], 4) if fmt["format_match"] > 0 else 0.0
            fmt["format_rate"] = round(fmt["format_match"] / fmt["total"], 4)

    with open(os.path.join(output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    gc.collect()
