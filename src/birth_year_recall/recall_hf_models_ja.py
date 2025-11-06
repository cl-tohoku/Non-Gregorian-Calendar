import gc
import json
import os
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


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
    lines = [f"{name}の生まれ年は{year}です。" for name, year in shots]
    lines.append(f"{target_name}の生まれ年は")
    return "\n".join(lines)

PROMPT_CONFIG = {
    "name": "ベースモデル向け3-shot形式",
    "description": "人物と和暦の例を提示し、次の人物の生年を予測させる。",
    "make_prompt": lambda name: build_base_model_fewshot_prompt(name)
}

def get_prompt_config(seirekiwareki: str):
    return {
        "name": f"ベースモデル向け3-shot形式（{seirekiwareki}）",
        "description": f"人物と{'和暦' if seirekiwareki == 'wareki' else '西暦'}の例を提示し、次の人物の生年を予測させる。",
        "make_prompt": lambda name: build_base_model_fewshot_prompt(name, seirekiwareki)
    }

# ==== 応答生成 ====
def generate_response(model, tokenizer, model_name, prompts: list, max_tokens=50, batch_size=64, injection_text=""):
    model_name_lower = model_name.lower()
    model_is_swallow_instruct = "swallow" in model_name_lower and "instruct" in model_name_lower
    model_is_elyza = "elyza" in model_name_lower and "instruct" in model_name_lower
    model_has_chat_template = (
        hasattr(tokenizer, "chat_template") and tokenizer.chat_template is not None
    )

    responses = []

    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i + batch_size]
        prompt_texts = []

        for prompt in batch_prompts:
            if model_is_swallow_instruct:
                prompt_texts.append(
                    f"{tokenizer.bos_token}以下に、あるタスクを説明する指示があります。\n"
                    f"リクエストを適切に完了するための回答を記述してください。\n\n"
                    f"### 指示:\n{prompt}\n\n"
                    f"### 応答:\n{injection_text}"
                )
            elif model_is_elyza:
                B_INST, E_INST = "[INST]", "[/INST]"
                B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
                system_prompt = "あなたは誠実で優秀な日本人のアシスタントです。"
                formatted = f"{tokenizer.bos_token}{B_INST} {B_SYS}{system_prompt}{E_SYS}{prompt} {injection_text} {E_INST} "
                prompt_texts.append(formatted)
            elif model_has_chat_template:
                chat_tensor = tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": "以下はタスクの指示です。適切に応答してください。"},
                        {"role": "user", "content": prompt + injection_text}
                    ],
                    add_generation_prompt=True,
                    return_tensors="pt"
                )
                chat_text = tokenizer.decode(chat_tensor[0], skip_special_tokens=False)
                prompt_texts.append(chat_text)
            else:
                prompt_texts.append(prompt + injection_text)

        encoding = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=256,
        )

        input_ids = encoding.input_ids.to(model.device)
        attention_mask = encoding.attention_mask.to(model.device)

        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_tokens,
                pad_token_id=tokenizer.pad_token_id,
                do_sample=False,
            )

        for _, output in enumerate(output_ids):
            input_len = input_ids.shape[1]
            generated_tokens = output[input_len:]
            response = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            responses.append(response)

        del input_ids, attention_mask, output_ids, encoding
        torch.cuda.empty_cache()

        print(f"バッチ {i // batch_size + 1} / {(len(prompts) - 1) // batch_size + 1} 処理完了")

    return responses

def main():
    model_names = [
        "Llama-2-7b-hf",
        "Llama-2-13b-hf",
        "Mistral-7B-v0.1",
        "llm-jp-3-13b",
        "Swallow-13b-hf",
        "Swallow-MS-7b-v0.1",
        "Llama-3.1-8B",
        "Llama-3-Swallow-8B-v0.1",
        "sarashina2-13b",
    ]

    prompt_lan = "japrompt"
    isen = "ja-ja-name"
    en = ""
    is_label_en = ""
    seirekiwareki = "seireki"
    degree = 20
    seirekiwareki_file = f"birthyear_{seirekiwareki}"
    input_dir = f"data/birthperiod/degree{degree}/merged/{en}"
    sample_size = 300
    seed = 42

    target_eras = ["meiji", "taisho", "showa", "heisei"]
    prompt_number = 3
    config = get_prompt_config(seirekiwareki)
    make_prompt = config["make_prompt"]

    for model_name in model_names:
        model_path = f"/work00/share/hf_models/{model_name}/"

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        output_dir = os.path.join("out/human_recall", model_name, seirekiwareki_file, prompt_lan, isen, f"prompt{prompt_number}")
        os.makedirs(output_dir, exist_ok=True)

        for era_name in target_eras:
            jsonl_path = os.path.join(input_dir, f"{era_name}_merged.jsonl")
            print(f"[{model_name}] [prompt {prompt_number}] {config['name']} - 処理中: {era_name}")

            with open(jsonl_path, encoding="utf-8") as f:
                data = [json.loads(line) for line in f]

            random.seed(seed)
            data = random.sample(data, min(sample_size, len(data)))

            prompts = []
            records = []

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

            responses = generate_response(model, tokenizer, model_name, prompts, injection_text="")

            output_path = os.path.join(output_dir, f"{era_name}_llm_response.jsonl")
            with open(output_path, "w", encoding="utf-8") as f:
                for rec, res in zip(records, responses):
                    rec["llm_response"] = res
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            print(f"[{model_name}] [prompt {prompt_number}] {era_name} の結果を保存しました: {output_path}")
        del model
        del tokenizer
        gc.collect()
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
