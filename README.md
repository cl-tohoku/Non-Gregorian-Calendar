# Can Language Models Handle a Non-Gregorian Calendar? The Case of the Japanese 
This repository contains code and data for the paper: 

*Can Language Models Handle a Non-Gregorian Calendar? The Case of the Japanese wareki* 

Our study presents a systematic evaluation of how language models handle the Japanese calendar, a representative non-Gregorian system. 
We designed three tasks requiring both temporal knowledge and reasoning: **Calendar Conversion**, **Japanese Calendar arithmetic**, **Birth year recall**.
We find that while some models succeed in basic calendar conversions, even Japanese-centric LMs and advanced models such as GPT-4o and DeepSeek V3 struggle with Japanese-calendar arithmetic and birth year recall. 
The findings reveal a strong bias toward the Gregorian calendar and highlight the need for culturally aware temporal reasoning in language models.

**🎉This paper has been accepted of AACL-IJCNLP 2025🎉**

## Requirements
If you use `uv`, you can set up the environment as follows. Please run the experiments inside `Non-Gregorian_Calendar` directory:
```
cd Non-Gregorian_Calendar
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt
```

## Models
We evaluated five Japanese-cenric LMs four English-centric LMs, and two state-of-the-art models.

- Japanese-centric LMs
    - LLM-JP-3-13B
    - Sarashina2-13B
    - Swallow-13b
    - Swallow-MS-7B
    - LLaMA3-Swallow-8B

- English-centric LMs
    - LLaMA2-7B
    - LLaMA2-13B
    - Mistral-7B
    - LLaMA-3.1-8B

## Dataset
We constructed datasets for three tasks (Calendar Conversion, Japanese Calendar Arithmetic, and Birth Year Recall) covering Japanese five most recent eras: Meiji, Taisho, Showa, Heisei, and Reiwa.
For details, please refer to the `data` directory.


## Calendar conversion
You can obtain the results by running the following command.

```
python src/calendar_conversion/plot_calendar_conversion.py \
  --base_dir out/calendar_conversion/ \
  --output_dir out/calendar_conversion/plot/ \
  --models llm-jp-3-13b sarashina2-13b Swallow-13b-hf Swallow-MS-7b-v0.1 Llama-3-Swallow-8B-v0.1 Llama-2-7b-hf Llama-2-13b-hf Mistral-7B-v0.1 Llama-3.1-8B gpt-4o deepseek-chat 

```

## Japanese calendar arithmetic
You can obtain the results by running the following command.

```
python src/japanese_calendar_arithmetic/plot_japanese_calendar_arithmetic.py \
  --base_dir out/japanese_calendar_arithmetic/ \
  --output_dir out/japanese_calendar_arithmetic//plot/ \
  --task_type after \
  --models llm-jp-3-13b sarashina2-13b Swallow-13b-hf Swallow-MS-7b-v0.1 Llama-3-Swallow-8B-v0.1 Llama-2-7b-hf Llama-2-13b-hf Mistral-7B-v0.1 Llama-3.1-8B gpt-4o deepseek-chat
```

## Birth year recall
You can obtain the results by running the following command.

```
python src/birth_year_recall/plot_birth_year_recall.py \
  --base_dir out/birth_year_recall/summary/ \
  --output_dir out/birth_year_recall/plot/ \
  --models llm-jp-3-13b sarashina2-13b Swallow-13b-hf Swallow-MS-7b-v0.1 Llama-3-Swallow-8B-v0.1 Llama-2-7b-hf Llama-2-13b-hf Mistral-7B-v0.1 Llama-3.1-8B gpt-4o deepseek-chat
  ```

## Discussion
We analyzed why the models failed, particularly in Japanese Calendar Arithmetic and Birth Year Recall, from the perspectives of estimated corpus frequency (by using infini-gram[1]) and typical error patterns.

### Estimated Corpus frequency
You can obtain the results of error analysis of estimated corpus frequency by running the following command.

- Japanese Calendar arithmetic
```
python src/discussion/plot_corpus_estimated_analysis_japanese_calendar_arithmetic.py \
  --corpus_dir data/infini-gram/ \
  --results_file out/japanese_calendar_arithmetic/llm-jp-3-13b/after10/ja-prompt/results_summary.json \
  --output /home/m_sasaki/Non-Gregorian_Calendar/out/discussion/estimates_corpus_freqency/japanese_calendar_arithmetic/
```

- Birth year recall
```
python src/discussion/plot_corpus_estimated_analysis_birth_year_recall.py \
  --corpus_dir data/infini-gram/ \
  --results_dir out/birth_year_recall/summary/llm-jp-3-13b/birthyear_wareki/japrompt/ja-ja-name/ \
  --output out/discussion/estimates_corpus_freqency/birth_year_recall/
  ```

### Typical Errors

- Japanese Calendar arithmetic
```
python src/discussion/plot_typical_error_japanese_calendar_arithmetic.py \
  --base_dir out/Japanese_calendar_arithmetic/ \
  --output_dir out/discussion/typical_error/japanese_calendar_arithmetic/ \
  --models sarashina2-13b Llama-3-Swallow-8B-v0.1 Mistral-7B-v0.1 
```

- Birth year recall
```
python src/discussion/plot_typical_error_birth_year_recall.py \
  --base_dir out/birth_year_recall/summary/ \
  --output_dir out/discussion/typical_error/birth_year_recall/ \
  --models llm-jp-3-13b sarashina2-13b Swallow-13b-hf Swallow-MS-7b-v0.1 Llama-3-Swallow-8B-v0.1 
```

## Reference
[1] Liu+ Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens (2024)
```
@article{Liu2024InfiniGram,
  title={Infini-gram: Scaling Unbounded n-gram Language Models to a Trillion Tokens},
  author={Liu, Jiacheng and Min, Sewon and Zettlemoyer, Luke and Choi, Yejin and Hajishirzi, Hannaneh},
  journal={arXiv preprint arXiv:2401.17377},
  year={2024}
}
```

## Citation
This is the version that has been published on arXiv.
**It will be updated soon.**

```
@misc{sasaki2025languagemodelshandlenongregorian,
      title={Can Language Models Handle a Non-Gregorian Calendar?}, 
      author={Mutsumi Sasaki and Go Kamoda and Ryosuke Takahashi and Kosuke Sato and Kentaro Inui and Keisuke Sakaguchi and Benjamin Heinzerling},
      year={2025},
      eprint={2509.04432},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2509.04432}, 
}
```