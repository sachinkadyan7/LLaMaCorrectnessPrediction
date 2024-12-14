from activations.run_inference import get_activations
from transformers import LlamaForCausalLM, AutoTokenizer


import json
import numpy as np
import pandas as pd
import os
import torch

from pathlib import Path

PROMPT_MAP = lambda x: f"""
Please answer the following question with a single letter: A, B, C, or D.

{x["Prompt"]}
Answer:"""


def run_prompts(model, tokenizer, prompts, output_path):
    temp = {}
    for q_id, question in prompts.items():
        prompt, answer = question["Prompt"], question["Answer"]

        full_prompt = PROMPT_MAP(question)

        answers, attentions, activations, guess = get_activations(model, tokenizer, full_prompt, q_id, output_dir=output_path, max_new_tokens=5)

        temp[q_id] = temp.get(q_id, {})
        temp[q_id]["Predicted Answer"] = guess
        temp[q_id]["Full Answer"] = answers
        temp[q_id]["Prompt"] = prompt
        temp[q_id]["Correct Answer"] = answer


    df = pd.DataFrame.from_dict(temp, orient="index")
    return df


if __name__ == "__main__":
    split = "auxiliary_train"
    input_dir = Path.home() / f"Downloads/mmlu_data_clean_json/{split}/"

    output_path = Path.home() / f"Downloads/mmlu_output/{split}/"
    model_name = "meta-llama/Meta-Llama-3-8B"

    model = LlamaForCausalLM.from_pretrained(model_name, device_map="cuda", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id  = tokenizer.eos_token_id
    model.eval()

    for f in input_dir.glob("*.json"):
        with open(f, "r") as infile:
            prompts = json.load(infile)

        category = f.stem
        out_df = run_prompts(model=model, tokenizer=tokenizer, prompts=prompts, output_path=output_path/category)
        out_df.to_json(output_path/category/ f"{category}_answers.json", orient='index', indent=1)
