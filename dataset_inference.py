from activations.run_inference import get_activations, get_batch_activations
from transformers import LlamaForCausalLM, AutoTokenizer


import json
import numpy as np
import pandas as pd
import dotenv
import os
import torch

from pathlib import Path

PROMPT_MAP = lambda x: f"""
Please answer the following question with a single letter: A, B, C, or D.

{x["Prompt"]}
Answer:"""


def test_get_activations(model, tokenizer, prompts, output_path):
    i = 0
    temp = {}
    for file_name, question in prompts.items():
        if i > 20:
            break
        prompt, answer = question["Prompt"], question["Answer"]

        full_prompt = PROMPT_MAP(question)

        answers, attentions, activations, guess = get_activations(model, tokenizer, full_prompt, file_name, output_dir=output_path, max_new_tokens=10)

        temp[file_name] = temp.get(file_name, {})
        temp[file_name]["Predicted Answer"] = guess
        temp[file_name]["Full Answer"] = answers
        temp[file_name]["Prompt"] = prompt
        temp[file_name]["Correct Answer"] = answer

        i += 1

    df = pd.DataFrame.from_dict(temp, orient="index")
    return df


def test_batch_get_activations(model, tokenizer, prompts, output_path, batch_size):
    prompts["Final Prompt"] = prompts.apply(lambda row: PROMPT_MAP(row), axis=1)

    batch_idx = 0
    
    full_answers, full_guesses = [], []
    while batch_idx * batch_size < prompts.shape[0]:
        print(f"Batch {batch_idx}...")
        l, r = batch_idx * batch_size, (batch_idx + 1) * batch_size

        batch_prompts = prompts['Final Prompt'][l: r].tolist()
        bs = len(batch_prompts)
        answers, guesses = get_batch_activations(
            model, 
            tokenizer, 
            batch_prompts, 
            output_dir=output_path, 
            batch_name=batch_idx, 
            bs=bs, 
            ignore_activations=True,
            ignore_attentions=False,
            save_activations=True)

        full_answers.extend(answers)
        full_guesses.extend(guesses)

        batch_idx += 1

    df = pd.DataFrame.from_dict({"Predicted Answer": full_guesses, "Full Answer": full_answers, "Prompts": prompts['Final Prompt'], "Correct Answer": prompts['Answer']})

    return df


if __name__ == "__main__":
    mmlu_dir = Path.home() / "Downloads\\mmlu_data_clean_json\\auxiliary_train\\"
    output_path = ".output/"


    factoid_cat = os.path.join(mmlu_dir, "science_elementary.json")
    reasoning_cat = os.path.join(mmlu_dir, "arc_hard.json")

    factoid_output_path = os.path.join(output_path, "science_elementary")
    reasoning_output_path = os.path.join(output_path, "arc_hard")

    dotenv.load_dotenv()
    token = os.getenv("HUGGINGFACE_TOKEN")
    model_name = "meta-llama/Meta-Llama-3-8B"

    model = LlamaForCausalLM.from_pretrained(model_name, device_map="cuda", torch_dtype=torch.bfloat16, token=token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model.eval()

    prompts_fact = pd.read_json(factoid_cat, orient='index')
    results_fact = test_batch_get_activations(model, tokenizer, prompts_fact, factoid_output_path, batch_size=1)
    results_fact.to_json(os.path.join(factoid_output_path, "science_elementary.csv"), orient="index", indent=1)
    
    prompts_res = pd.read_json(reasoning_cat, orient='index')
    results_res = test_batch_get_activations(model, tokenizer, prompts_res, reasoning_output_path, batch_size=1)
    results_res.to_json(os.path.join(reasoning_output_path, "arc_hard.json"), orient="index", indent=1)
    
    #test_get_activations(model=model, tokenizer=tokenizer, prompts=prompts, output_path=output_path)
    #test_batch_get_activations(model=model, tokenizer=tokenizer, prompts=prompts, output_path=output_path)