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
        answers, _, _, guesses = get_batch_activations(
            model, 
            tokenizer, 
            batch_prompts, 
            output_dir=output_path, 
            batch_name=batch_idx, 
            bs=bs, 
            ignore_activations=False,
            ignore_attentions=True,
            save_activations=True)

        full_answers.extend(answers)
        full_guesses.extend(guesses)

        batch_idx += 1

    df = pd.DataFrame.from_dict({"Predicted Answer": full_guesses, "Full Answer": full_answers, "Prompts": prompts['Final Prompt'], "Correct Answer": prompts['Answer']})

    return df


if __name__ == "__main__":
    mmlu_dir = Path.home() / "Downloads\\mmlu_data_clean\\auxiliary_train\\"
    output_path = ".output/"


    factoid_cat = os.path.join(mmlu_dir, "science_elementary.csv")
    reasoning_cat = os.path.join(mmlu_dir, "arc_hard.csv")

    factoid_output = os.path.join(output_path, "science_elementary")
    reasoning_output = os.path.join(output_path, "arc_hard")

    dotenv.load_dotenv()
    token = os.getenv("HUGGINGFACE_TOKEN")
    model_name = "meta-llama/Meta-Llama-3-8B"

    model = LlamaForCausalLM.from_pretrained(model_name, device_map="cuda", torch_dtype=torch.bfloat16, token=token)
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
    model.eval()

    prompts_fact = pd.read_csv(factoid_cat)
    prompts_fact = prompts_fact[prompts_fact["Prompt"].apply(lambda x: len(x) < 512)]
    results_fact = test_batch_get_activations(model, tokenizer, prompts_fact, factoid_output, batch_size=32)
    results_fact.to_csv(output_path + "science_elementary.csv")
    
    prompts_res = pd.read_csv(reasoning_cat)
    prompts_res = prompts_res[prompts_res["Prompt"].apply(lambda x: len(x) < 512)]
    results_res = test_batch_get_activations(model, tokenizer, prompts_res, reasoning_output, batch_size=32)
    results_res.to_csv(output_path + "arc_hard.csv")
    
    #test_get_activations(model=model, tokenizer=tokenizer, prompts=prompts, output_path=output_path)
    #test_batch_get_activations(model=model, tokenizer=tokenizer, prompts=prompts, output_path=output_path)