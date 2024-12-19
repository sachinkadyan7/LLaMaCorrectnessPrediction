from activations.run_inference import get_activations, get_batch_activations
from transformers import LlamaForCausalLM, AutoTokenizer


import json
import numpy as np
import pandas as pd
import dotenv
import os
import torch

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

        answers, attentions, activations, guess = get_activations(model, tokenizer, full_prompt, file_name, output_dir=output_path, max_new_tokens=5)

        temp[file_name] = temp.get(file_name, {})
        temp[file_name]["Predicted Answer"] = guess
        temp[file_name]["Full Answer"] = answers
        temp[file_name]["Prompt"] = prompt
        temp[file_name]["Correct Answer"] = answer

        i += 1

    df = pd.DataFrame.from_dict(temp, orient="index")
    return df


def test_batch_get_activations(model, tokenizer, prompts, output_path):

    
    batched_prompts = [PROMPT_MAP(question) for i, question in enumerate(prompts.values()) if i < 16]
    labels = [question["Answer"] for i, question in enumerate(prompts.values()) if i < 16]
    answers, attentions, activations, guesses = get_batch_activations(model, tokenizer, batched_prompts, output_dir=output_path, batch_name="test_batch", bs=16)

    df = pd.DataFrame.from_dict({"Predicted Answer": guesses, "Full Answer": answers, "Prompt": batched_prompts, "Correct Answer": labels})
    return df


if __name__ == "__main__":
    example_json = ".data\\mmlu_data_clean_json\\auxiliary_train\\arc_easy.json"

    output_path = ".output\\auxiliary_train\\"
    dotenv.load_dotenv()
    token = os.getenv("HUGGINGFACE_TOKEN")
    model_name = "meta-llama/Meta-Llama-3-8B"
    with open(example_json, "r") as f:
        prompts = json.load(f)

    model = LlamaForCausalLM.from_pretrained(model_name, device_map="cuda", torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()

    out_df = test_get_activations(model=model, tokenizer=tokenizer, prompts=prompts, output_path=output_path+"arc_easy")
    out_df.to_json(output_path+"arc_easy\\arc_easy_answers.json", orient='index', indent=1)
    #test_batch_get_activations(model=model, tokenizer=tokenizer, prompts=prompts, output_path=output_path)