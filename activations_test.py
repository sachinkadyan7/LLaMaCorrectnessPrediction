from activations.run_inference import get_activations, model

import json
import numpy as np
import pandas as pd

example_dir = "mmlu_data_clean_json\\auxiliary_train\\arc_easy.json"

with open(example_dir, "r") as f:
    prompts = json.load(f)
    
i = 0
temp = {}
for file_name, question in prompts.items():
    if i > 20:
        break
    prompt, answer = question["Prompt"], question["Answer"]

    answers, attentions, activations, guess = get_activations(model, prompt, file_name, max_new_tokens=10)

    temp[file_name] = temp.get(file_name, {})
    temp[file_name]["Predicted Answer"] = guess
    temp[file_name]["Full Answer"] = answers
    temp[file_name]["Prompt"] = prompt
    temp[file_name]["Correct Answer"] = answer

    i += 1

df = pd.DataFrame.from_dict(temp, orient="index")
