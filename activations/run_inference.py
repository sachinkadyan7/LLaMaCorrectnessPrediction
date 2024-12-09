import transformers
import os
from transformers import LlamaForCausalLM
import torch


def save_activation(activation, activation_save_path, file_name, suffix="activations"):
    if not os.path.exists(activation_save_path):
        os.makedirs(activation_save_path)

    activation_save_path = os.path.join(activation_save_path, f"{os.path.splitext(file_name)[0]}_{suffix}.pt")

    torch.save(activation, activation_save_path)


def get_activations(
        model: LlamaForCausalLM, 
        tokenizer: transformers.PreTrainedTokenizer,
        text: str, 
        file_name: str, 
        output_dir: str,
        max_new_tokens: int = 10,
        save_activations: bool = True):

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    generate_ids = model.generate(inputs.input_ids, max_new_tokens=max_new_tokens, output_hidden_states=True, output_attentions=True)
    answers = tokenizer.decode(generate_ids[0][inputs.input_ids.shape[1]:]).split()

    # vocabulary indices for B are 33 but also 426 which is rly weird. 426 is the output one it seems. 

    intersection = set([val.strip() for val in answers]).intersection(["A", "B", "C", "D"])
    if not intersection:
        guess = None
    else:
        # join all the letters that show up just so that == comes up as false if we have a degenerate answer with multiple letters. 
        guess = "".join(intersection)
    
    if len(answers) == 2:
        outputs = model(inputs.input_ids, output_hidden_states=True, output_attentions=True)
        attentions, activations = outputs["attentions"], outputs["hidden_states"]
    else:
        attentions, activations = None, None

    if save_activations:
        save_activation(activations, os.path.join(output_dir, "activations"), file_name, suffix="activations")
        save_activation(attentions, os.path.join(output_dir, "attentions"), file_name, suffix="attentions")       

    return answers, attentions, activations, guess


def get_batch_activations(
        model: LlamaForCausalLM,
        tokenizer: transformers.PreTrainedTokenizer,
        text: str, 
        output_dir: str,
        batch_name: str="",
        bs=16,
        save_activations: bool = True):
    
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512, return_attention_mask=True).to(model.device)

    generate_ids = model.generate(
        inputs.input_ids, 
        max_new_tokens=1, 
        output_hidden_states=True, 
        output_attentions=True,
        eos_token_id=tokenizer.eos_token_id,
        attention_mask=inputs.attention_mask
        )
    answers = [tokenizer.decode(generate_ids[i, -1]).strip() for i in range(bs)]


    # vocabulary indices for "B" is 33, and " B" is 426 which is rly weird. Based on experimentation, " B" is the output one but I cant be sure. 
    guesses = [answer if answer in ["A", "B", "C", "D"] else None for answer in answers]

    with torch.no_grad():
        outputs = model(inputs.input_ids, output_hidden_states=True, output_attentions=True, attention_mask=inputs.attention_mask)

    attentions, activations = [], []
    for i in range(bs):
        if guesses[i] is not None:
            attentions.append(outputs["attentions"][i])
            activations.append(outputs["hidden_states"][i])
        else:
            attentions.append(None)
            activations.append(None)
    
    if save_activations:
        save_activation(activations, os.path.join(output_dir, "activations"), batch_name, suffix="activations")
        save_activation(attentions, os.path.join(output_dir, "attentions"), batch_name, suffix="attentions")   

    return answers, attentions, activations, guesses

