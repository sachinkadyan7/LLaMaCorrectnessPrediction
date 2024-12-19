import transformers
import os
from transformers import LlamaForCausalLM
import torch

from typing import List

def save_activation(activation, activation_save_path, file_name, suffix="activations"):
    if not os.path.exists(activation_save_path):
        os.makedirs(activation_save_path)

    activation_save_path = os.path.join(activation_save_path, f"{file_name}_{suffix}.pt")

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

    generate_ids = model.generate(inputs.input_ids, attention_mask=inputs.attention_mask, max_new_tokens=max_new_tokens, output_hidden_states=True, output_attentions=True)
    answers = tokenizer.decode(generate_ids[0][inputs.input_ids.shape[1]:]).split()

    # vocabulary indices for B are 33 but also 426 which is rly weird. 426 is the output one it seems. 

    intersection = set([val.strip() for val in answers]).intersection(["A", "B", "C", "D"])
    if not intersection:
        guess = None
    else:
        # join all the letters that show up just so that == comes up as false if we have a degenerate answer with multiple letters. 
        guess = "".join(intersection)
    
    if len(answers) >= 2:
        outputs = model(inputs.input_ids, output_hidden_states=True, output_attentions=True)
        attentions, activations = outputs["attentions"], outputs["hidden_states"]
    else:
        attentions, activations = None, None


    if save_activations:
        save_activation(activations, os.path.join(output_dir, "activations"), file_name, suffix="activations")
        save_activation(attentions, os.path.join(output_dir, "attentions"), file_name, suffix="attentions")       

    torch.cuda.empty_cache() 
    return answers, attentions, activations, guess


def get_batch_activations(
        model: LlamaForCausalLM,
        tokenizer: transformers.PreTrainedTokenizer,
        text: List[str], 
        output_dir: str,
        batch_name: str="",
        bs=16,
        ignore_attentions: bool = False,
        ignore_activations: bool = False,
        filter_activations: bool = False,
        save_activations: bool = True):
    
    with torch.no_grad():
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512, return_attention_mask=True).to(model.device)


        outputs = model(inputs.input_ids, output_hidden_states=not ignore_activations, output_attentions=not ignore_attentions, attention_mask=inputs.attention_mask)

        ids = outputs["logits"][:, -1].argmax(dim=1)

        if not ignore_attentions:
            attentions = outputs["attentions"]
            attentions = torch.stack(attentions, dim=0)
            attentions = attentions.permute(1, 0, 2, 3, 4) # attentions now have shape (batch_size, num_layers, num_heads, seq_len, seq_len)
            attentions = attentions[0, :, :, :, :] # attentions now have shape (num_layers, num_heads, seq_len, seq_len)
            if save_activations:
                save_activation(attentions, os.path.join(output_dir, "attentions"), batch_name, suffix="attentions")   
            del attentions
        if not ignore_activations:
            activations = outputs["hidden_states"] if not ignore_activations else None
            activations = torch.stack([activations[i] for i in (1, 11, 21, 31)], dim=0)
            activations = activations.permute(1, 0, 2, 3) # activations now have shape (batch_size, num_layers, seq_len, hidden_size)
            activations = activations[:, :, -1, :] # activations now have shape (batch_size, num_layers, hidden_size)

            if save_activations:
                save_activation(activations, os.path.join(output_dir, "activations"), batch_name, suffix="activations")
            del activations

        
        answers = [tokenizer.decode(ids[i]).strip() for i in range(bs)]

        # vocabulary indices for "B" is 33, and " B" is 426 which is rly weird. Based on experimentation, " B" is the output one but I cant be sure. 
        guesses = [answer if answer in ["A", "B", "C", "D"] else None for answer in answers]
        
        del inputs, outputs, ids
        torch.cuda.empty_cache()

    return answers, guesses

