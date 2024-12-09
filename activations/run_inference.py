import transformers
import torch
import dotenv
import os

from transformers import LlamaForCausalLM, AutoTokenizer

dotenv.load_dotenv()
token = os.getenv("HUGGINGFACE_TOKEN")

model_name = "meta-llama/Meta-Llama-3-8B"
data_path = ".data"
activations_path = os.path.join(data_path, "activations")
attentions_path = os.path.join(data_path, "attentions")
if not os.path.exists(activations_path):
    os.makedirs(activations_path)
    print(f"Created directory {activations_path}")

if not os.path.exists(attentions_path):
    os.makedirs(attentions_path)
    print(f"Created directory {attentions_path}")


tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
model = LlamaForCausalLM.from_pretrained(model_name, device_map="cuda", torch_dtype=torch.bfloat16, token=token)
model.eval()


def get_activations(model: LlamaForCausalLM, text: str, file_name: str, max_new_tokens: int = 10):
    activations_save_path = os.path.join(activations_path, os.path.splitext(file_name)[0] + "_activations.pt")
    attentions_save_path = os.path.join(attentions_path, os.path.splitext(file_name)[0] + "_attentions.pt")

    

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    generate_ids = model.generate(inputs.input_ids, max_new_tokens=max_new_tokens, output_hidden_states=True, output_attentions=True)
    outputs = model(inputs.input_ids, output_hidden_states=True, output_attentions=True)

    answers = tokenizer.decode(generate_ids[0][-max_new_tokens:]).split()
    attentions, activations = outputs["attentions"], outputs["hidden_states"]

    torch.save(activations, activations_save_path)
    torch.save(attentions, attentions_save_path)

    guess = ""
    for answer in answers:
        if answer.strip() in ["A", "B", "C", "D"]:
            guess = answer
            break
    

    return answers, attentions, activations, guess

    
