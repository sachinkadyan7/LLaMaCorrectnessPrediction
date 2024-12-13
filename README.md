# LLama_behavior_classification
Study into Llama-3's ability to reason and our ability to classify its behaviors from the hidden states and attention maps


## Data

We will gather data from the MMLU dataset as well as the ARC-AGI dataset, and preprocess it into multiple choice, one-token answer formats. 

- MMLU (almost ready)
- ARC-AGI (not started)

TODO:
- clean up structure of the data into one train, val, test datasets ready for inference. 
- Process ARC-AGI dataset into proper format. 


## Activations

We will keep the hidden states and the attention maps for analysis. The model must output a single token response for the activations to make any sense. 

- hidden states (done)
- attention maps (done)

TODO: 
- Setup inference mechanism with capture of the hidden states and attention maps



Issues:
![alt text](image.png)

How to handle the fact that capital letters like A, B, C, D have multiple token representations? Seems like we can't filter the output probs by specific dict entries. Unless we know for sure that " A" is the correct output token instead of "A" for Llama. 


- in bfloat16 (2 bytes), a set of attentions relating to a single model pass takes the form of a tuple of 32 tensors (1 per layer) which have shape (batch_size=1, num_heads=32, seq_len, seq_len). So, for a seq of len 134 in my case its a total object of 32 * 32 * 134 * 134 elements each of size 2 bytes, so 36773888 (36 Mb). Huge...
- extend that to dataset of 500 elements: 18386944000 18 GB

- in bfloat16 (2 bytes), a set of activations relating to a single model pass takes the form of a tuple of 32 tensors (1 per layer) which have shape (batch_size=1, num_heads=32, seq_len, seq_len). So, for a seq of len 134 in my case its a total object of 32 * 32 * 134 * 134 elements each of size 2 bytes, so 36773888 (36 Mb). Huge...
- extend that to dataset of 500 elements: 18386944000 18 GB


Activations Experiment results:
layer distribution: 25 % each exactly. Model achieves 100% separability
correctness: model acieves 81% accuracy. Real distribution is 75%.