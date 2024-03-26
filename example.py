# transfromers version 4.38.2
import warnings
warnings.filterwarnings("ignore")

import torch 
import json
import time
from transformers.models.llama.modeling_llama import LlamaAttention
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

import SelfExtend 


window_size = 1024
group_size = 32
use_flash = True

# model_lists = ['google/gemma-7b-it', 'meta-llama/Llama-2-7b-chat-hf', 'mistralai/Mistral-7B-Instruct-v0.1', ]
model_lists = ['meta-llama/Llama-2-7b-chat-hf']

# Set the device to the first GPU if available
# Make sure CUDA is available and you have GPUs
if torch.cuda.is_available():
    # Specify the first GPU
    device = torch.device("cuda:0")
else:
    # Fallback to CPU if CUDA is not available
    device = torch.device("cpu")


for model_name in model_lists:
    if 'Mistral' in model_name:
        # Disable Mistral's sliding window
        config = AutoConfig.from_pretrained(model_name)
        config.sliding_window = None
        # model = AutoModelForCausalLM.from_pretrained(model_name, config=config, device_map="auto", torch_dtype=torch.bfloat16, use_flash_attention_2=use_flash)
        model = AutoModelForCausalLM.from_pretrained(model_name, config=config, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16,attn_implementation="flash_attention_2")
        # model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.bfloat16, use_flash_attention_2=use_flash)
    
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model.eval()
    file_name = "./passkey_examples.jsonl"

    print("=========="*2 + "**Original**" + "=========="*2)
    for line in open(file_name, "r"):
        example = json.loads(line)
        prompt_postfix = "What is the pass key? The pass key is "
        prompt = example["input"] + prompt_postfix
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        print( "-----------------------------------" )
        print( f"#Tokens of Prompt:", input_ids.shape[1], end=" " )
        print( "Passkey target:", example["target"] )

        start_time = time.time()
        tokens = model.generate(input_ids, max_new_tokens=len(example["target"]))
        end_time = time.time()
        answer = prompt_postfix + tokenizer.decode(tokens[0].tolist()[input_ids.shape[1]:], skip_special_tokens=True)
        answer = answer.replace("\n", "\\n")
        answer= f"{model_name}:\n     [ {answer} ]"
        print( answer )
        print( f"Runing Time: {end_time - start_time:.2f} sec" )
        print( "-----------------------------------\n" )

    
    print("=========="*2 + "**SelfExtend**" + "=========="*2)
    SelfExtend.apply(model, group_size, window_size, enable_flash_attention=use_flash)
    for line in open(file_name, "r"):
        example = json.loads(line)
        prompt_postfix = "What is the pass key? The pass key is "
        prompt = example["input"] + prompt_postfix
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids
        input_ids = input_ids.to(device)
        print( f"#Tokens of Prompt:", input_ids.shape[1], end=" " )
        print( "Passkey target:", example["target"] )

        start_time = time.time()
        tokens = model.generate(input_ids, max_new_tokens=len(example["target"]))
        end_time = time.time()
        answer = prompt_postfix + tokenizer.decode(tokens[0].tolist()[input_ids.shape[1]:], skip_special_tokens=True)
        answer = answer.replace("\n", "\\n")
        answer= f"SelfExtended-{model_name}:\n     [ {answer} ]"
        print( answer )
        print( f"Runing Time: {end_time - start_time:.2f} sec" )
        print( "-----------------------------------\n" )
