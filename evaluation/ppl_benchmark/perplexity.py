"""
Adapted from https://github.com/mit-han-lab/streaming-llm

Note: Although this script measures latency, it is not optimized whatsoever!
The latency is only tracked to see the impact of speed over time.

Usage:

python benchmark/perplexity.py --experiment attention_sinks
python benchmark/perplexity.py --experiment transformers
python benchmark/perplexity.py --experiment windowed
"""

import sys
import os
import argparse
import itertools
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import pandas as pd
import torch
from datasets import load_dataset
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM

# calculate perplexity using a sliding window
def compute_perplexity_sliding_window(
    model,
    encodings,
    context_max_length: Optional[int] = 2048,
    stride: Optional[int] = 512,
) -> float:

    # For long sequences, we process in chunks
    seq_len = encodings.input_ids.size(1)

    nlls = []
    prev_end_loc = 0
    start_t = time.time()
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + context_max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(model.device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss
        nlls.append(neg_log_likelihood)


        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    ppl = float(torch.exp(torch.stack(nlls).mean()).item())
    gpu_memory = torch.cuda.memory_allocated(1) / 1024 / 1024 / 1024  # in GB
    end_t = time.time()

    print(f"Time taken: {end_t - start_t:.2f} seconds")
    print(f"PPL: {ppl:.2f}")
    print(f"GPU Memory Used: {gpu_memory:.2f} GB")

    return ppl




# calculate perplexity one token by one token for one dataset
def compute_perplexity_per_token(
    model,
    tokenizer,
    dataset,
    experiment: str,
    output_dir: str = "outputs",
    data_column: str = "text",
    num_samples: int = 1,
    num_tokens: Optional[int] = None,
    overwrite: bool = False,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{experiment}.csv"

    if output_file.exists() and not overwrite:
        raise ValueError(
            f"The {output_file!r} output file already exists - if you really want to override it, then use `--overwrite`."
        )

    logs = defaultdict(list)
    loss_fn = CrossEntropyLoss(reduction="none")
    past_key_values = None
    num_processed_tokens = 0
    for text in itertools.islice(dataset, num_samples):
        encodings = tokenizer(text[data_column], return_tensors="pt")

        seq_len = encodings.input_ids.size(1)
        print(f"sequence length: {seq_len}")
        pbar = tqdm(range(0, seq_len - 1))

        for idx in pbar:
            start_t = time.time()
            input_ids = encodings.input_ids[:, idx : idx + 1].to(model.device)
            with torch.no_grad():
                outputs = model(input_ids, past_key_values=past_key_values, use_cache=True)
                logits = outputs.logits.view(-1, model.config.vocab_size)
                past_key_values = outputs.past_key_values
                label = encodings.input_ids[:, idx + 1 : idx + 2].to(logits.device).view(-1)
                neg_log_likelihood = loss_fn(logits, label)
                perplexity = neg_log_likelihood.exp()
            pbar.set_description(f"nll: {neg_log_likelihood.item():>5.2f}, ppl: {perplexity.item():>8.2f}")

            # Store data and save every 10 tokens
            logs["input_length"].append(idx + 1)
            logs["nll"].append(neg_log_likelihood.item())
            logs["ppl"].append(perplexity.item())
            logs["overall_ppl"].append(torch.tensor(logs["nll"]).mean().exp().item())
            logs["cuda_vram_allocated"].append(torch.cuda.memory_allocated(0) / 1024 / 1024 / 1024)  # in GB
            logs["latency"].append(time.time() - start_t)
            if num_processed_tokens % 10 == 0:
                try:
                    pd.DataFrame(logs).to_csv(output_file, index=False)
                except KeyboardInterrupt as ex:
                    # If there's a Keyboard Interrupt, still write the file, and then stop
                    pd.DataFrame(logs).to_csv(output_file, index=False)
                    raise ex

            num_processed_tokens += 1
            if num_tokens and num_processed_tokens >= num_tokens:
                return
    

def main():
    parser = argparse.ArgumentParser()
    # Which experiment to run?
    parser.add_argument(
        "--experiment", choices=["transformers", "self_extended"], default="transformers"
    )

    # Model args
    parser.add_argument("--model_name_or_path", type=str, default="meta-llama/Llama-2-7b-hf")
    parser.add_argument("--revision", type=str, default="main")
    parser.add_argument("--trust_remote_code", action="store_true")

    # Dataset args
    parser.add_argument("--dataset_name", type=str, default="emozilla/pg19-test")
    parser.add_argument("--data_column", type=str, default="text")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--split", type=str, default="test", choices=["validation", "test"])
    # parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--num_tokens", type=int, default=8192)

    # Where to log
    parser.add_argument("--output_dir", type=str, default="benchmark/outputs")
    parser.add_argument("--overwrite", action="store_true")


    # Self Extended settings
    parser.add_argument("--group_size", type=int, default=32)
    parser.add_argument("--use_flash", type=bool, default=True)
    parser.add_argument("--window_size", type=int, default=1024)

    args = parser.parse_args()

    # Initialize the model,  via self_extended, transformers or via attention_sinks
    if  args.experiment == "self_extended":
        # Get the absolute path of the grandparent directory
        grandparent_dir = os.path.abspath(os.path.join(os.getcwd(), '..', '..'))
        # Add the grandparent directory to the Python path
        sys.path.insert(0, grandparent_dir)
        import SelfExtend
        
        model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2")
        SelfExtend.apply(model, args.group_size, args.window_size, enable_flash_attention=args.use_flash) 
    else: 
             
        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            revision=args.revision,
            trust_remote_code=bool(args.trust_remote_code),
            torch_dtype=torch.float16,
            # attn_implementation="flash_attention_2",
            max_position_embeddings = 16384,
        )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, trust_remote_code=bool(args.trust_remote_code))

    # Set up the dataset
    dataset = load_dataset(args.dataset_name, args.task, split=args.split, streaming=True)

    compute_perplexity_per_token(
        model,
        tokenizer,
        dataset,
        args.experiment,
        output_dir=args.output_dir,
        data_column=args.data_column,
        num_samples=1,  # <- No support for more than one instance now
        num_tokens=args.num_tokens,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
