python benchmark/perplexity.py --experiment transformers --output_dir ppl_benchmark/outputs_llama_2_7b --overwrite --num_tokens 16384
python benchmark/perplexity.py --experiment self_extended --output_dir ppl_benchmark/outputs_llama_2_7b --overwrite --num_tokens 16384
python /home/jr151/code/LongLM/evaluation/ppl_benchmark/perplexity.py --experiment self_extended_sw --output_dir /home/jr151/code/LongLM/evaluation/ppl_benchmark/outputs/llama_2_7b --overwrite --num_tokens 16384  --stride 256 --device 2

python /home/jr151/code/LongLM/evaluation/ppl_benchmark/plot_perplexity.py --features perplexity vram --title "Log perplexity & VRAM usage of Llama 2 7B as a function of input lengths" --output_dir /home/jr151/code/LongLM/evaluation/ppl_benchmark/outputs --log_perplexity_limit 4