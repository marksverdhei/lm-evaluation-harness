# Plan: Run evaluations using:
# /home/me/Repos/lm-evaluation-harness/.venv/bin/lm_eval
alias lm_eval=/home/me/Repos/lm-evaluation-harness/.venv/bin/lm_eval
# Models:
# Datasets: all noreval tasks with configs that work for
#
# lm_eval --model local-chat-completions --tasks nortruthfulqa_gen_nob_p5 --model_args base_url=http://localhost:8000/v1/chat/completions,num_concurrent=1,max_retries=999,tokenized_requests=False,model="GLM4.5-Air-Turbo" --apply_chat_template --gen_kwargs max_tokens=5000,max_completion_tokens=4096 --output_path ./results/ --log_samples --use_cache /home/me/Repos/lm-evaluation-harness/cache
lm_eval --model local-chat-completions --tasks nortruthfulqa_gen_nob_p6 --model_args base_url=http://localhost:8000/v1/chat/completions,num_concurrent=1,max_retries=999,tokenized_requests=False,model="GLM4.5-Air-Turbo" --apply_chat_template --gen_kwargs max_tokens=5000,max_completion_tokens=4096 --output_path ./results/ --log_samples --use_cache /home/me/Repos/lm-evaluation-harness/cache
