# TODO: add use_cache
#
lm_eval --model local-chat-completions --tasks nortruthfulqa_gen_nob_p5 --model_args base_url=http://localhost:8000/v1/chat/completions,num_concurrent=1,max_retries=999,tokenized_requests=False,model="GLM4.5-Air-Turbo" --apply_chat_template --gen_kwargs max_tokens=5000,max_completion_tokens=4096 --output_path ./results/ --log_samples --use_cache /home/me/Repos/lm-evaluation-harness/cache
#
#
lm_eval --model local-chat-completions --tasks nortruthfulqa_gen_nob_p5 --model_args base_url=http://localhost:8000/v1/chat/completions,num_concurrent=1,max_retries=999,tokenized_requests=False,model="GLM4.5-Air" --apply_chat_template --gen_kwargs max_tokens=5000,max_completion_tokens=4096 --output_path ./results/ --log_samples
#

lm_eval --model local-chat-completions --tasks nortruthfulqa_gen_nob_p5 --model_args base_url=http://localhost:8000/v1/chat/completions,num_concurrent=1,max_retries=999,tokenized_requests=False,model="GLM4.6" --apply_chat_template --gen_kwargs max_tokens=5000,max_completion_tokens=4096 --output_path ./results/ --log_samples



lm_eval --model local-chat-completions --tasks nortruthfulqa_gen_nob_p5 --model_args base_url=http://localhost:8000/v1/chat/completions,num_concurrent=1,max_retries=999,tokenized_requests=False,model="GPTOSS-120B" --apply_chat_template --gen_kwargs max_tokens=5000,max_completion_tokens=4096 --output_path ./results/ --log_samples
#
lm_eval --model local-chat-completions --tasks nortruthfulqa_gen_nob_p5 --model_args base_url=http://localhost:8000/v1/chat/completions,num_concurrent=1,max_retries=999,tokenized_requests=False,model="GPTOSS-20B" --apply_chat_template --gen_kwargs max_tokens=5000,max_completion_tokens=4096 --output_path ./results/ --log_samples

lm_eval --model local-chat-completions --tasks nortruthfulqa_gen_nob_p5 --model_args base_url=http://localhost:8000/v1/chat/completions,num_concurrent=1,max_retries=999,tokenized_requests=False,model="Q3-Thinking" --apply_chat_template --gen_kwargs temperature=0.6,top_k=20,top_p=0.95,max_tokens=5000,max_completion_tokens=4096 --output_path ./results/Q3-Thinking --log_samples

lm_eval --model local-chat-completions --tasks nortruthfulqa_gen_nob_p5 --model_args base_url=http://localhost:8000/v1/chat/completions,num_concurrent=1,max_retries=999,tokenized_requests=False,model="Q3-Instruct" --apply_chat_template --gen_kwargs temperature=0.7,top_k=20,top_p=0.8,max_tokens=5000,max_completion_tokens=4096 --output_path ./results/Q3-Instruct --log_samples

lm_eval --model local-chat-completions --tasks nortruthfulqa_gen_nob_p5 --model_args base_url=http://localhost:8000/v1/chat/completions,num_concurrent=1,max_retries=999,tokenized_requests=False,model="Q3-Coder" --apply_chat_template --gen_kwargs temperature=0.7,top_k=20,top_p=0.8,max_tokens=5000,max_completion_tokens=4096 --output_path ./results/Q3-Coder --log_samples

lm_eval --model local-chat-completions --tasks nortruthfulqa_gen_nob_p5 --model_args base_url=http://localhost:8000/v1/chat/completions,num_concurrent=1,max_retries=999,tokenized_requests=False,model="Q3-Big" --apply_chat_template --gen_kwargs temperature=0.6,top_k=20,top_p=0.95,max_tokens=5000,max_completion_tokens=4096 --output_path ./results/Q3-Big --log_samples
