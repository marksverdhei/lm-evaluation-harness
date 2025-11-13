lm_eval \
  --model local-chat-completions \
  --tasks nortruthfulqa_gen_nob_p5 \
  --model_args base_url=http://localhost:8000/v1/chat/completions,num_concurrent=1,max_retries=999,tokenized_requests=False,model="Q3-Coder-Fast" \
  --apply_chat_template \
  --gen_kwargs temperature=0.7,top_k=20,top_p=0.8,max_tokens=5000,max_completion_tokens=4096 \
  --output_path ./results/Q3-Coder \
  --log_samples


lm_eval \
  --model local-chat-completions \
  --tasks nortruthfulqa_gen_nob_p5 \
  --model_args base_url=http://localhost:8000/v1/chat/completions,num_concurrent=1,max_retries=999,tokenized_requests=False,model="Q3-Coder" \
  --apply_chat_template \
  --gen_kwargs temperature=0.7,top_k=20,top_p=0.8,max_tokens=5000,max_completion_tokens=4096 \
  --output_path ./results/Q3-Coder \
  --log_samples
