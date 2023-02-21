export CUDA_VISIBLE_DEVICES=1

nohup accelerate launch train_plain.py \
  --pretrained_model_name_or_path="stabilityai/stable-diffusion-2-1-base" \
  --dataset_name="jasonchoi3/nordstrom96568" --caption_column="text" \
  --resolution=512 --random_flip \
  --train_batch_size=4 \
  --max_train_steps=400000 --checkpointing_steps=20000 \
  --learning_rate=1e-5 --lr_scheduler="constant" --lr_warmup_steps=0 \
  --seed=42 \
  --output_dir="models" --mixed_precision="fp16" \
  --report_to="wandb" \
  --local_rank=0  --resume_from_checkpoint="latest" &