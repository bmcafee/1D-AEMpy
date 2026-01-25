# Basic usage with defaults
python executor.py \
 --wandb_run_name default \
 --num_epochs 50 \
 --hidden_layers 32 32 \
 --batch_size 1024 \
 --learning_rate 1e-4 \
 --decay_steps 500 \
 --decay_rate 0.01 \

python executor.py \
 --wandb_run_name v1 \
 --num_epochs 50 \
 --hidden_layers 32 32 \
 --batch_size 1024 \
 --learning_rate 0.01 \
 --decay_steps 10 \
 --decay_rate 0.5