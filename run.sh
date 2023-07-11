#!/bin/bash
CUDA_VISIBLE_DEVICES=0 python main.py --cuda \
	--data_path data/FB15k-237-betae \
	--do_train --do_test \
	-n 128 -b 512 -d 400 -g 20 \
	-lr 0.00005 --max_steps 350001 --cpu_num 2 --geo scone --valid_steps 30000 \
	-projm "(1600,2)" --save_checkpoint_steps 30000 -logic "geometry" \
	--seed 0 --print_on_screen -p 0.9 -projn "rtrans_mlp" -conj "all" -delta 0.5

# CUDA_VISIBLE_DEVICES=0 python main.py --cuda \
	# --data_path data/NELL-betae \
	# --do_train --do_test \
	# -n 128 -b 512 -d 400 -g 20 \
	# -lr 0.00005 --max_steps 350001 --cpu_num 2 --geo scone --valid_steps 30000 \
	# -projm "(1600,2)" --save_checkpoint_steps 30000 -logic "geometry" \
	# --seed 0 --print_on_screen -p 0.9 -projn "rtrans_mlp" -conj "all" -delta 0.5

# CUDA_VISIBLE_DEVICES=0 python main.py --cuda \
	# --data_path data/FB15k-betae \
	# --do_train --do_test \
	# -n 128 -b 512 -d 400 -g 30 \
	# -lr 0.00005 --max_steps 450001 --cpu_num 2 --geo scone --valid_steps 30000 \
	# -projm "(1600,2)" --save_checkpoint_steps 30000 -logic "geometry" \
	# --seed 0 --print_on_screen -p 0.9 -projn "rtrans_mlp" -conj "all" -delta 0.5
