CKPT_PATH=checkpoints/lr1e-5_bs2_ep200_eot_urdf-params/epoch_200.pth # TODO: change to your checkpoint path

# python scripts/inference.py \
#     --image_path assets/display_2.png \
#     --model_path $CKPT_PATH \
#     --whole_mesh_path assets/display_2.obj \
#     --output_dir outputs/display_2 \
#     --seed 42

# python scripts/inference.py \
#     --image_path assets/refrigerator.png \
#     --model_path $CKPT_PATH \
#     --whole_mesh_path assets/refrigerator.obj \
#     --output_dir outputs/refrigerator \
#     --seed 41

# python scripts/inference.py \
#     --image_path assets/knife.png \
#     --model_path $CKPT_PATH \
#     --in_the_wild \
#     --output_dir outputs/knife \
#     --seed 42

# python scripts/inference.py \
#     --image_path assets/faucet.png \
#     --model_path $CKPT_PATH \
#     --in_the_wild \
#     --output_dir outputs/faucet \
#     --seed 42

# python scripts/inference.py \
#     --image_path assets/display.jpg \
#     --model_path $CKPT_PATH \
#     --in_the_wild \
#     --output_dir outputs/display \
#     --seed 42

# python scripts/inference.py \
#     --image_path assets/laptop.jpg \
#     --model_path $CKPT_PATH \
#     --in_the_wild \
#     --output_dir outputs/laptop \

# python scripts/inference.py \
#     --image_path assets/faucet_2.png \
#     --model_path $CKPT_PATH \
#     --whole_mesh_path assets/faucet_2.obj \
#     --output_dir outputs/faucet_2 \
#     --seed 42

# python scripts/inference.py \
#     --image_path assets/microwave.png \
#     --model_path $CKPT_PATH \
#     --whole_mesh_path assets/microwave.obj \
#     --output_dir outputs/microwave \
#     --seed 42