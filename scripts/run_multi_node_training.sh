# Multi-node distributed training. Run from repo root: bash scripts/run_multi_node_training.sh [node_rank] [master_addr] [nproc_per_node] [training params...]
set -e
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$REPO_ROOT"

NODE_RANK=${1:-0}
MASTER_ADDR=${2:-localhost}
NPROC_PER_NODE=${3:-8}
shift 3

if [ "$MASTER_ADDR" = "localhost" ] && [ "$NODE_RANK" != "0" ]; then
    echo "Error: The slave node must specify the master node IP address"
    echo "Usage: bash scripts/run_multi_node_training.sh <node_rank> <master_addr> <nproc_per_node> [training parameters...]"
    exit 1
fi

export MASTER_ADDR=$MASTER_ADDR
export MASTER_PORT=${MASTER_PORT:-12356}
export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NPROC_PER_NODE-1)))

NNODES=1
WORLD_SIZE=$((NNODES * NPROC_PER_NODE))

echo "=========================================="
echo "Multi-node distributed training configuration"
echo "=========================================="
echo "Repo root: $REPO_ROOT"
echo "Node rank (node_rank): $NODE_RANK"
echo "Master node address (master_addr): $MASTER_ADDR"
echo "Master node port (master_port): $MASTER_PORT"
echo "Number of nodes (nnodes): $NNODES"
echo "Number of GPUs per node (nproc_per_node): $NPROC_PER_NODE"
echo "Total number of GPUs (world_size): $WORLD_SIZE"
echo "Visible GPUs: $CUDA_VISIBLE_DEVICES"
echo "Training parameters: $*"
echo "=========================================="
echo ""

if [ "$NODE_RANK" = "0" ]; then
    echo "Master node started, waiting for slave nodes to connect..."
    echo "Please ensure that the slave nodes have been started and can access the master node $MASTER_ADDR:$MASTER_PORT"
else
    echo "Slave node started, connecting to the master node $MASTER_ADDR:$MASTER_PORT"
    if ! nc -z $MASTER_ADDR $MASTER_PORT 2>/dev/null; then
        echo "Warning: Unable to connect to the master node $MASTER_ADDR:$MASTER_PORT"
        echo "Please check:"
        echo "  1. Whether the master node has been started"
        echo "  2. Whether the firewall allows port $MASTER_PORT"
        echo "  3. Whether the network is connected"
    fi
fi


echo ""
echo "Starting multi-node distributed training..."
echo ""
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --nproc_per_node=$NPROC_PER_NODE \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    scripts/train.py \
    --batch_size 2 \
    --cache_path cache/laptop_eot_token512 \
    --init_mode resume_from_ckpt \
    --checkpoint_path checkpoints/lr1e-5_bs2_ep500/epoch_500.pth \
    --learning_rate 1e-5 \
    --max_epochs 200 \
    --save_interval 100 \
    --use_wandb True \
    --save_optimizer False \
    --save_checkpoint_dir checkpoints \
    --train_urdf_params True \
    --train_eot True \
    "$@"

echo ""
echo "Distributed training completed"

# Example with multiple caches:
# torchrun \
#     --nnodes=$NNODES \
#     --node_rank=$NODE_RANK \
#     --nproc_per_node=$NPROC_PER_NODE \
#     --master_addr=$MASTER_ADDR \
#     --master_port=$MASTER_PORT \
#     scripts/train.py \
#     --batch_size 2 \
#     --cache_path cache-dinov3-h-normalize/dishwasher_eot_token512 cache-dinov3-h-normalize/display_eot_token512 cache-dinov3-h-normalize/door_eot_token512 cache-dinov3-h-normalize/faucet_eot_token512 cache-dinov3-h-normalize/knife_eot_token512 cache-dinov3-h-normalize/laptop_eot_token512 cache-dinov3-h-normalize/microwave_eot_token512 cache-dinov3-h-normalize/refrigerator_eot_token512 cache-dinov3-h-normalize/scissors_eot_token512 cache-dinov3-h-normalize/storagefurniture_eot_token512 \
#     --init_mode train_from_scratch \
#     --learning_rate 1e-5 \
#     --max_epochs 200 \
#     --save_interval 50 \
#     --use_wandb True \
#     --save_optimizer True \
#     --save_checkpoint_dir checkpoints \
#     "$@"
