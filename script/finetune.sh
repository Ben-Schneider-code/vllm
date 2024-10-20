GPU=$1
PORT=$2
CONFIG=$3
deepspeed --include localhost:$GPU --master_port $PORT $VLLM/internvl/train/internvl_chat_finetune.py $VLLM$CONFIG