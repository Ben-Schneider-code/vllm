GPU=$1
PORT=$2
CONFIG=$3
deepspeed --include localhost:$GPU --master_port $PORT $VLLM/llava/llava_train.py $VLLM$CONFIG