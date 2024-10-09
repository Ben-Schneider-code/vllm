echo "USE GPU:$2 TO EMBED CONFIG $1"
cd $VLLM
deepspeed --include localhost:$2 $VLLM/visualization/embed.py $VLLM/config/embed/$1