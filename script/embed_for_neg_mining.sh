if [ "$1" ]; then
  CONFIG=$1
else
  echo "MUST PROVIDE A CONFIG"
fi

if [ "$2" ]; then
  echo "USING GPU:$2"
  GPU=$2
else
  echo "DEFAULTING TO GPU:0"
  GPU="0"
fi

cd $VLLM
CUDA_VISIBLE_DEVICES=$GPU python $VLLM/pretrain/embed_training_data.py $VLLM$CONFIG