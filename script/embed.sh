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


if [ "$3" ]; then
  echo "USING TOP-K DEPTH OF $3"
  TOPK=$3
else
  echo "USING DEFAULT TOP-K DEPTH OF 10"
  TOPK="10"
fi

cd $VLLM
CUDA_VISIBLE_DEVICES=$GPU python $VLLM/visualization/embed_.py $VLLM$CONFIG $TOPK