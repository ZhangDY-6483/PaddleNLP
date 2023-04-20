source /root/paddlejob/workspace/env_run/gpt_benchmark/env/gpt_benchmark/bin/activate
export USE_FAST_LN=1
export USE_LINEAR_WITH_GRAD_ADD=1
export load_torch_random_175B_ckpt=True
export NVIDIA_TF32_OVERRIDE=0

rm -rf ./data/*indexmap*
rm -rf ./log/*

DISTRIBUTED_ARGS="--nnodes=2 --master=10.95.147.140:36712"

python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 $DISTRIBUTED_ARGS ./tools/train.py -c ./ppfleetx/configs/nlp/gpt/pretrain_gpt_10B_mp8_pp2.yaml

