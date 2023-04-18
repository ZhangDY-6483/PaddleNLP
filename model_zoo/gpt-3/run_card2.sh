export USE_FAST_LN=1
export USE_LINEAR_WITH_GRAD_ADD=1
export load_torch_random_175B_ckpt=False
export NVIDIA_TF32_OVERRIDE=0

rm -rf ./data/*indexmap*

python -m paddle.distributed.launch --gpus 0,1,2,3,4,5,6,7 ./tools/train.py -c ./ppfleetx/configs/nlp/gpt/pretrain_gpt_175B_mp8_pp16.yaml

