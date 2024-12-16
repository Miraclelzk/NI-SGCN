exp_name=$1
TT=$2

exp_dir=../Exp/${exp_name}

summary_dir=${exp_dir}/logs

export PYTHONPATH=$(pwd)/..

python -u train_snn_Noisy.py \
  --exp_dir ${exp_dir}\
  --summary_dir ${summary_dir}\
  --T ${TT}