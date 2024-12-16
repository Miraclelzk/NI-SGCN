exp_name=$1
ckpt=$2
dataset=$3
resolution=$4
noise=$5
niters=$6
TT=$7

output_root=../Result/${exp_name}

export PYTHONPATH=$(pwd)/..

python -u test_snn_Noisy.py \
  --ckpt ${ckpt}\
  --output_root ${output_root}\
  --dataset ${dataset}\
  --resolution ${resolution}\
  --noise ${noise}\
  --niters ${niters}\
  --T ${TT}