exp_name=NI-HSGCN-T4
ckpt=../Exp/NI-HSGCN-T4/checkpoint/ckpt_0.000361_974000.pt
TT=4

# # PUNet dataset 10K Points 
# bash test_snn.sh ${exp_name} ${ckpt} PUNet 10000_poisson 0.01 1 ${TT}
# bash test_snn.sh ${exp_name} ${ckpt} PUNet 10000_poisson 0.02 1 ${TT}
# bash test_snn.sh ${exp_name} ${ckpt} PUNet 10000_poisson 0.03 1 ${TT}
# # PUNet dataset 50K Points 
# bash test_snn.sh ${exp_name} ${ckpt} PUNet 50000_poisson 0.01 1 ${TT}
# bash test_snn.sh ${exp_name} ${ckpt} PUNet 50000_poisson 0.02 1 ${TT}
# bash test_snn.sh ${exp_name} ${ckpt} PUNet 50000_poisson 0.03 1 ${TT}
# PCNet dataset 10K Points 
bash test_snn.sh ${exp_name} ${ckpt} PCNet 10000_poisson 0.01 1 ${TT}
bash test_snn.sh ${exp_name} ${ckpt} PCNet 10000_poisson 0.02 1 ${TT}
bash test_snn.sh ${exp_name} ${ckpt} PCNet 10000_poisson 0.03 1 ${TT}
# PCNet dataset 50K Points 
bash test_snn.sh ${exp_name} ${ckpt} PCNet 50000_poisson 0.01 1 ${TT}
bash test_snn.sh ${exp_name} ${ckpt} PCNet 50000_poisson 0.02 1 ${TT}
bash test_snn.sh ${exp_name} ${ckpt} PCNet 50000_poisson 0.03 1 ${TT}

# # PUNet dataset 10K Points 
# bash test_snn.sh ${exp_name} ${ckpt} PUNet 10000_poisson 0.01 2 ${TT}
# bash test_snn.sh ${exp_name} ${ckpt} PUNet 10000_poisson 0.02 2 ${TT}
# bash test_snn.sh ${exp_name} ${ckpt} PUNet 10000_poisson 0.03 2 ${TT}
# # PUNet dataset 50K Points 
# bash test_snn.sh ${exp_name} ${ckpt} PUNet 50000_poisson 0.01 2 ${TT}
# bash test_snn.sh ${exp_name} ${ckpt} PUNet 50000_poisson 0.02 2 ${TT}
# bash test_snn.sh ${exp_name} ${ckpt} PUNet 50000_poisson 0.03 2 ${TT}
# PCNet dataset 10K Points 
bash test_snn.sh ${exp_name} ${ckpt} PCNet 10000_poisson 0.01 2 ${TT}
bash test_snn.sh ${exp_name} ${ckpt} PCNet 10000_poisson 0.02 2 ${TT}
bash test_snn.sh ${exp_name} ${ckpt} PCNet 10000_poisson 0.03 2 ${TT}
# PCNet dataset 50K Points 
bash test_snn.sh ${exp_name} ${ckpt} PCNet 50000_poisson 0.01 2 ${TT}
bash test_snn.sh ${exp_name} ${ckpt} PCNet 50000_poisson 0.02 2 ${TT}
bash test_snn.sh ${exp_name} ${ckpt} PCNet 50000_poisson 0.03 2 ${TT}

# # PUNet dataset 10K Points 
# bash test_snn.sh ${exp_name} ${ckpt} PUNet 10000_poisson 0.01 3 ${TT}
# bash test_snn.sh ${exp_name} ${ckpt} PUNet 10000_poisson 0.02 3 ${TT}
# bash test_snn.sh ${exp_name} ${ckpt} PUNet 10000_poisson 0.03 3 ${TT}
# # PUNet dataset 50K Points 
# bash test_snn.sh ${exp_name} ${ckpt} PUNet 50000_poisson 0.01 3 ${TT}
# bash test_snn.sh ${exp_name} ${ckpt} PUNet 50000_poisson 0.02 3 ${TT}
# bash test_snn.sh ${exp_name} ${ckpt} PUNet 50000_poisson 0.03 3 ${TT}
# PCNet dataset 10K Points 
bash test_snn.sh ${exp_name} ${ckpt} PCNet 10000_poisson 0.01 3 ${TT}
bash test_snn.sh ${exp_name} ${ckpt} PCNet 10000_poisson 0.02 3 ${TT}
bash test_snn.sh ${exp_name} ${ckpt} PCNet 10000_poisson 0.03 3 ${TT}
# PCNet dataset 50K Points 
bash test_snn.sh ${exp_name} ${ckpt} PCNet 50000_poisson 0.01 3 ${TT}
bash test_snn.sh ${exp_name} ${ckpt} PCNet 50000_poisson 0.02 3 ${TT}
bash test_snn.sh ${exp_name} ${ckpt} PCNet 50000_poisson 0.03 3 ${TT}