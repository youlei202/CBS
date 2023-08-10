##### instructions ####
# first generate commands via `bash filename> > commands.txt` and then execute `bash commands.txt`
#######################

#!/bin/bash
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J resnet20-ot-baseline
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 4:00
# request system-memory
#BSUB -R "rusage[mem=10GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u leiyo@dtu.dk
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o logs/resnet20/many_seeds/many_seeds_std=1_prop=0.1/gpu_resnet20_reg=10_ot_baseline.out
#BSUB -e logs/resnet20/many_seeds/many_seeds_std=1_prop=0.1/gpu_resnet20_reg=10_ot_baseline.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load cuda/11.6

ot_reg=10.0

NOISE_STD=1
NOISE_PROP=0.1

#TARGET_SPARSITYS=(0.2 0.4 0.6 0.7 0.8 0.9 0.1 0.3 0.5 0.98)
TARGET_SPARSITYS=(0.95)
MODULES=("layer1.0.conv1_layer1.0.conv2_layer1.1.conv1_layer1.1.conv2_layer1.2.conv1_layer1.2.conv2_layer2.0.conv1_layer2.0.conv2_layer2.1.conv1_layer2.1.conv2_layer2.2.conv1_layer2.2.conv2_layer3.0.conv1_layer3.0.conv2_layer3.1.conv1_layer3.1.conv2_layer3.2.conv1_layer3.2.conv2")

SEEDS=(0 1 2 3 4)
# SEEDS=(0)
FISHER_SUBSAMPLE_SIZES=(100)
#PRUNERS=(woodfisherblock globalmagni magni diagfisher)
PRUNERS=(optimal_transport)
JOINTS=(1)
FISHER_DAMP="1e-10"
EPOCH_END="10"
PROPER_FLAG="1"
ROOT_DIR="/zhome/b2/8/197929/GitHub/CBS"
DATA_DIR="/zhome/b2/8/197929/GitHub/CBS/datasets"
SWEEP_NAME="exp_oct_21_resnet20_ot_baseline"
NOWDATE=""
DQT='"'
GPUS=(0)
LOG_DIR="${ROOT_DIR}/prob_regressor_results/${SWEEP_NAME}/log/"
CSV_DIR="${ROOT_DIR}/prob_regressor_results/${SWEEP_NAME}/csv/"
mkdir -p ${LOG_DIR}
mkdir -p ${CSV_DIR}
# ONE_SHOT="--one-shot"
CKP_PATH="${ROOT_DIR}/checkpoints/resnet20.ckpt"
# CKP_PATH="${ROOT_DIR}/../exp_root/exp_oct_21_resnet20_all_rep/20211026_18-55-28_694095_239/regular_checkpoint.ckpt"

# OPTIMAL_TRANSPORTATION="--ot"
ADD_NOISE="--add-noise ${NOISE_STD} ${NOISE_PROP}"
REG="--reg ${ot_reg}"
extra_cmd=" ${ONE_SHOT} ${OPTIMAL_TRANSPORTATION}  ${ADD_NOISE} ${REG}"

ID=0

for PRUNER in "${PRUNERS[@]}"
do
    for JOINT in "${JOINTS[@]}"
    do
        if [ "${JOINT}" = "0" ]; then
            JOINT_FLAG=""
        elif [ "${JOINT}" = "1" ]; then
            JOINT_FLAG="--woodburry-joint-sparsify"
        fi

        for SEED in "${SEEDS[@]}"
        do
            for MODULE in "${MODULES[@]}"
            do
                for TARGET_SPARSITY in "${TARGET_SPARSITYS[@]}"
                do
                    for FISHER_SUBSAMPLE_SIZE in "${FISHER_SUBSAMPLE_SIZES[@]}"
                    do
                        if [ "${FISHER_SUBSAMPLE_SIZE}" = 100 ]; then
                            FISHER_MINIBSZS=(1)
                        elif [ "${FISHER_SUBSAMPLE_SIZE}" = 1000 ]; then
                            #FISHER_MINIBSZS=(1 50)
                            FISHER_MINIBSZS=(1)
                        elif [ "${FISHER_SUBSAMPLE_SIZE}" = 4000 ]; then
                            FISHER_MINIBSZS=(1)
                        elif [ "${FISHER_SUBSAMPLE_SIZE}" = 5000 ]; then
                            FISHER_MINIBSZS=(1)
                        elif [ "${FISHER_SUBSAMPLE_SIZE}" = 8000 ]; then
                            FISHER_MINIBSZS=(10)
                        fi

                        for FISHER_MINIBSZ in "${FISHER_MINIBSZS[@]}"
                        do


                            CUDA_VISIBLE_DEVICES=${GPUS[$((${ID} % ${#GPUS[@]}))]} python3 ${ROOT_DIR}/main.py --exp_name=${SWEEP_NAME}  --dset=cifar10 --dset_path=${DATA_DIR} --arch=resnet20 --config_path=${ROOT_DIR}/configs/resnet20_optimal_transport.yaml --workers=1 --batch_size=64 --logging_level debug --gpus=0 --from_checkpoint_path=${CKP_PATH} --batched-test --not-oldfashioned --disable-log-soft --use-model-config --sweep-id ${ID} --fisher-damp ${FISHER_DAMP} --prune-modules ${MODULE} --fisher-subsample-size ${FISHER_SUBSAMPLE_SIZE} --fisher-mini-bsz ${FISHER_MINIBSZ} --update-config --prune-class ${PRUNER} --target-sparsity ${TARGET_SPARSITY} --result-file ${CSV_DIR}/prune_module-woodfisher-based_all_epoch_end-${EPOCH_END}.csv --seed ${SEED} --deterministic --full-subsample ${JOINT_FLAG} --epochs ${EPOCH_END} --fisher-optimized --fisher-parts 5 --offload-grads --offload-inv $extra_cmd 
                            #CUDA_VISIBLE_DEVICES=${GPUS[$((${ID} % ${#GPUS[@]}))]} python3 ${ROOT_DIR}/main.py --exp_name=${SWEEP_NAME}  --dset=cifar10 --dset_path=${DATA_DIR} --arch=resnet20 --config_path=${ROOT_DIR}/configs/resnet20_woodfisher.yaml --workers=1 --batch_size=64 --logging_level debug --gpus=0 --from_checkpoint_path=${CKP_PATH} --batched-test --not-oldfashioned --disable-log-soft --use-model-config --sweep-id ${ID} --fisher-damp ${FISHER_DAMP} --prune-modules ${MODULE} --fisher-subsample-size ${FISHER_SUBSAMPLE_SIZE} --fisher-mini-bsz ${FISHER_MINIBSZ} --update-config --prune-class ${PRUNER} --target-sparsity ${TARGET_SPARSITY} --prune-end ${EPOCH_END} --prune-freq ${EPOCH_END} --result-file ${CSV_DIR}/prune_module-woodfisher-based_all_epoch_end-${EPOCH_END}.csv --seed ${SEED} --deterministic --full-subsample ${JOINT_FLAG} --fisher-optimized --fisher-parts 5 --offload-grads --offload-inv $extra_cmd &> ${LOG_DIR}/${PRUNER}_proper-1_joint-${JOINT}_module-all_target_sp-${TARGET_SPARSITY}_epoch_end-${EPOCH_END}_samples-${FISHER_SUBSAMPLE_SIZE}_${FISHER_MINIBSZ}_damp-${FISHER_DAMP}_seed-${SEED}.txt

                            ID=$((ID+1))

                        done
                    done
                done
            done
        done
    done
done
