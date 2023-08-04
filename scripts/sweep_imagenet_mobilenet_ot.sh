
#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J imagenet-ot
### -- ask for number of cores (default: 1) --
#BSUB -n 8
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 23:59
# request system-memory
#BSUB -R "rusage[mem=10GB]"
### impose machine
#BSUB -R "select[model == XeonGold6242]"
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
#BSUB -o logs/gpu_imagenet_ot.out
#BSUB -e logs/gpu_imagenet_ot.err
# -- end of LSF options --

nvidia-smi
# Load the cuda module
module load cuda/11.6

script_name=`basename "$0"`
EXP_NAME="${script_name%.*}"
echo $EXP_NAME

TARGET_SPARSITYS=(0.6 0.5 0.4 0.3)
SEED=0
GPU=(0)


if [ "$#" -eq 4 ] && [ $4 != test ]; then
    NOWDATE=$4
    is_test=0
else
    NOWDATE="test.$(date +%Y%m%d.%H_%M_%S)"
    is_test=1
fi

#SEED=0

ROOT_DIR=./
DATA_DIR=${ROOT_DIR}/../datasets
WEIGHT_DIR=$ROOT_DIR/prob_regressor_data/
EXP_DIR=$ROOT_DIR/prob_regressor_results/
CODE_DIR='./'

DATASET=imagenet
MODEL=mobilenet
DATA_PATH=/work3/leiyo/imagenet
CONFIG_PATH=./configs/mobilenetv1_optimal_transport.yaml
#PRUNER=woodfisherblockdynamic
PRUNER=optimal_transport
FITTABLE=10000
EPOCHS=5
FISHER_SUBSAMPLE_SIZE=1000
FISHER_MINI_BSZ=1
MAX_MINI_BSZ=1
LOAD_FROM="./checkpoints/MobileNetV1-Dense-STR.pth"
BSZ=256

N_SAMPLE=$(($FISHER_SUBSAMPLE_SIZE * $FISHER_MINI_BSZ))
ARCH_NAME=${MODEL}_${DATASET}_${N_SAMPLE}samples_${FISHER_SUBSAMPLE_SIZE}batches_${SEED}seed
#ARCH_NAME="seed${SEED}_batchsize${BSZ}_fittable${FITTABLE}"
name="sparsity${TARGET}"

LOG_DIR="${EXP_DIR}/${EXP_NAME}/log.${NOWDATE}/$ARCH_NAME"
CSV_DIR="${EXP_DIR}/${EXP_NAME}/csv.${NOWDATE}/$ARCH_NAME"
mkdir -p ${CSV_DIR}
mkdir -p ${LOG_DIR}
RESULT_PATH="${CSV_DIR}/${name}.csv"
LOG_PATH="${LOG_DIR}/${name}.log"

# ONE_SHOT="--one-shot"
SCALE_PRUNE_UPDATE=0.9

OPTIMAL_TRANSPORTATION="--ot"

echo "EXPERIMENT $EXP_NAME"
export PYTHONUNBUFFERED=1

for TARGET in "${TARGET_SPARSITYS[@]}"
do
    args="
    --exp_name=$EXP_NAME \
    --dset=$DATASET \
    --dset_path=$DATA_PATH \
    --arch=$MODEL \
    --config_path=$CONFIG_PATH \
    --workers=4 --batch_size=${BSZ} --logging_level info \
    --pretrained --from_checkpoint_path $LOAD_FROM \
    --batched-test --not-oldfashioned --disable-log-soft --use-model-config \
    --sweep-id 20 --fisher-damp 1e-5 --fisher-subsample-size ${FISHER_SUBSAMPLE_SIZE} --fisher-mini-bsz ${FISHER_MINI_BSZ} --update-config --prune-class $PRUNER \
    --target-sparsity $TARGET \
    --seed ${SEED} --full-subsample --fisher-split-grads --fittable-params $FITTABLE \
    --woodburry-joint-sparsify --offload-inv --offload-grads \
    ${ONE_SHOT} \
    --result-file $RESULT_PATH --epochs $EPOCHS --eval-fast \
    --scale-prune-update ${SCALE_PRUNE_UPDATE} \
    ${OPTIMAL_TRANSPORTATION} \
    "
    if [ "$is_test" -eq 0 ] ; then
        CUDA_VISIBLE_DEVICES=${GPU} python3 ${CODE_DIR}/main.py $args $greedy_args &> $LOG_PATH 2>&1
    else
        CUDA_VISIBLE_DEVICES=${GPU} python3 ${CODE_DIR}/main.py $args $greedy_args
    fi
done