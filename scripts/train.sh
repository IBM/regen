#!/bin/bash

set -o errexit
set -o nounset
set -o pipefail

# handles paths
scp_path=$(realpath -m $0)
scp_dir=$(dirname $scp_path)
pwd_dir=$(pwd -P)

ver='1'

# find the repos root
root=$(git rev-parse --show-toplevel)
pushd $root > /dev/null

# usage function
function usage()
{
    local scpt=$(basename $0)
    echo
    echo "usage: $scpt -j <job_id> -d <dataset> -x <direction>"
    echo "ver: $ver"
    cat <<'EOF'
    Options:
-h                    help
-j <job_id >          train model for job <job_id> w/ job_id = 1, 2, ...
                      In default setting, a job_id is similar to an epoch number
-x <direction>        direction of generation: 'g2t' (graph-to-text) or 't2g' (text-to-graph)
-d <dataset>          dataset to use 'webnlg' or 'tekgen'
-t <model_type>       training type: 'ce' for cross entropy, 'scst' for reinforcement learning
                      instead.
EOF
}

job_id='unknown'
dataset='unknown'
direction='unknown'
mtype='unknown'

while getopts "hj:d:t:x:" options; do
    case $options in
        h ) echo "help requested"
            usage
            exit 0;;
        d ) dataset="$OPTARG";;
        j ) job_id="$OPTARG";;
        t ) mtype="$OPTARG";;
        x ) direction="$OPTARG";;
        \? ) usage
            exit 0;;
        * ) usage
            exit 1;;
    esac
done

# echo "@ job_id $job_id dataset $dataset direction $direction"

if [[ $job_id == "unknown" ]]
then
    echo
    echo "# error: you must specify a job_id !"
    usage
    exit 1
fi

if [[ $dataset == "unknown" ]]
then
    echo
    echo "# error: you must specify a dataset !"
    usage
    exit 1
fi

if [[ $direction == "direction" ]]
then
    echo
    echo "# error: you must specify a direction !"
    usage
    exit 1
fi

if [[ $mtype == "unknown" ]]
then
    echo
    echo "# error: you must specify a model type !"
    usage
    exit 1
fi



## Checks

# check job_id (is it an integer)
re_int='^[0-9]+$'
if ! [[ $job_id =~ $re_int ]]
then
    echo
    echo "# error: job_id is not a positive integer"
    usage
    exit 1
fi


# check dataset
unk='unknown'
case $dataset in

    webnlg | tekgen)
        : echo "dataset: $dataset"
        ;;
    *)
        # echo "unk $arg"sd
        unk=$dataset
        ;;
esac

if [ "$unk" != "unknown" ]
then
    echo "# error: invalid dataset arg -- $unk"
    exit 1
fi


# check direction
unk='unknown'
case $direction in

    g2t | t2g)
        : echo "direction: $direction"
        ;;
    *)
        # echo "unk $arg"sd
        unk=$direction
        ;;
esac

if [ "$unk" != "unknown" ]
then
    echo "# error: invalid direction arg -- $unk"
    exit 1
fi

# check model type
unk='unknown'
case $mtype in

    ce | scst)
        : echo "model type: $mtype"
        ;;
    *)
        # echo "unk $arg"
        unk=$mtype
        ;;
esac

if [[ $unk != "unknown" ]]
then
    echo "# error: invalid model type arg -- $unk"
    exit 1
fi


# You need GPU(s) for training
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]
then
    echo "# Cannot find cuda devices visible -- CUDA_VISIBLE_DEVICES is unset"
    exit 1
fi

# manage GPUs
echo "# host: $(hostname --long)"
num_gpus=$(( $(echo "$CUDA_VISIBLE_DEVICES" | tr -cd , | wc -c) + 1 ))
world_size=$num_gpus
num_procs=$(nproc)
echo "# cuda: " $(printenv | grep "CUDA_VISIBLE_DEVICES")

echo "# num procs:  $num_procs"
echo "# num gpus:   $num_gpus"
echo "# world size: $world_size"
echo "# jobid :     $job_id"

# master port
randomdigits=${RANDOM}
master_port=49${randomdigits:(-3)}
echo "# random ${randomdigits} -> 3digits ${randomdigits:(-3)} -> master_port $master_port"

PYTHONPATH=""
unset XDG_RUNTIME_DIR
WANDB_SILENT=true


set -x
python -m torch.distributed.launch \
       --nproc_per_node=$num_gpus \
       --master_addr 127.0.0.1 \
       --master_port $master_port \
       train.py \
       --world_size  $world_size \
       --jid         $job_id \
       --config      ./cfg/train.$dataset.$direction.$mtype.cfg
set +x

popd > /dev/null
