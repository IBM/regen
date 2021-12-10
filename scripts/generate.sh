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
    echo "usage: $scpt [-h] -j <job_id> -s <split> -d <dataset> -t <model_type>"
    echo "ver: $ver"
    cat <<'EOF'
Options:
-h                    Help

-j <job_id >          Generates outputs from a state file saved at the end of a particular job.
                      A job id is usually the same as an epoch number (e.g., 1, 2, etc.)

-s <split>            Dataset split to use for generation:
                      T2G: 'testA', 'valA'
                      G2T: 'testB', 'valB'

-d <dataset>          Dataset to use: 'webnlg' or 'tekgen'

-t <model_type>       Use model from <model_type>, either 'ce' for CE models, or 'scst' for SCST models,
                      trained w/ reinforcement learning (RL)
                      if model_type is 'ce':
                      - assume states will come from './output.<dataset>.[g2t|t2g].ce'
                      - generated output will be saved in './output_eval.<dataset>.<split>.ce'
                      if model_type is 'scst'
                      - assume states will come from './output.<dataset>.[g2t|t2g].scst'
                      - generated output will be saved in './output_eval.<dataset>.<split>.scsy'
                      Note: [g2t|t2g] is derived automatically from <split>, (e.g. 'testA' and 'valA'
                            imply 't2g', while 'testB', 'valB' imply 'g2t')
Notes:
                      You *need* a GPU to run generation

EOF
}


job_id='unknown'
dataset='unknown'
split='unknown'
mtype='unknown'

while getopts "hj:s:d:t:" options; do
    case $options in
        h ) echo "help requested"
            usage
            exit 0;;
        d ) dataset="$OPTARG";;
        j ) job_id="$OPTARG";;
        s ) split="$OPTARG";;
        t ) mtype="$OPTARG";;
        \? ) usage
            exit 0;;
        * ) usage
            exit 1;;
    esac
done

# echo "@ job_id $job_id dataset $dataset split $split"


# You need a GPU for generation...
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]
then
    echo "# Cannot find cuda devices visible -- CUDA_VISIBLE_DEVICES is unset"
    exit 1
fi


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

if [[ $split == "unknown" ]]
then
    echo
    echo "# error: you must specify a dataset split !"
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


# check job_id (is it an integer?)
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

# check split
unk='unknown'
case $split in

    testA | testB | valA | valB)
        : echo "split: $split"
        ;;
    *)
        # echo "unk $arg"
        unk=$split
        ;;
esac

if [[ $unk != "unknown" ]]
then
    echo "# error: invalid split arg -- $unk"
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

# infer direction
dir='g2t'
case $split in

    testA | valA)
        dir='t2g'
        ;;
    *)
        : echo $dir
        ;;
esac


# format job_id
jid=$job_id
jid_zero=$(printf "%02d" $jid)


# define state file and output_dir
state_path='unknown'
output_dir='unknown'

# input state and output dir
state_path="./output.$dataset.$dir.$mtype/checkpoints/state.$jid.pkl"
output_dir="./output_eval.$dataset.$split.$mtype/${jid_zero}"

# prepare engine
prepare=""
if [[ $dataset == "tekgen" ]]
then
    prepare="tekgen-official"
else
    prepare=$dataset
fi

echo "# Generation for jobid= $jid  output_dir= $output_dir"

set -x
python ./generate.py \
       --state_path               $state_path \
       --dataset                  $dataset \
       --prepare                  $prepare \
       --split                    $split \
       --num_workers              2 \
       --batch_size_eval          2 \
       --validate                 True \
       --generate                 True \
       --valid_beam_size          5 \
       --valid_max_length         192 \
       --output_dir               $output_dir
set +x

popd > /dev/null
# echo "# done"
