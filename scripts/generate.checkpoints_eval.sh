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
    echo "usage: $scpt [-h] -f <state_path> -s <split> -d <dataset> -t <model_type>"
    echo "ver: $ver"
    cat <<'EOF'
Options:
-h                    Help

-f <state_path>       Generates outputs for a state file (model) checkpointed for evaluation during training.
                      A state file and models are checkpointed within './output.<dataset>.[g2t|t2g].<model_type>/checkpoints_eval/' dirs:
                      - for CE training within './output.<dataset>.[g2t|t2g].ce/checkpoints_eval/' dir
                      - for RL training within './output.<dataset>.[g2t|t2g].scst/checkpoints_eval/' dir
                      Given a state file path, hypotheses will be generated for the <dataset> split requested

-s <split>            Dataset split to use for generation:
                      T2G: 'testA', 'valA'
                      G2T: 'testB', 'valB'

-d <dataset>          Dataset to use: 'webnlg' or 'tekgen'

-t <model_type>       Use model from <model_type>: either 'ce' for CE models, or 'scst' for SCST models trained w/
                      reinforcement learning (RL)
                      if <model_type> is 'ce':   generated output in './output_checkpoints_eval.<dataset>.<split>.ce'
                      if <model_type> is 'scst': generated output in './output_checkpoints_eval.<dataset>.<split>.scst'
                      Note: [g2t|t2g] is derived automatically from <split>, (e.g. 'testA' and 'valA'
                            imply 't2g', while 'testB', 'valB' imply 'g2t')
Notes:
                      You *need* a GPU to run generation

EOF
}

state_path='unknown'
dataset='unknown'
split='unknown'
mtype='unknown'

while getopts "hf:s:d:t:" options; do
    case $options in
        h ) echo "help requested"
            usage
            exit 0;;
        d ) dataset="$OPTARG";;
        f ) state_path="$OPTARG";;
        s ) split="$OPTARG";;
        t ) mtype="$OPTARG";;
        \? ) usage
            exit 0;;
        * ) usage
            exit 1;;
    esac
done

# echo "@ state_path $state_path dataset $dataset split $split"


# You need a GPU for generation...
if [[ -z ${CUDA_VISIBLE_DEVICES:-} ]]
then
    echo "# Cannot find cuda devices visible -- CUDA_VISIBLE_DEVICES is unset"
    exit 1
fi

if [[ $state_path == "unknown" ]]
then
    echo
    echo "# error: you must specify a state_path !"
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


#  state path
# check if state_path exists

if ! [[ -f $state_path ]]
then
    echo
    echo "# error: state_path does not exist: cannot find $state_path"
    exit 1
fi

# check if epoch can be parsed from state_path
# get epoch from state_path
# state_path ./output/checkpoints_eval/state.6.epoch.5.1942186.iter.23000.pkl"
# -> get 5.1942186
epoch=$(echo $state_path | sed -e 's/^.*epoch\.//' -e 's/\.iter.*//')

# check if epoch is a number int or positive float
#
refloat='^[0-9]+([.][0-9]+)?$'
if ! [[ $epoch =~ $refloat ]]
then
    echo
    echo "# error: epoch '$epoch' parsed from state path '$state_path' is not a floating point number"
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

if [[ $unk != "unknown" ]]
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


# define output_dir
output_dir="./output_eval_checkpoints.$dataset.$split.$mtype/${epoch}"

# prepare engine
prepare=""
if [[ $dataset == "tekgen" ]]
then
    prepare="tekgen-official"
else
    prepare=$dataset
fi


echo "# Generation for eval checkpoint epoch= $epoch  output_dir= $output_dir  state_path= $state_path"

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
