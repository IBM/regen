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
    echo "usage: $scpt [-h] -f <hypotheses_file> -s <split> -d <dataset> -t <model_type>"
    echo "ver: $ver"
    cat <<'EOF'
Options:
-h                    help

-f <hypotheses_file>  Score hypotheses generated from a state file checkpointed for evaluation (usually saved mid-training)

-s <split>            dataset split to use for generation:
                      T2G: 'testA', 'valA'
                      G2T: 'testB', 'valB'

-d <dataset>          dataset to use: 'webnlg' or 'tekgen'

-t <model_type>       Use models from <model_type>: 'ce' for CE models, or 'scst' for SCST models, trained w/
                      reinforcement learning (RL)
                      if model_type is 'ce':
                      - scoring output will be in './score_eval_checkpoints.<dataset>.<split>.ce'
                      if model_type is 'scst'
                      - scoring output will be in './score_eval_checkpoints.<dataset>.<split>.scst'

EOF
}

hypothesis='unknown'
dataset='unknown'
split='unknown'
mtype='unknown'

while getopts "hf:s:d:t:" options; do
    case $options in
        h ) echo "help requested"
            usage
            exit 0;;
        d ) dataset="$OPTARG";;
        f ) hypothesis="$OPTARG";;
        s ) split="$OPTARG";;
        t ) mtype="$OPTARG";;
        \? ) usage
            exit 0;;
        * ) usage
            exit 1;;
    esac
done

# echo "@ hypothesis $hypothesis dataset $dataset split $split"

if [[ $hypothesis == "unknown" ]]
then
    echo
    echo "# error: you must specify a hypothesis !"
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


# hypothesis
# check if hypothesis file exits

if ! [[ -f $hypothesis ]]
then
    echo
    echo "# error: hypothesis does not exist: cannot find $hypothesis"
    exit 1
fi

# check if epoch can be parsed from hypothesis
# get epoch from hypothesis
# hypothesis ./output_eval_checkpoints.scst.webnlg.testB/5.1942186/generate/hypothesis
# -> get 5.1942186
epoch=$(basename $(dirname $(dirname $hypothesis)))

# check if epoch is a number int or positive float
#
re_float='^[0-9]+([.][0-9]+)?$'
if ! [[ $epoch =~ $re_float ]]
then
    echo
    echo "# error: epoch '$epoch' parsed from state path '$hypothesis' is not a floating point number"
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

if [ "$unk" != "unknown" ]
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

if [ "$unk" != "unknown" ]
then
    echo "# error: invalid model type arg -- $unk"
    exit 1
fi


# direction
direction='unknown'

case $split in

    testA | valA)
        direction='t2g'
        ;;

    testB | valB)
        direction='g2t'
        ;;

    *)
        direction='unknown'
        ;;
esac

if [ "$direction" == "unknown" ]
then
    echo "# error: cannot figure out generation direction from split $split"
    exit 1
fi


### Evaluations/scoring

# define output_dir
generate_dir="./output_eval_checkpoints.$dataset.$split.$mtype/$epoch/generate"
output_dir="./score_eval_checkpoints.$dataset.$split.$mtype/$epoch"

mkdir -p $output_dir  && echo "created $output_dir"

# get split root: testA -> test
ref_split=${split%?}


if [ "$direction" == "g2t" ]
then

    hypothesis=$generate_dir/hypothesis
    reference="unknown"
    numref=4

    if [[ $dataset == 'webnlg' ]]
    then
        reference="./corpora/webnlg-references/release_v3.0/en/references.$ref_split/reference"
    else
        reference=$generate_dir/target
        numref=1
    fi

    echo "# Evaluation G2T epoch= $epoch  dataset= $dataset split= $split hypothesis= $hypothesis"
    pushd ./eval/GenerationEval > /dev/null

    set -x
    python ./eval.py \
           --reference ../../$reference \
           --hypothesis ../../$hypothesis \
           -m bleu,meteor,chrf++ \
           -nr $numref | tee ../../$output_dir/score.out
    set +x

    popd > /dev/null

else

    hypothesis=$generate_dir/hyps.xml
    reference="unknown"

    if [[ $dataset == 'webnlg' ]]
    then
        reference="./corpora/webnlg-references/release_v3.0/en/references.rdf.$ref_split/$ref_split.xml"
    else
        reference=$generate_dir/targets.xml
    fi

    echo "# Evaluation T2G epoch= $epoch  dataset= $dataset split= $split hypothesis= $hypothesis"
    pushd ./eval/WebNLG-Text-to-triples > /dev/null

    set -x
    python ./Evaluation_script.py \
           ../../$reference \
           ../../$hypothesis | tee ../../$output_dir/score.out
    set +x

    popd > /dev/null

fi

popd > /dev/null
echo "# done"
