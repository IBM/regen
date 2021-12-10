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
-h                    help
-j <job_id >          score hypotheses generated from a state file saved at the end of a particular job.
                      A job id is usually the same as an epoch number.
-s <split>            dataset split to use for generation testA, testB, valA, or valB
-d <dataset>          dataset to use webnlg or tekgen

-t <model_type>       type of model that generated the hypotheses, 'ce' or 'scst'.
                      hypotheses should be in './output_eval.<dataset>.<split>.<model_type>/<job_id>/generate/hypothesis
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


jid=$job_id
jid_zero=$(printf "%02d" $jid)


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

if [[ $direction == "unknown" ]]
then
    echo "# error: cannot figure out generation direction from split $split"
    exit 1
fi


### Evaluations/Scoring

# define input generate and output dirs
generate_dir="./output_eval.$dataset.$split.$mtype/$jid_zero/generate"
output_dir="./score_eval.$dataset.$split.$mtype/${jid_zero}"

mkdir -p $output_dir && echo "created $output_dir"

# get split root: testA -> test
ref_split=${split%?}


if [[ $direction == "g2t" ]]
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

    echo "# Evaluation G2T jobid= $jid  dataset= $dataset split= $split hypothesis= $hypothesis"
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

    echo "# Evaluation T2G jobid= $jid  dataset= $dataset split= $split hypothesis= $hypothesis"
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
