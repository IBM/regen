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
    echo "usage: $scpt -j <job_id>"
    echo "ver: $ver"
    cat <<'EOF'
    Options:
-h                    help
-j <job_id >          train model for job <job_id> w/ job_id = 1, 2, ...
                      A job id is usually the same as an epoch number
EOF
}


job_id='unknown'

while getopts "hj:" options; do
    case $options in
        h ) echo "help requested"
            usage
            exit 0;;
        j ) job_id="$OPTARG";;
        \? ) usage
            exit 0;;
        * ) usage
            exit 1;;
    esac
done

if [[ $job_id == "unknown" ]]
then
    echo
    echo "# error: you must specify a job_id !"
    usage
    exit 1
fi

bash ./scripts/train.sh -j $job_id -d tekgen -x g2t -t ce

popd > /dev/null
