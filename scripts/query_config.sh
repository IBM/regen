set -o errexit
set -o nounset
set -o pipefail

# echo "# $(uname -a)"

# handles paths
scp_path=$(realpath -m $0)
scp_dir=$(dirname $scp_path)
pwd_dir=$(pwd -P)

ver='1'

# find the repos root
root=$(git rev-parse --show-toplevel)
pushd $root > /dev/null

if [ $# -ne 2 ]
then
    exit 1
fi
python -m tools.query_config \
       -c "$1" \
       -v "$2"
popd > /dev/null
