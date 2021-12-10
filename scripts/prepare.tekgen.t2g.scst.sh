set -o errexit
set -o nounset
set -o pipefail

echo "# $(uname -a)"

# handles paths
scp_path=$(realpath -m $0)
scp_dir=$(dirname $scp_path)
pwd_dir=$(pwd -P)

ver='1'

# find the repos root
root=$(git rev-parse --show-toplevel)
pushd $root > /dev/null

python -m tools.prepare_scst \
       --config ./cfg/train.tekgen.t2g.scst.cfg \
       --mode copy

popd > /dev/null
echo "# done"
