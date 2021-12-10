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

python -m tools.generate_references --split test --output_dir ./corpora/webnlg-references/release_v3.0/en/references.test

popd > /dev/null
echo "# done"
