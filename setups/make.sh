# bash setups/make.sh SETUP DIR   ->  a fresh run folder DIR (put it outside this repo) with everything the agent needs
# SETUP is one of: before, free, look, live, forced
set -e; H="$(cd "$(dirname "$0")/.." && pwd)"; D="${2:?}"
test ! -e "$D"; mkdir -p "$D"
cp "$H/fixed.py" "$H/shakespeare.txt" "$H/setups/optimizer.py" "$H"/setups/"${1:?}"/*.py "$D/"
echo "$D"
