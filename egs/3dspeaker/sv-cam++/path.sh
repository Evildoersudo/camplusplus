script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
project_root=$(cd "${script_dir}/../../.." && pwd)

export PATH="${script_dir}:$PATH"
export PYTHONPATH="${project_root}:$PYTHONPATH"
export OMP_NUM_THREADS=1
