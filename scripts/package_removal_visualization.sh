#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! "$1" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "Usage: bash scripts/package_removal_visualization.sh <run_id>" >&2
  exit 2
fi

run_id="$1"
run_dir="outputs/tree_ring_removal_visualization/${run_id}"
package_dir="/root/autodl-tmp/experiment_packages"
archive="${package_dir}/${run_id}.tar.gz"
checksum="${archive}.sha256"
contents="${archive}.contents.txt"

[[ -d "${run_dir}" ]] || { echo "Missing run directory: ${run_dir}" >&2; exit 1; }
[[ -f "${run_dir}/manifest.json" ]] || { echo "Missing complete manifest" >&2; exit 1; }
[[ ! -f "${run_dir}/manifest.partial.json" ]] || { echo "Run is incomplete" >&2; exit 1; }
python -c 'import json,sys; m=json.load(open(sys.argv[1], encoding="utf-8")); assert m["status"] == "complete" and m["run_id"] == sys.argv[2]' "${run_dir}/manifest.json" "${run_id}"
mkdir -p "${package_dir}"
for output in "${archive}" "${checksum}" "${contents}"; do
  [[ ! -e "${output}" ]] || { echo "Refusing to overwrite: ${output}" >&2; exit 1; }
done
tar -czf "${archive}" -C "$(dirname "${run_dir}")" "${run_id}"
sha256sum "${archive}" > "${checksum}"
tar -tzf "${archive}" > "${contents}"
printf 'Created:\n%s\n%s\n%s\n' "${archive}" "${checksum}" "${contents}"
