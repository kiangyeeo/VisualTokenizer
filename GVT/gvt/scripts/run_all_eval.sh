#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "${script_dir}/.." && pwd)"

data_root=""
vicuna_path=""
load_path=""
batch_size="4"
output_dir="benchmark_outputs"
tasks="task_eval_coco_count,task_eval_coco_multiclass,task_eval_vcr_count,task_eval_vcr_multiclass,task_eval_coco_caption,task_eval_vqav2"
skip_missing_data="false"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --data-root)
      data_root="$2"
      shift 2
      ;;
    --vicuna-path)
      vicuna_path="$2"
      shift 2
      ;;
    --load-path)
      load_path="$2"
      shift 2
      ;;
    --batch-size)
      batch_size="$2"
      shift 2
      ;;
    --output-dir)
      output_dir="$2"
      shift 2
      ;;
    --tasks)
      tasks="$2"
      shift 2
      ;;
    --skip-missing-data)
      skip_missing_data="true"
      shift
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${data_root}" || -z "${vicuna_path}" || -z "${load_path}" ]]; then
  echo "Usage: $0 --data-root PATH --vicuna-path PATH --load-path PATH [--batch-size 4] [--output-dir benchmark_outputs] [--tasks task_a,task_b] [--skip-missing-data]" >&2
  exit 2
fi

abs_path() {
  if [[ "$1" = /* ]]; then
    echo "$1"
  else
    echo "$(pwd)/$1"
  fi
}

data_root="$(abs_path "${data_root}")"
vicuna_path="$(abs_path "${vicuna_path}")"
load_path="$(abs_path "${load_path}")"
output_dir="$(abs_path "${output_dir}")"

mkdir -p "${output_dir}/pred_results/count" "${output_dir}/output"

required_arrow_for_task() {
  case "$1" in
    task_eval_coco_count) echo "coco_oc.arrow" ;;
    task_eval_coco_multiclass) echo "coco_mci.arrow" ;;
    task_eval_vcr_count) echo "vcr_oc.arrow" ;;
    task_eval_vcr_multiclass) echo "vcr_mci.arrow" ;;
    task_eval_coco_caption) echo "coco_caption_karpathy_val.arrow" ;;
    task_eval_vqav2) echo "vqav2_rest_val.arrow" ;;
    *) echo "" ;;
  esac
}

IFS=',' read -ra task_list <<< "${tasks}"
for task in "${task_list[@]}"; do
  task="$(echo "${task}" | xargs)"
  [[ -z "${task}" ]] && continue

  required_arrow="$(required_arrow_for_task "${task}")"
  if [[ -n "${required_arrow}" && ! -f "${data_root}/${required_arrow}" ]]; then
    if [[ "${skip_missing_data}" == "true" ]]; then
      echo "[skip] ${task}: missing ${data_root}/${required_arrow}"
      continue
    fi
    echo "[error] ${task}: missing ${data_root}/${required_arrow}" >&2
    exit 1
  fi

  echo "[eval] ${task}"
  (
    cd "${project_root}"
    python run.py with \
      "${task}" \
      num_gpus=1 num_nodes=1 \
      test_only=True \
      test_on_val=True \
      image_size=224 \
      num_latents=32 \
      per_gpu_batchsize="${batch_size}" \
      data_root="${data_root}" \
      vicuna_path="${vicuna_path}" \
      load_path="${load_path}" \
      output_dir="${output_dir}" \
      log_dir="${output_dir}/output"
  )
done

python "${project_root}/../tools/collect_results.py" \
  --result-dir "${output_dir}/pred_results" \
  --output-json "${output_dir}/summary_results.json" \
  --output-md "${output_dir}/summary_results.md"
