#!/usr/bin/env bash
set -uo pipefail

BASE=${BASE:-/scratch/users/nus/<NSCC_USER>/robomme_nscc}
CODE=${CODE:-${BASE}/code/robomme_policy_learning}
BENCH=${BENCH:-${BASE}/code/robomme_benchmark}
DATA_ROOT=${DATA_ROOT:-${BASE}/data}
FULL_DATA=${FULL_DATA:-${DATA_ROOT}/robomme_preprocessed_data}
CKPT_BASE=${CKPT_BASE:-${BASE}/runs/ckpts}
ASSETS_BASE=${ASSETS_BASE:-${BASE}/runs/assets}
EVAL_BASE=${EVAL_BASE:-${BASE}/runs/evaluation}
LOG_BASE=${LOG_BASE:-${BASE}/logs}
UV=${UV:-${HOME}/.local/bin/uv}
RUN_STAMP=${RUN_STAMP:-nscc_full_$(date +%Y%m%d_%H%M%S)}
LOG_ROOT=${LOG_ROOT:-${LOG_BASE}/${RUN_STAMP}}
SUMMARY=${SUMMARY:-${LOG_ROOT}/summary.tsv}
HISTORY_CONFIG=${HISTORY_CONFIG:-perceptual-framesamp-modul.yaml}
WANDB_PROJECT=${WANDB_PROJECT:-starVLA_RoboMME_NSCC}
WANDB_ENTITY=${WANDB_ENTITY:-laockets-nus}
MAX_EP=${MAX_EP:-10}
TRAIN_TIMEOUT=${TRAIN_TIMEOUT:-150m}
EVAL_TIMEOUT=${EVAL_TIMEOUT:-90m}
BATCH_SIZE=${BATCH_SIZE:-8}
NUM_WORKERS=${NUM_WORKERS:-2}
TRAIN_GPU=${TRAIN_GPU:-0}
SERVER_GPU=${SERVER_GPU:-0}
EVAL_GPU=${EVAL_GPU:-1}
VIEW_SUFFIX=${VIEW_SUFFIX:-full}
VIEW_SLUGS=${VIEW_SLUGS:-binfill,pickxtimes,videoplaceorder,videounmask,stopcube}

mkdir -p "${LOG_ROOT}" "${CKPT_BASE}" "${ASSETS_BASE}" "${EVAL_BASE}"

export PYTHONUNBUFFERED=1
export PYTHONPATH=${BENCH}/src:${BENCH}:${PYTHONPATH:-}
export UV_CACHE_DIR=${UV_CACHE_DIR:-${BASE}/uv-cache}
export UV_PROJECT_ENVIRONMENT=${UV_PROJECT_ENVIRONMENT:-${BASE}/envs/robomme_policy}
export UV_LINK_MODE=copy
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-${BASE}/openpi_data}
export XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}
export JAX_COMPILATION_CACHE_DIR=${JAX_COMPILATION_CACHE_DIR:-${BASE}/jax_cache}
export MMEVLA_INIT_PARAMS=${MMEVLA_INIT_PARAMS:-${BASE}/runs/official/perceptual-framesamp-modul/79999/params}
export OFFICIAL_NORM_STATS=${OFFICIAL_NORM_STATS:-${BASE}/runs/official/perceptual-framesamp-modul/79999/assets/robomme/norm_stats.json}
export WANDB_ENTITY
export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_DIR=${WANDB_DIR:-${BASE}/wandb}
export WANDB_CACHE_DIR=${WANDB_CACHE_DIR:-${BASE}/wandb_cache}
mkdir -p "${WANDB_DIR}" "${WANDB_CACHE_DIR}"

cd "${CODE}" || exit 1

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "${LOG_ROOT}/cycle.log"
  echo "$*" > "${LOG_ROOT}/state.txt"
}

prepare_assets() {
  local config="$1"
  mkdir -p "${ASSETS_BASE}/${config}/robomme"
  if [ -f "${OFFICIAL_NORM_STATS}" ]; then
    cp "${OFFICIAL_NORM_STATS}" "${ASSETS_BASE}/${config}/robomme/norm_stats.json"
  else
    cp "${CODE}/assets/norm_stats.json" "${ASSETS_BASE}/${config}/robomme/norm_stats.json"
  fi
}

ensure_ready() {
  if [ ! -x "${UV}" ]; then
    log "uv missing at ${UV}"
    return 2
  fi
  if [ ! -f "${BASE}/envs/.robomme_policy_ready" ]; then
    log "environment marker missing: ${BASE}/envs/.robomme_policy_ready"
    return 3
  fi
  if [ ! -f "${FULL_DATA}/.ready.json" ]; then
    log "dataset marker missing: ${FULL_DATA}/.ready.json"
    return 4
  fi
}

build_views() {
  local missing=()
  IFS=',' read -r -a view_slugs <<< "${VIEW_SLUGS}"
  for slug in "${view_slugs[@]}"; do
    if [ ! -f "${DATA_ROOT}/robomme_preprocessed_${slug}_${VIEW_SUFFIX}/meta/stats.json" ]; then
      missing+=("${slug}")
    fi
  done
  if [ "${#missing[@]}" -eq 0 ]; then
    log "task-specialist full-data views already exist"
    return 0
  fi
  log "building missing task-specialist views: ${missing[*]}"
  "${UV}" run python "${BASE}/scripts/nscc_build_views_by_episode.py" \
    --source "${FULL_DATA}" \
    --output-root "${DATA_ROOT}" \
    --suffix "${VIEW_SUFFIX}" \
    --slugs "$(IFS=,; echo "${missing[*]}")" \
    > "${LOG_ROOT}/filter_views.log" 2>&1
}

latest_ckpt() {
  local root="$1"
  find "${root}" -mindepth 1 -maxdepth 1 -type d -regex '.*/[0-9]+' -printf '%f\n' 2>/dev/null | sort -n | tail -1
}

parse_rate() {
  local policy="$1"
  local ckpt="$2"
  local task="$3"
  "${UV}" run python - "${EVAL_BASE}" "${policy}" "${ckpt}" "${task}" <<'PY'
import json
import pathlib
import sys

root, policy, ckpt, task = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
base = root / policy / f"ckpt{ckpt}" / "seed42"
logp = base / "log.json"
progp = base / "progress.json"
rate = "NA"
true_count = "NA"
total = "NA"
if logp.exists():
    data = json.loads(logp.read_text())
    rate = data.get("success_rate", {}).get(task, data.get("total_success_rate", "NA"))
if progp.exists():
    data = json.loads(progp.read_text()).get(task, {})
    total = len(data)
    true_count = sum(bool(v) for v in data.values())
print(f"{rate}\t{true_count}\t{total}")
PY
}

run_eval() {
  local task="$1"
  local config="$2"
  local exp="$3"
  local ckpt="$4"
  local port="$5"
  local ckpt_dir="$6"
  local policy="${exp}_${task}"
  local server_log="${LOG_ROOT}/server_${policy}_ckpt${ckpt}.log"
  local eval_log="${LOG_ROOT}/eval_${policy}_ckpt${ckpt}.log"
  local ready=0

  log "eval start task=${task} config=${config} exp=${exp} ckpt=${ckpt} max_ep=${MAX_EP}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    local gpu_count
    gpu_count="$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')"
    if [ "${gpu_count:-0}" -lt 2 ]; then
      EVAL_GPU="${SERVER_GPU}"
    fi
  fi

  CUDA_VISIBLE_DEVICES="${SERVER_GPU}" timeout "${EVAL_TIMEOUT}" "${UV}" run python scripts/serve_policy.py \
    --seed=7 \
    --port="${port}" \
    policy:checkpoint \
    --policy.dir="${ckpt_dir}" \
    --policy.config="${config}" > "${server_log}" 2>&1 &
  local server_pid=$!

  for _ in $(seq 1 150); do
    if ! kill -0 "${server_pid}" 2>/dev/null; then
      log "server died before ready task=${task} exp=${exp}"
      tail -100 "${server_log}" >> "${LOG_ROOT}/cycle.log" || true
      return 97
    fi
    if "${UV}" run python - "${port}" >/dev/null 2>&1 <<'PY'
import socket
import sys
s = socket.socket()
s.settimeout(1)
s.connect(("127.0.0.1", int(sys.argv[1])))
s.close()
PY
    then
      ready=1
      break
    fi
    sleep 5
  done

  if [ "${ready}" != "1" ]; then
    log "server timeout task=${task} exp=${exp}"
    kill "${server_pid}" 2>/dev/null || true
    wait "${server_pid}" 2>/dev/null || true
    return 98
  fi

  CUDA_VISIBLE_DEVICES="${EVAL_GPU}" timeout "${EVAL_TIMEOUT}" "${UV}" run python examples/robomme/eval.py \
    --args.host=127.0.0.1 \
    --args.port="${port}" \
    --args.policy-name="${policy}" \
    --args.model-ckpt-id="${ckpt}" \
    --args.save-dir="${EVAL_BASE}" \
    --args.only-tasks="${task}" \
    --args.max-episodes-per-task="${MAX_EP}" \
    --args.max-steps=1300 \
    --args.overwrite > "${eval_log}" 2>&1
  local eval_rc=$?

  kill "${server_pid}" 2>/dev/null || true
  wait "${server_pid}" 2>/dev/null || true

  local parsed
  parsed="$(parse_rate "${policy}" "${ckpt}" "${task}")"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date '+%F %T')" "${task}" "${config}" "${exp}" "${ckpt}" "${eval_rc}" "${parsed}" "${eval_log}" >> "${SUMMARY}"
  log "eval done task=${task} exp=${exp} ckpt=${ckpt} rc=${eval_rc} parsed=${parsed}"
  return "${eval_rc}"
}

train_then_eval() {
  local task="$1"
  local slug="$2"
  local config="$3"
  local steps="$4"
  local port="$5"
  local dataset="${DATA_ROOT}/robomme_preprocessed_${slug}_${VIEW_SUFFIX}"
  local exp="${RUN_STAMP}_${slug}_${config}_${steps}"
  local train_log="${LOG_ROOT}/train_${exp}.log"
  local ckpt_root="${CKPT_BASE}/${config}/${exp}"
  local train_rc=0
  local eval_rc=99
  local ckpt

  prepare_assets "${config}"
  log "train start task=${task} config=${config} steps=${steps} dataset=${dataset}"
  CUDA_VISIBLE_DEVICES="${TRAIN_GPU}" timeout "${TRAIN_TIMEOUT}" "${UV}" run python scripts/train.py "${config}" \
    --exp-name="${exp}" \
    --project-name="${WANDB_PROJECT}" \
    --batch-size="${BATCH_SIZE}" \
    --num-workers="${NUM_WORKERS}" \
    --fsdp-devices=1 \
    --num-train-steps="${steps}" \
    --save-interval=500 \
    --keep-period=500 \
    --checkpoint-base-dir="${CKPT_BASE}" \
    --assets-base-dir="${ASSETS_BASE}" \
    --dataset-path="${dataset}" \
    --model.use_history \
    --model.history_config="${HISTORY_CONFIG}" \
    --overwrite > "${train_log}" 2>&1
  train_rc=$?
  ckpt="$(latest_ckpt "${ckpt_root}")"
  log "train done task=${task} config=${config} rc=${train_rc} latest_ckpt=${ckpt:-NA}"

  if [ -n "${ckpt}" ]; then
    run_eval "${task}" "${config}" "${exp}" "${ckpt}" "${port}" "${ckpt_root}/${ckpt}"
    eval_rc=$?
  fi

  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$(date '+%F %T')" "${task}" "${config}" "${exp}" "${steps}" "${ckpt:-NA}" "${train_rc}" "${eval_rc}" "${train_log}" >> "${SUMMARY}"
}

eval_official_best() {
  local task="$1"
  local port="$2"
  local official_dir="${BASE}/runs/official/perceptual-framesamp-modul/79999"
  if [ ! -d "${official_dir}/params" ]; then
    log "official checkpoint missing, skip official eval: ${official_dir}/params"
    return 0
  fi
  run_eval "${task}" "mme_vla_suite" "official_perceptual_framesamp_modul" "79999" "${port}" "${official_dir}"
}

eval_official_as() {
  local task="$1"
  local exp="$2"
  local port="$3"
  local official_dir="${BASE}/runs/official/perceptual-framesamp-modul/79999"
  if [ ! -d "${official_dir}/params" ]; then
    log "official checkpoint missing, skip official eval: ${official_dir}/params"
    return 0
  fi
  run_eval "${task}" "mme_vla_suite" "${exp}" "79999" "${port}" "${official_dir}"
}

main_default() {
  printf "time\ttask\tconfig\texp\tckpt_or_steps\trc_or_ckpt\trate\ttrue_count\ttotal_or_log\n" > "${SUMMARY}"
  ensure_ready || exit $?
  build_views || exit $?

  # Official baseline first, then the strongest Vast-derived specialist routes.
  eval_official_best "PickXtimes" 18101 || true
  eval_official_best "VideoPlaceOrder" 18102 || true
  eval_official_best "VideoUnmask" 18103 || true

  train_then_eval "PickXtimes" "pickxtimes" "mme_vla_suite_mem_action" 500 18201
  train_then_eval "VideoPlaceOrder" "videoplaceorder" "mme_vla_suite_mem_only" 1000 18202
  train_then_eval "VideoUnmask" "videounmask" "mme_vla_suite_mem_action" 1000 18203

  log "cycle finished"
}

eval_existing_ckpt() {
  local task="$1"
  local config="$2"
  local exp="$3"
  local ckpt="$4"
  local port="$5"
  local ckpt_dir="${CKPT_BASE}/${config}/${exp}/${ckpt}"

  if [ ! -d "${ckpt_dir}/params" ]; then
    log "existing checkpoint missing, skip task=${task} exp=${exp} ckpt=${ckpt}"
    return 0
  fi
  run_eval "${task}" "${config}" "${exp}" "${ckpt}" "${port}" "${ckpt_dir}"
}

main_conservative() {
  printf "time\ttask\tconfig\texp\tckpt_or_steps\trc_or_ckpt\trate\ttrue_count\ttotal_or_log\n" > "${SUMMARY}"
  ensure_ready || exit $?
  build_views || exit $?

  # First check whether the already-saved 500-step checkpoints are better
  # than the 1000-step checkpoints before spending training credit.
  eval_existing_ckpt \
    "VideoPlaceOrder" \
    "mme_vla_suite_mem_only" \
    "${BASELINE_RUN_STAMP:-nscc_full_smoke_20260519_113718}_videoplaceorder_mme_vla_suite_mem_only_1000" \
    "500" \
    18301 || true
  eval_existing_ckpt \
    "VideoUnmask" \
    "mme_vla_suite_mem_action" \
    "${BASELINE_RUN_STAMP:-nscc_full_smoke_20260519_113718}_videounmask_mme_vla_suite_mem_action_1000" \
    "500" \
    18302 || true

  # Then test the two safest alternatives for the failed routes: allow action
  # adapters on VideoPlaceOrder, and lower LR for tasks where the official
  # policy is already near the short-eval ceiling.
  train_then_eval "VideoPlaceOrder" "videoplaceorder" "mme_vla_suite_mem_action" 500 18303
  train_then_eval "VideoPlaceOrder" "videoplaceorder" "mme_vla_suite_mem_action_lr2e5" 500 18304
  train_then_eval "VideoUnmask" "videounmask" "mme_vla_suite_mem_action_lr2e5" 500 18305

  log "conservative cycle finished"
}

main_formal_eval() {
  printf "time\ttask\tconfig\texp\tckpt_or_steps\trc_or_ckpt\trate\ttrue_count\ttotal_or_log\n" > "${SUMMARY}"
  ensure_ready || exit $?
  build_views || exit $?

  # Larger same-episode-count evaluation for routes that survived smoke.
  # Keep official baselines in the same run so stochastic/eval settings are
  # directly comparable.
  eval_official_best "PickXtimes" 18401 || true
  eval_existing_ckpt \
    "PickXtimes" \
    "mme_vla_suite_mem_action" \
    "${PICKXTIMES_BEST_EXP:-nscc_full_smoke_20260519_113718_pickxtimes_mme_vla_suite_mem_action_500}" \
    "${PICKXTIMES_BEST_CKPT:-499}" \
    18402 || true

  eval_official_best "VideoPlaceOrder" 18403 || true
  eval_existing_ckpt \
    "VideoPlaceOrder" \
    "mme_vla_suite_mem_action" \
    "${VPO_BEST_EXP:-nscc_conservative_20260519_135438_videoplaceorder_mme_vla_suite_mem_action_500}" \
    "${VPO_BEST_CKPT:-499}" \
    18404 || true
  eval_existing_ckpt \
    "VideoPlaceOrder" \
    "mme_vla_suite_mem_action_lr2e5" \
    "${VPO_LR2E5_EXP:-nscc_conservative_20260519_135438_videoplaceorder_mme_vla_suite_mem_action_lr2e5_500}" \
    "${VPO_LR2E5_CKPT:-499}" \
    18405 || true

  eval_official_best "VideoUnmask" 18406 || true

  log "formal eval cycle finished"
}

main_alltask_smoke() {
  printf "time\ttask\tconfig\texp\tckpt_or_steps\trc_or_ckpt\trate\ttrue_count\ttotal_or_log\n" > "${SUMMARY}"
  ensure_ready || exit $?
  build_views || exit $?

  local tasks=(
    BinFill StopCube PickXtimes SwingXtimes
    ButtonUnmask VideoUnmask VideoUnmaskSwap ButtonUnmaskSwap
    PickHighlight VideoRepick VideoPlaceButton VideoPlaceOrder
    MoveCube InsertPeg PatternLock RouteStick
  )
  local slugs=(
    binfill stopcube pickxtimes swingxtimes
    buttonunmask videounmask videounmaskswap buttonunmaskswap
    pickhighlight videorepick videoplacebutton videoplaceorder
    movecube insertpeg patternlock routestick
  )

  local i task slug base_port
  for i in "${!tasks[@]}"; do
    task="${tasks[$i]}"
    slug="${slugs[$i]}"
    base_port=$((18500 + i * 3))
    eval_official_as "${task}" "${RUN_STAMP}_official_${slug}" "${base_port}" || true
    train_then_eval "${task}" "${slug}" "${ALLTASK_CONFIG:-mme_vla_suite_mem_action}" "${ALLTASK_STEPS:-500}" "$((base_port + 1))"
  done

  log "alltask smoke cycle finished"
}

main() {
  case "${CYCLE_MODE:-default}" in
    conservative)
      main_conservative "$@"
      ;;
    formal_eval)
      main_formal_eval "$@"
      ;;
    alltask_smoke)
      main_alltask_smoke "$@"
      ;;
    default)
      main_default "$@"
      ;;
    *)
      log "unknown CYCLE_MODE=${CYCLE_MODE}"
      exit 64
      ;;
  esac
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  main "$@"
fi
