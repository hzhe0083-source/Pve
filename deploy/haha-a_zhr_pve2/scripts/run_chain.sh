#!/usr/bin/env bash
# A chain of skills in one network: Pick (expert) -> Stack -> PlaceCube. Each seed continues the Stack run of the same
# seed in CHAIN_SRC (scripts/run_protect*.sh output on this host). Arms, all learning PlaceCube with the gripper floor:
#   chain64    one more null-space level (width 64) on the Stack adapter run; the new head starts as Stack's head
#   chain64p   the same, but the new head starts as Pick's head: against place_nsa64 (one level straight from Pick,
#              same head start) this isolates what reading the Stack adapter's features is worth
#   ftchain    plain full fine-tuning from the full-FT Stack run (stack_monitor): does forgetting pile up?
#   chain64np  the prefix-sum ablation: no projection at either level (base: stack_nsa64np in CHAIN_SRC_NP); every
#              skill is exact only through its own level (scripts/nsa_eval.py --read own)
# CHAIN_ARMS picks the arms (default chain64 chain64p ftchain).
# Afterwards scripts/nsa_eval.py scores Pick, Stack and PlaceCube through the final network.
#   CHAIN_SEEDS="1 2 3" CHAIN_P=3 bash scripts/run_chain.sh
set -u
R=${CHAIN_ROOT:-$HOME/zhr_pve}
cd "$R" || exit 1
source .venv/bin/activate
export PYTHONPATH=$R PVE_ROOT=$R PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
G="bash $R/scripts/with_gpu_mem.sh 6000 bash $R/scripts/gpu_polite.sh"   # yields to other users' GPU jobs
PICK=$R/runs/experts_wc/runs/PickCube_wc_c1/final_ckpt.pt
SRC=${CHAIN_SRC:-$R/runs/protect_c/runs}
D=${CHAIN_DIR:-runs/chain}
mkdir -p "$D" logs
J=$D/jobs_$(hostname).txt
: > "$J"
COMMON="--env_id=PlaceCube-v1 --env_kwargs='{\"robot_uids\": \"panda_wristcam\", \"reward_mode\": \"sparse\"}' \
--num_envs=1024 --num_steps=50 --eval_freq=2 --num_eval_envs=256 --total_timesteps=25_000_000 --halt_at=8_000_000 \
--bootstrap_at_done=truncated --gamma=0.95 --stop_success=0.9 --stop_patience=3 --target_kl=1e9 --min_grip_std=1.0 \
--init_ckpt=$PICK --init_actor_only --save_every=500000"
for seed in ${CHAIN_SEEDS:-1 2 3}; do
    NSA=$SRC/stack_nsa64_s$seed/final_ckpt.pt
    FT=$SRC/stack_monitor_s$seed/final_ckpt.pt
    NP=${CHAIN_SRC_NP:-$SRC}/stack_nsa64np_s$seed/final_ckpt.pt
    for a in ${CHAIN_ARMS:-chain64 chain64p ftchain}; do
        case $a in chain64|chain64p) f=$NSA ;; ftchain) f=$FT ;; chain64np) f=$NP ;; *) echo "UNKNOWN ARM $a"; exit 1 ;; esac
    done
    n() {
        e=place_$1_s$seed
        [ -f "$D/runs/$e/final_ckpt.pt" ] && return
        echo "$G python -m wrwl.ppo --exp_name=$e $COMMON --seed=$seed $2 > logs_$e.txt 2>&1 && echo DONE $e || echo EXIT \$? $e" >> "$J"
    }
    for a in ${CHAIN_ARMS:-chain64 chain64p ftchain}; do
        case $a in chain64|chain64p) f=$NSA ;; ftchain) f=$FT ;; chain64np) f=$NP ;; esac
        [ -f "$f" ] || { echo "MISSING $f on $(hostname): place_${a}_s$seed skipped"; continue; }
        case $a in
            chain64) n chain64 "--nsa_base=$NSA --nullspace_adapter=64" ;;
            chain64p) n chain64p "--nsa_base=$NSA --nullspace_adapter=64 --nsa_head_from=init" ;;
            ftchain) n ftchain "--init_ckpt=$FT" ;;
            chain64np) n chain64np "--nsa_base=$NP --nullspace_adapter=64 --nsa_project=False" ;;
        esac
    done
done
echo "CHAIN_START $(date +%T) $(hostname) jobs=$(wc -l < "$J")"
(cd "$D" && xargs -P "${CHAIN_P:-3}" -d '\n' -I CMD bash -c CMD < "$(basename "$J")")
echo "CHAIN_END $(date +%T)"
