#!/usr/bin/env bash
# One-off on haha-c (01 Nov, ~03:30), new code in ~/zhr_pve2 (wrwl/nsa.py, FN-PPO) so the Pick->PlaceCube queue running
# from ~/zhr_pve keeps its code. After that queue ends: GPU smoke tests (FN-PPO, one chain level), the skill chain
# Pick -> Stack -> PlaceCube (scripts/run_chain.sh) with scripts/nsa_eval.py on every chain run, then FN-PPO and its
# comparisons on Stack to 3M steps. Every job yields the card to other users (gpu_polite.sh); up to 4 passes per queue.
R=$HOME/zhr_pve2
cd "$R" || exit 1
source .venv/bin/activate
export PYTHONPATH=$R PVE_ROOT=$R PYTHONUNBUFFERED=1 CUDA_VISIBLE_DEVICES=0
SRC=$HOME/zhr_pve/runs/protect_c/runs
PICK=$R/runs/experts_wc/runs/PickCube_wc_c1/final_ckpt.pt
P="bash $R/scripts/gpu_polite.sh"
until [ -f "$HOME/zhr_pve/logs/drv_c4.done" ]; do sleep 60; done
echo "start $(date +%T)"
mkdir -p smoke && cd smoke
SM="--num_envs=256 --num_steps=50 --eval_freq=1000 --num_eval_envs=16 --total_timesteps=38400 --bootstrap_at_done=truncated \
--gamma=0.95 --target_kl=1e9 --min_grip_std=1.0 --init_ckpt=$PICK --init_actor_only --protect_states=$R/data/bc_pick_c1.pt"
$P python -m wrwl.ppo --exp_name=smoke_fn --env_id=StackCubeNarrow-v1 --env_kwargs='{"reward_mode": "sparse"}' \
    --obs_adapter=stack_to_pick $SM --protect_old=fn --protect_env=PickCube-v1 --protect_dagger=1 > smoke_fn.txt 2>&1
rf=$?
$P python -m wrwl.ppo --exp_name=smoke_chain --env_id=PlaceCube-v1 \
    --env_kwargs='{"robot_uids": "panda_wristcam", "reward_mode": "sparse"}' $SM \
    --nsa_base=$SRC/stack_nsa64_s1/final_ckpt.pt --nullspace_adapter=64 > smoke_chain.txt 2>&1
rc=$?
cd "$R"
echo "smoke fn rc=$rf chain rc=$rc $(date +%T)"
if [ $rf != 0 ] || [ $rc != 0 ]; then echo "PROT_END smoke failed $(date +%T)" > logs/drv_c5.done; exit 1; fi
for pass in 1 2 3 4; do
    CHAIN_ROOT=$R CHAIN_SRC=$SRC CHAIN_DIR=runs/chain CHAIN_SEEDS="1 2 3" CHAIN_P=3 bash scripts/run_chain.sh \
        >> logs/run_chain_c5.log 2>&1
    left=$(for a in chain64 chain64p ftchain; do for s in 1 2 3; do
        [ -f runs/chain/runs/place_${a}_s$s/final_ckpt.pt ] || echo x; done; done | wc -l)
    echo "chain pass $pass $(date +%T) unfinished $left"
    [ "$left" = 0 ] && break
    sleep 300
done
for s in 1 2 3; do
    for a in chain64 chain64p ftchain; do
        c=runs/chain/runs/place_${a}_s$s/final_ckpt.pt
        [ -f "$c" ] || continue
        if [ $a = ftchain ]; then st=$SRC/stack_monitor_s$s/final_ckpt.pt; ls=0; lp=0
        else st=$SRC/stack_nsa64_s$s/final_ckpt.pt; ls=1; lp=2; fi
        $P python scripts/nsa_eval.py --ckpt=$c \
            --skills=pick:PickCube-v1:$PICK:0,stack:StackCubeNarrow-v1:$st:$ls,place:PlaceCube-v1:$c:$lp \
            --out=runs/chain/eval_place_${a}_s$s.json > runs/chain/eval_place_${a}_s$s.log 2>&1
        echo "nsa_eval place_${a}_s$s rc=$? $(date +%T)"
    done
done
for pass in 1 2 3 4; do
    PROT_ROOT=$R PROT_DIR=runs/protect_3m PROT_STOP="--halt_at=3_000_000 --save_every=102400" PROT_SEEDS="1 2 3" \
        PROT_P=3 PROT_ARMS="monitor fncl fnbank fncl_w5 klcl1 klcl3" bash scripts/run_protect.sh >> logs/run_fn_c5.log 2>&1
    left=$(for a in monitor fncl fnbank fncl_w5 klcl1 klcl3; do for s in 1 2 3; do
        [ -f runs/protect_3m/runs/stack_${a}_s$s/final_ckpt.pt ] || echo x; done; done | wc -l)
    echo "fn pass $pass $(date +%T) unfinished $left"
    [ "$left" = 0 ] && break
    sleep 300
done
echo "PROT_END $(date +%T)" > logs/drv_c5.done
