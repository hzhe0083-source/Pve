# Task keys of the queues (scripts/run_protect.sh PROT_TASK, scripts/run_chain.sh CHAIN_TASKS) and their env flags for
# wrwl.ppo. A key has no underscore: run names are <task>_<arm>_s<seed>, which the analysis scripts split on "_".
#   place    PlaceCube-v1, PickCube's own layout           stack    StackCubeNarrow-v1 via stack_to_pick
#   sphere   PlaceSphere-v1 via sphere_to_pick             ycb      PickYCBSubset-v1 via ycb_to_pick (YCB assets)
#   pushng   PushCubeNoGrasp-v1 via push_to_pick           pullng   PullCubeNoGrasp-v1 via push_to_pick
#   beside   PyramidBeside-v1 via beside_to_pick           onpair   PyramidOnPair-v1 via onpair_to_pick
#   liftpeg  LiftPegUprightG-v1 via peg_to_pick
# The new tasks all run the Pick expert's robot (panda_wristcam) with the sparse reward for 50 steps. Which of them
# count in the pre-registered analysis (counted / near-transfer / development) is fixed in scripts/suite_gate.py.
#   source scripts/suite_tasks.sh; task_env sphere; task_env_id sphere
task_env() {
    local s='"robot_uids": "panda_wristcam", "reward_mode": "sparse"'
    case $1 in
        place) echo "--env_id=PlaceCube-v1 --env_kwargs='{$s}'" ;;
        stack) echo "--env_id=StackCubeNarrow-v1 --env_kwargs='{\"reward_mode\": \"sparse\"}' --obs_adapter=stack_to_pick" ;;
        sphere) echo "--env_id=PlaceSphere-v1 --env_kwargs='{$s}' --obs_adapter=sphere_to_pick" ;;
        ycb) echo "--env_id=PickYCBSubset-v1 --env_kwargs='{$s}' --obs_adapter=ycb_to_pick" ;;
        pushng) echo "--env_id=PushCubeNoGrasp-v1 --env_kwargs='{$s}' --obs_adapter=push_to_pick" ;;
        pullng) echo "--env_id=PullCubeNoGrasp-v1 --env_kwargs='{$s}' --obs_adapter=push_to_pick" ;;
        beside) echo "--env_id=PyramidBeside-v1 --env_kwargs='{$s}' --obs_adapter=beside_to_pick" ;;
        onpair) echo "--env_id=PyramidOnPair-v1 --env_kwargs='{$s}' --obs_adapter=onpair_to_pick" ;;
        liftpeg) echo "--env_id=LiftPegUprightG-v1 --env_kwargs='{$s}' --obs_adapter=peg_to_pick" ;;
        *) echo UNKNOWN ;;
    esac
}

task_env_id() {
    local f
    for f in $(task_env "$1"); do
        case $f in --env_id=*) echo "${f#--env_id=}" ;; esac
    done
}
