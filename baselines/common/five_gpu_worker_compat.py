"""Original outcome-only worker with a scoped legacy Transformers dtype alias."""
from src.online_rl.tree import five_gpu_worker as original
from baselines.common.policy_loader import compatible_dtype
prepare=original.prepare
run_many=original.run_many


def initialize(*args,**kwargs):
    with compatible_dtype():return original.initialize(*args,**kwargs)
