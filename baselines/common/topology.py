"""Role placement only; batch size and scientific recipe remain independent."""
def is_four(spec):return spec.get('execution_topology')=='doctor2_assistant2_v1'
def assistant_devices(spec,runtime):
    return list(spec['assistant_physical_gpus']) if is_four(spec) else list(runtime['training_physical_gpus'])
def doctor_devices(spec,runtime):
    return list(spec['doctor_physical_gpus']) if is_four(spec) else list(runtime['training_physical_gpus'])
def owned_devices(spec,runtime):
    if is_four(spec):return doctor_devices(spec,runtime)+assistant_devices(spec,runtime)
    return assistant_devices(spec,runtime)+([str(spec['reference_physical_gpu'])] if spec.get('reference_physical_gpu') is not None else [])
