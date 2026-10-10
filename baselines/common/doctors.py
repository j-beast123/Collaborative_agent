"""Stop only this method's verified GPU/port/executable Doctor processes."""
import os
import shutil
from pathlib import Path
from baselines.common.topology import doctor_devices


def assert_endpoints_unused(cfg):
    """Read-only guard: never adopt/restart an already-running Doctor server."""
    from urllib.parse import urlparse
    requested={urlparse(url).port for url in cfg['five_gpu']['doctor_urls']}
    for table in ('/proc/net/tcp','/proc/net/tcp6'):
        for line in Path(table).read_text().splitlines()[1:]:
            fields=line.split()
            port=int(fields[1].split(':')[1],16)
            if fields[3]=='0A' and port in requested:
                raise RuntimeError(f'Doctor port {port} is already listening; existing process left untouched')
    root=Path(cfg['output_dir'])/'doctor_servers'
    for path in root.glob('gpu*.pid'):
        pid=int(path.read_text())
        if (Path('/proc')/str(pid)).exists():
            raise RuntimeError(f'Existing Doctor PID {pid} referenced by {path}; left untouched')


def stop(spec,runtime):
    root=Path(spec['output_dir'])/'doctor_servers'
    binary=runtime.get('ollama_binary')
    binary=str(Path(shutil.which(binary) or binary).resolve()) if binary else None
    for i,device in enumerate(doctor_devices(spec,runtime)):
        path=root/f'gpu{i}.pid'
        if not path.exists():continue
        pid=int(path.read_text());proc=Path('/proc')/str(pid)
        try:
            env=proc.joinpath('environ').read_bytes().split(b'\0')
            if (f'CUDA_VISIBLE_DEVICES={device}'.encode() not in env or
                f'OLLAMA_HOST=127.0.0.1:{spec["doctor_port_start"]+i}'.encode() not in env or
                str(proc.joinpath('exe').resolve())!=binary):
                raise RuntimeError(f'Refusing to stop unowned Doctor PID {pid}')
            os.kill(pid,15)
        except FileNotFoundError:pass


def start(cfg):
    if 'four_gpu' not in cfg:
        from scripts.online_rl.start_five_gpu_doctors import start as original
        return original(cfg)
    import copy,importlib.util
    from baselines.common.protocol import ROOT
    path=ROOT/'scripts/online_rl/start_five_gpu_doctors.py'
    spec=importlib.util.spec_from_file_location('comparison_doctor_servers',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    # Original server ownership/model checks, with an explicit role-specific map.
    module.physical_gpus=lambda:cfg['four_gpu']['doctor_physical_gpus']
    local=copy.deepcopy(cfg);local['five_gpu']['doctor_gpu_ids']=[0,1]
    return module.start(local)
