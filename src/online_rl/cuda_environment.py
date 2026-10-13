"""Local linker aliases for hosts exposing libcuda.so.1 without libcuda.so."""
import os
from pathlib import Path


def configure_cuda_linker(root):
    if os.environ.get('TRITON_LIBCUDA_PATH'):return
    sources=[Path('/lib/x86_64-linux-gnu/libcuda.so.1'),Path('/usr/lib/x86_64-linux-gnu/libcuda.so.1')]
    source=next((p for p in sources if p.exists()),None)
    if source is None:return
    directory=Path(root)/'.cuda_libraries';directory.mkdir(exist_ok=True)
    for name in ('libcuda.so','libcuda.so.1'):
        target=directory/name
        if not target.exists():
            try:target.symlink_to(source)
            except FileExistsError:pass
    os.environ['TRITON_LIBCUDA_PATH']=str(directory.resolve())
