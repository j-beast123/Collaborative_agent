"""Preserve original SharedPolicy; support the installed legacy BF16 keyword."""
from contextlib import contextmanager
import threading
_LOCK=threading.RLock()


@contextmanager
def compatible_dtype():
    import transformers
    version=tuple(int(p) for p in transformers.__version__.split('.')[:2])
    if version>=(4,56):
        yield
        return
    # Older Transformers forwards unknown dtype to the model constructor;
    # torch_dtype is its equivalent supported keyword. No model/dtype change.
    cls=transformers.AutoModelForCausalLM
    with _LOCK:
        previous=cls.__dict__.get('from_pretrained');native=cls.from_pretrained
        def compatible(*args,**kwargs):
            if 'dtype' in kwargs:kwargs['torch_dtype']=kwargs.pop('dtype')
            return native(*args,**kwargs)
        cls.from_pretrained=staticmethod(compatible)
        try:yield
        finally:
            if previous is None:delattr(cls,'from_pretrained')
            else:cls.from_pretrained=previous


def load_policy(cfg,key,*,device='cuda:0'):
    from src.online_rl.runtime import SharedPolicy
    with compatible_dtype():return SharedPolicy(cfg,key,device=device)
