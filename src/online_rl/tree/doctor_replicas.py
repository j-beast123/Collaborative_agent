"""Runtime-only placement and queue capacity of identical local Doctor replicas."""
import queue
from urllib.parse import urlparse


def endpoint_groups(cfg):
    fg=cfg['five_gpu']
    primary=fg['doctor_urls']
    extras=fg.get('doctor_replica_urls',[[] for _ in primary])
    if len(extras)!=len(primary) or any(not isinstance(row,list) for row in extras):
        raise ValueError('Doctor replica URLs need one list per GPU')
    groups=[[url,*row] for url,row in zip(primary,extras)]
    flat=[url for group in groups for url in group]
    if len(set(flat))!=len(flat):raise ValueError('Doctor endpoints must be distinct across replicas and GPUs')
    capacity=fg.get('doctor_concurrency',1)
    for group in groups:
        if capacity % len(group):raise ValueError('Doctor concurrency must be divisible by replica count on each GPU')
        for url in group:
            parsed=urlparse(url)
            if parsed.scheme!='http' or parsed.hostname not in ('127.0.0.1','localhost') or not parsed.port:
                raise ValueError('Doctor replicas require local HTTP endpoints with explicit ports')
    return groups


def bind_backend(cfg,gpu):
    backend=cfg['rollout']['doctor_backend']
    urls=endpoint_groups(cfg)[gpu]
    backend['endpoint']=urls[0]
    backend['runtime_endpoints']=urls


def bind_queue(doctor,cfg,gpu):
    urls=endpoint_groups(cfg)[gpu]
    capacity=cfg['five_gpu'].get('doctor_concurrency',1)
    doctor.endpoints=list(urls)
    doctor._free=queue.Queue()
    for _ in range(capacity//len(urls)):
        for url in urls:doctor._free.put(url)
