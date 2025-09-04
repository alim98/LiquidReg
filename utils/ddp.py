# utils/ddp.py
import os
import torch
import torch.distributed as dist

def _cuda_or_cpu():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def ddp_is_available() -> bool:
    return dist.is_available()

def is_distributed() -> bool:
    return ddp_is_available() and dist.is_initialized()

def init_distributed(backend: str = "nccl"):
    """
    Initialize torch.distributed using torchrun env vars.
    Returns (is_ddp, local_rank, device).
    """
    if not ddp_is_available():
        return False, 0, _cuda_or_cpu()

    # Detect torchrun
    if "RANK" not in os.environ or "WORLD_SIZE" not in os.environ:
        return False, 0, _cuda_or_cpu()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, init_method="env://")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    return True, local_rank, device

def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1

def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0

def is_main_process() -> bool:
    return get_rank() == 0

def barrier():
    if is_distributed():
        dist.barrier()

@torch.no_grad()
def reduce_mean(t: torch.Tensor) -> torch.Tensor:
    """Average a scalar tensor across all processes; returns a new tensor."""
    if not is_distributed():
        return t.detach()
    rt = t.detach().clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= get_world_size()
    return rt

def cleanup():
    if is_distributed():
        dist.destroy_process_group()
