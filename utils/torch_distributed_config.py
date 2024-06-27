import logging
import os
import pickle
import random
import shutil
import subprocess

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import datetime

from torch._utils import _flatten_dense_tensors, _take_tensors, _unflatten_dense_tensors
from collections import OrderedDict


def init_distributed_device(launcher, tcp_port, local_rank=None, backend="nccl"):
    """
    modified from https://github.com/open-mmlab/mmdetection
    Args:
        tcp_port:
        backend:

    Returns:

    """
    if launcher == "slurm":
        logging.info(f"config distributed training with launcher: {launcher}")
        proc_id = int(os.environ["SLURM_PROCID"])
        ntasks = int(os.environ["SLURM_NTASKS"])
        node_list = os.environ["SLURM_NODELIST"]
        num_gpus = torch.cuda.device_count()
        torch.cuda.set_device(proc_id % num_gpus)
        addr = subprocess.getoutput(
            "scontrol show hostname {} | head -n1".format(node_list)
        )
        os.environ["MASTER_PORT"] = str(tcp_port)
        os.environ["MASTER_ADDR"] = addr
        os.environ["WORLD_SIZE"] = str(ntasks)
        os.environ["RANK"] = str(proc_id)
        dist.init_process_group(backend=backend)

        total_gpus = dist.get_world_size()
        rank = dist.get_rank()
        return total_gpus, rank
    elif launcher == "pytorch":
        logging.info(f"config distributed training with launcher: {launcher}")
        assert local_rank is not None
        if mp.get_start_method(allow_none=True) is None:
            mp.set_start_method("spawn")

        num_gpus = torch.cuda.device_count()
        logging.info("Available GPUs:{}".format(num_gpus))
        logging.info("Using TCP Port:{}".format(tcp_port))

        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank % num_gpus)
        logging.info("local_rank:{}".format(local_rank))
        dist.init_process_group(
            backend=backend,
            # init_method='tcp://localhost:%d' % tcp_port,
            rank=local_rank,
            world_size=num_gpus,
            timeout=datetime.timedelta(seconds=60),
        )

        rank = dist.get_rank()
        os.environ["WORLD_SIZE"] = str(num_gpus)
        os.environ["RANK"] = str(local_rank)
        return num_gpus, rank
    else:
        raise NotImplementedError


def get_dist_info():
    if torch.__version__ < "1.0":
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def merge_results_dist(result_part, size, tmpdir):
    rank, world_size = get_dist_info()
    os.makedirs(tmpdir, exist_ok=True)

    dist.barrier()
    pickle.dump(
        result_part, open(os.path.join(tmpdir, "result_part_{}.pkl".format(rank)), "wb")
    )
    dist.barrier()

    if rank != 0:
        return None

    part_list = []
    for i in range(world_size):
        part_file = os.path.join(tmpdir, "result_part_{}.pkl".format(i))
        part_list.append(pickle.load(open(part_file, "rb")))

    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    ordered_results = ordered_results[:size]
    shutil.rmtree(tmpdir)
    return ordered_results


def _allreduce_coalesced(tensors, world_size, bucket_size_mb=-1):
    if bucket_size_mb > 0:
        bucket_size_bytes = bucket_size_mb * 1024 * 1024
        buckets = _take_tensors(tensors, bucket_size_bytes)
    else:
        buckets = OrderedDict()
        for tensor in tensors:
            tp = tensor.type()
            if tp not in buckets:
                buckets[tp] = []
            buckets[tp].append(tensor)
        buckets = buckets.values()

    for bucket in buckets:
        flat_tensors = _flatten_dense_tensors(bucket)
        dist.all_reduce(flat_tensors)
        flat_tensors.div_(world_size)
        for tensor, synced in zip(
            bucket, _unflatten_dense_tensors(flat_tensors, bucket)
        ):
            tensor.copy_(synced)


def allreduce_grads(params, coalesce=True, bucket_size_mb=-1):
    """Allreduce gradients.

    Args:
        params (list[torch.Parameters]): List of parameters of a model
        coalesce (bool, optional): Whether allreduce parameters as a whole.
            Defaults to True.
        bucket_size_mb (int, optional): Size of bucket, the unit is MB.
            Defaults to -1.
    """
    grads = [
        param.grad.data
        for param in params
        if param.requires_grad and param.grad is not None
    ]
    world_size = dist.get_world_size()
    if coalesce:
        _allreduce_coalesced(grads, world_size, bucket_size_mb)
    else:
        for tensor in grads:
            dist.all_reduce(tensor.div_(world_size))
