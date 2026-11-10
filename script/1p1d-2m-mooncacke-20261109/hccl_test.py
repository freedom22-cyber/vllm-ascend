import os, torch, torch.distributed as dist
rank = int(os.environ["RANK"]); world = int(os.environ["WORLD_SIZE"])
torch.npu.set_device(rank)
dist.init_process_group("hccl", rank=rank, world_size=world)
t = torch.ones(4, device=f"npu:{rank}")
dist.all_reduce(t); torch.npu.synchronize()
if rank == 0: print("ALLREDUCE OK", t.cpu().tolist())
dist.destroy_process_group()
