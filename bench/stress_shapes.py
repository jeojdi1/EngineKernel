import os,sys,time,torch,traceback
sys.path.insert(0,"engine")
hold_gb=float(os.environ.get("HOLD_GB","0"))
hog=torch.empty(int(hold_gb*(1<<30)),dtype=torch.uint8,device="cuda") if hold_gb else None
free,total=torch.cuda.mem_get_info(); print(f"simulated card: {(total-hold_gb*(1<<30))/(1<<30):.0f} GiB usable of {total/(1<<30):.0f}, free now {free/(1<<30):.0f}")
from engine import Engine
eng=Engine(os.path.expanduser("~/qwen3-4b"))
print(f"cache {eng.cache_b}x{eng.cache_s}, allocated {torch.cuda.memory_allocated()/(1<<30):.1f} GiB")
for b,s,n in [(16,512,128),(64,512,128),(64,2048,64),(32,4096,64),(128,512,64),(8,8192,32),(1,16000,64)]:
    ids=[[(1000+r+i)%150000 for i in range(s)] for r in range(b)]
    torch.cuda.reset_peak_memory_stats()
    try:
        t0=time.perf_counter(); out=list(eng.generate(ids,n)); torch.cuda.synchronize()
        ok = len(out)==n and all(len(o)==b for o in out)
        print(f"  b={b:3d} {s:5d}->{n:3d}: {'ok ' if ok else 'BAD'} {b*n/(time.perf_counter()-t0):8.1f} tok/s  peak {torch.cuda.max_memory_allocated()/(1<<30):5.1f} GiB")
    except Exception as e:
        print(f"  b={b:3d} {s:5d}->{n:3d}: CRASH {type(e).__name__}: {str(e)[:110]}")
