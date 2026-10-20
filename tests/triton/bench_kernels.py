"""
Microbenchmarks for the Triton kernel backend.

Run directly (not under pytest):  python tests/triton/bench_kernels.py
Reports TFLOP/s next to a torch reference where one exists.
"""

import importlib.util, sys, time, os
from pathlib import Path
import torch

D = Path("nunchaku/ops/triton")
pkg = importlib.util.module_from_spec(importlib.util.spec_from_loader("_t", loader=None, is_package=True))
pkg.__path__=[str(D)]; sys.modules["_t"]=pkg
def load(n):
    s=importlib.util.spec_from_file_location(f"_t.{n}", D/f"{n}.py")
    m=importlib.util.module_from_spec(s); sys.modules[f"_t.{n}"]=m; s.loader.exec_module(m); return m

w4a4=load("gemm_w4a4"); attn=load("attention"); w8a8=load("gemm_w8a8"); gfp16=load("gemm_fp16")

def bench(fn, iters=50, warmup=10):
    for _ in range(warmup): fn()
    torch.cuda.synchronize(); t=time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter()-t)/iters*1e3

print(f"{'kernel':<42}{'ms':>9}{'TFLOP/s':>11}")
print("-"*62)

# W4A4 GEMM, FLUX-ish shapes
for M,N,K in [(4096,3072,3072),(4096,12288,3072),(4096,3072,12288)]:
    act=torch.randint(0,256,(M,K//2),dtype=torch.uint8,device="cuda")
    wgt=torch.randint(0,256,(N,K//2),dtype=torch.uint8,device="cuda")
    a=(torch.rand(K//64,M,device="cuda")*.05+.01).to(torch.bfloat16)
    w=(torch.rand(K//64,N,device="cuda")*.05+.01).to(torch.bfloat16)
    out=torch.empty(M,N,dtype=torch.bfloat16,device="cuda")
    # NUNCHAKU_NATIVE_GEMM=0: the Triton kernel; default: the native int4-WMMA kernel on gfx11 (Triton elsewhere)
    for native, label in (("0", "triton"), ("", "default")):
        os.environ["NUNCHAKU_NATIVE_GEMM"] = native
        ms=bench(lambda: w4a4.gemm_w4a4(act,wgt,out=out,ascales=a,wscales=w,out_dtype=torch.bfloat16))
        print(f"{'w4a4 int4 M%d N%d K%d (%s)'%(M,N,K,label):<42}{ms:>9.3f}{2*M*N*K/ms/1e9:>11.1f}")
    os.environ.pop("NUNCHAKU_NATIVE_GEMM")

# W4A4 GEMM with fused epilogues: QKV + per-head RMSNorm/RoPE, MLP fc1 + GELU/next-layer quantize (FLUX.1 shapes)
M,N,K,R=4608,9216,3072,32
act=torch.randint(0,256,(M,K//2),dtype=torch.uint8,device="cuda"); wgt=torch.randint(0,256,(N,K//2),dtype=torch.uint8,device="cuda")
a=(torch.rand(K//64,M,device="cuda")*.05+.01).to(torch.bfloat16); w=(torch.rand(K//64,N,device="cuda")*.05+.01).to(torch.bfloat16)
la=torch.randn(M,R,device="cuda"); lu=(torch.randn(N,R,device="cuda")*.02).to(torch.bfloat16)
nq=torch.rand(128,device="cuda").to(torch.bfloat16); rot=torch.randn(M,64,1,2,device="cuda")
out=torch.empty(M,N,dtype=torch.bfloat16,device="cuda")
for native, label in (("0", "triton"), ("", "default")):
    os.environ["NUNCHAKU_NATIVE_GEMM"] = native
    ms=bench(lambda: w4a4.gemm_w4a4(act,wgt,out=out,ascales=a,wscales=w,lora_act_in=la,lora_up=lu,norm_q=nq,norm_k=nq,rotary_emb=rot))
    print(f"{'w4a4+qknorm M%d N%d K%d (%s)'%(M,N,K,label):<42}{ms:>9.3f}{2*M*N*K/ms/1e9:>11.1f}")
M,N,K=4608,12288,3072
act=torch.randint(0,256,(M,K//2),dtype=torch.uint8,device="cuda"); wgt=torch.randint(0,256,(N,K//2),dtype=torch.uint8,device="cuda")
a=(torch.rand(K//64,M,device="cuda")*.05+.01).to(torch.bfloat16); w=(torch.rand(K//64,N,device="cuda")*.05+.01).to(torch.bfloat16)
la=torch.randn(M,R,device="cuda"); lu=(torch.randn(N,R,device="cuda")*.02).to(torch.bfloat16)
qo=torch.empty(M,N//2,dtype=torch.uint8,device="cuda"); osc=torch.empty(N//64,M,dtype=torch.bfloat16,device="cuda")
sm=(torch.rand(N,device="cuda")+.5).to(torch.bfloat16); ld=(torch.randn(N,R,device="cuda")*.02).to(torch.bfloat16)
lao=torch.empty(M,R,device="cuda")
for native, label in (("0", "triton"), ("", "default")):
    os.environ["NUNCHAKU_NATIVE_GEMM"] = native
    ms=bench(lambda: w4a4.gemm_w4a4(act,wgt,ascales=a,wscales=w,lora_act_in=la,lora_up=lu,qout=qo,oscales=osc,smooth_factor=sm,lora_down=ld,lora_act_out=lao))
    print(f"{'w4a4+geluq M%d N%d K%d (%s)'%(M,N,K,label):<42}{ms:>9.3f}{2*M*N*K/ms/1e9:>11.1f}")
os.environ.pop("NUNCHAKU_NATIVE_GEMM")

# bf16 reference for the same shapes
for M,N,K in [(4096,3072,3072)]:
    x=torch.randn(M,K,dtype=torch.bfloat16,device="cuda"); ww=torch.randn(N,K,dtype=torch.bfloat16,device="cuda")
    ms=bench(lambda: torch.matmul(x,ww.t()))
    print(f"{'  torch bf16 same shape (reference)':<42}{ms:>9.3f}{2*M*N*K/ms/1e9:>11.1f}")
    ms=bench(lambda: gfp16.gemm_fp16(x,ww))
    print(f"{'  triton gemm_fp16 same shape':<42}{ms:>9.3f}{2*M*N*K/ms/1e9:>11.1f}")

# W8A8
M,N,K=4096,3072,3072
act=torch.randint(-127,127,(M,K),dtype=torch.int8,device="cuda")
wgt=torch.randint(-127,127,(N,K),dtype=torch.int8,device="cuda")
asc=torch.rand(M,dtype=torch.bfloat16,device="cuda"); wsc=torch.rand(N,dtype=torch.bfloat16,device="cuda")
o=torch.empty(M,N,dtype=torch.bfloat16,device="cuda")
ms=bench(lambda: w8a8.gemm_w8a8(act,wgt,o,asc,wsc))
print(f"{'w8a8 int8 M4096 N3072 K3072':<42}{ms:>9.3f}{2*M*N*K/ms/1e9:>11.1f}")

# attention
for B,H,T,Dh in [(1,24,4096,128),(2,24,1024,128)]:
    q=torch.randn(B,H,T,Dh,dtype=torch.bfloat16,device="cuda")
    k=torch.randn(B,H,T,Dh,dtype=torch.bfloat16,device="cuda")
    v=torch.randn(B,H,T,Dh,dtype=torch.bfloat16,device="cuda")
    o2=torch.empty(B,T,H*Dh,dtype=torch.bfloat16,device="cuda")
    ms=bench(lambda: attn.attention_fp16(q,k,v,o2,Dh**-0.5), iters=20)
    fl=4*B*H*T*T*Dh
    print(f"{'attention B%d H%d T%d D%d'%(B,H,T,Dh):<42}{ms:>9.3f}{fl/ms/1e9:>11.1f}")
    ms=bench(lambda: torch.nn.functional.scaled_dot_product_attention(q,k,v,scale=Dh**-0.5), iters=20)
    print(f"{'  torch SDPA (reference)':<42}{ms:>9.3f}{fl/ms/1e9:>11.1f}")

# quantize
M,K=4096,3072
x=torch.randn(M,K,dtype=torch.bfloat16,device="cuda")
ld=torch.randn(K,32,dtype=torch.bfloat16,device="cuda")
ms=bench(lambda: w4a4.quantize_w4a4_act_fuse_lora(x,lora_down=ld))
gb=(x.numel()*2+M*K//2)/ms/1e6
print(f"{'quantize_w4a4+lora M4096 K3072':<42}{ms:>9.3f}{gb:>9.1f} GB/s")
