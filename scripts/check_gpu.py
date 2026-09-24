# Week 1 item 1: verify torch.cuda on RTX 5080 (sm_120). Run: .venv/bin/python scripts/check_gpu.py
import time, torch
print("torch", torch.__version__, "| torch cuda build", torch.version.cuda, "| cudnn", torch.backends.cudnn.version())
print("cuda available:", torch.cuda.is_available(), "| device count:", torch.cuda.device_count())
p = torch.cuda.get_device_properties(0)
print("device:", p.name, "| capability:", torch.cuda.get_device_capability(0), "| total mem GiB: %.2f" % (p.total_memory/2**30), "| SMs:", p.multi_processor_count)
print("arch list:", torch.cuda.get_arch_list())
print("bf16 supported:", torch.cuda.is_bf16_supported())
torch.manual_seed(0)
a = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
b = torch.randn(2048, 2048, device="cuda", dtype=torch.bfloat16)
c = a @ b; torch.cuda.synchronize()
ref = a.float().cpu() @ b.float().cpu()
err = (c.float().cpu() - ref).abs().max().item() / ref.abs().max().item()
print("bf16 matmul 2048^2 OK | out dtype:", c.dtype, "| max rel err vs fp32 cpu: %.4f" % err)
for _ in range(3): c = a @ b
torch.cuda.synchronize(); t = time.perf_counter(); n = 50
for _ in range(n): c = a @ b
torch.cuda.synchronize(); dt = (time.perf_counter()-t)/n
print("bf16 matmul avg %.3f ms -> %.1f TFLOP/s" % (dt*1e3, 2*2048**3/dt/1e12))
x = torch.randn(4, 3, 64, 64, device="cuda", dtype=torch.bfloat16); w = torch.randn(8, 3, 3, 3, device="cuda", dtype=torch.bfloat16)
y = torch.nn.functional.conv2d(x, w); torch.cuda.synchronize(); print("cudnn bf16 conv OK", tuple(y.shape))
