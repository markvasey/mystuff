#!/usr/bin/env python3
import torch
import time

def main():
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. Cannot run GPU load test.")
        return

    device = torch.device('cuda')
    print("Using Device:", torch.cuda.get_device_name(0))
    print("Allocating large tensors on the GPU...")
    
    # 8000x8000 float32 matrices use about 256MB VRAM each
    a = torch.randn(8000, 8000, device=device)
    b = torch.randn(8000, 8000, device=device)
    
    print("Running heavy matrix multiplication loop for 30 seconds...")
    print("Open 'nvtop' now to see the RTX 5060 Ti usage spikes!")
    
    start_time = time.time()
    count = 0
    while time.time() - start_time < 30:
        # Perform matrix multiplication
        c = torch.matmul(a, b)
        # Force CUDA synchronization so the CPU waits for the GPU to finish
        torch.cuda.synchronize()
        count += 1
        if count % 100 == 0:
            elapsed = time.time() - start_time
            print(f"[{elapsed:.1f}s] Completed {count} matrix multiplications...")
            
    print(f"\nDone! Executed {count} matrix multiplications in 30 seconds.")

if __name__ == "__main__":
    main()
