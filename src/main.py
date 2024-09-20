import time
import torch
from mvit import MViT_V2

if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MViT_V2(10).to(device)
    x = torch.rand(1, 3, 16, 224, 224).to(device)

    times = []
    for i in range(100):
        tic = time.time()
        y = model(x)
        toc = time.time()
        times.append(toc - tic)
    avg = sum(times) / len(times)
    print(f"Average time: {avg:.4f} s")
    print(f"FPS: {1 / avg:.4f}")
