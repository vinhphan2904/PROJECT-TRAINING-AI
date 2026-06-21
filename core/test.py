import torch
import time
from cnn import MyCNN
device = torch.device("mps")

model = MyCNN().to(device)

for bs in [16, 32, 64, 128]:
    try:
        x = torch.randn(bs, 3, 128, 128).to(device)

        start = time.time()
        y = model(x)
        torch.mps.synchronize()  # rất quan trọng cho Mac
        end = time.time()

        print(f"batch_size={bs} OK, time={end-start:.4f}s")

    except Exception as e:
        print(f"batch_size={bs} FAIL → {e}")