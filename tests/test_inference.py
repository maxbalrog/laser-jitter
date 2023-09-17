import torch
from laser_jitter.inference import calculate_metrics

size = (32,100,2)
a = torch.rand(size)
b = torch.rand(size)

def test_calculate_metrics():
    mae, rms = calculate_metrics(a, b)

