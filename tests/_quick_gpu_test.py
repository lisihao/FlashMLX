import mlx.core as mx
import time

print(f"Device: {mx.default_device()}")
a = mx.random.normal((1000, 1000))
mx.eval(a)
print(f"Random OK: {a[0,0].item():.4f}")
b = a @ a
mx.eval(b)
print(f"Matmul OK: {b[0,0].item():.4f}")
print("GPU works fine")
