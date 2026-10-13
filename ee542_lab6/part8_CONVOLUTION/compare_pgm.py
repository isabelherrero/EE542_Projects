import numpy as np

def read_pgm(filename):
    with open(filename, "rb") as file:
        magic = file.readline().strip()
        dimensions = file.readline().strip()
        maximum = file.readline().strip()
        data = np.frombuffer(file.read(), dtype=np.uint8)

    return magic, dimensions, maximum, data

_, _, _, cpu = read_pgm("cpu_output_M1024_N7.pgm")
_, _, _, gpu = read_pgm("python_gpu_output_M1024_N7.pgm")

difference = np.abs(
    cpu.astype(np.int16) -
    gpu.astype(np.int16)
)

print(f"Total pixels: {cpu.size}")
print(f"Different pixels: {np.count_nonzero(difference)}")
print(f"Maximum pixel difference: {difference.max()}")
print(f"Mean pixel difference: {difference.mean():.10f}")
print(f"Results match within 1: {np.all(difference <= 1)}")
