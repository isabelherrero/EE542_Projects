import csv
import os
import re
import statistics
import subprocess


ms = [512, 1024, 2048]
ns = [3, 5, 7]
filters = ["blur", "edge", "sharpen"]

runs = 5

os.makedirs("results_python", exist_ok=True)

rows = []


for name in filters:
    for m in ms:
        for n in ns:
            gpu_times = []
            total_times = []

            inp = f"image_{m}.tiff"
            out = f"results_python/{name}_{m}_{n}.tiff"

            for run in range(runs):
                cmd = [
                    "python3",
                    "test_convolution.py",
                    inp,
                    out,
                    name,
                    str(n)
                ]

                p = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True
                )

                gpu = re.search(
                    r"GPU kernel time:\s+([\d.]+)",
                    p.stdout
                )

                total = re.search(
                    r"Python library time:\s+([\d.]+)",
                    p.stdout
                )

                if gpu and total:
                    gpu_times.append(
                        float(gpu.group(1))
                    )

                    total_times.append(
                        float(total.group(1))
                    )

            gpu_med = statistics.median(
                gpu_times
            )

            total_med = statistics.median(
                total_times
            )

            rows.append([
                name,
                m,
                n,
                gpu_med,
                total_med
            ])

            print(
                f"{name:8} "
                f"M={m:4} "
                f"N={n} "
                f"kernel={gpu_med:.6f} ms "
                f"python={total_med:.6f} ms"
            )


with open(
    "results_python/timings.csv",
    "w",
    newline=""
) as f:
    writer = csv.writer(f)

    writer.writerow([
        "filter",
        "M",
        "N",
        "kernel_ms",
        "python_ms"
    ])

    writer.writerows(rows)


print("\nSaved results_python/timings.csv")
