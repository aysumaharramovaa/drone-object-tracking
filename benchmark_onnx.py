import argparse
import csv
import os
import platform
import time

import numpy as np
import onnxruntime as ort


def ort_latency(path, threads=0, runs=30, warmup=5):
    so = ort.SessionOptions()
    if threads:
        so.intra_op_num_threads = threads
    sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    x = np.random.rand(*shape).astype(np.float16 if "float16" in inp.type else np.float32)
    for _ in range(warmup):                       # ilk işə salmalar yavaş olur
        sess.run(None, {inp.name: x})
    ts = []
    for _ in range(runs):
        t = time.perf_counter()
        sess.run(None, {inp.name: x})
        ts.append((time.perf_counter() - t) * 1000)
    return float(np.median(ts)), float(np.percentile(ts, 95)), shape


def main():
    p = argparse.ArgumentParser(description="ONNX gecikmə ölçümü (CPU)")
    p.add_argument("models", nargs="+", help="ONNX fayllarının yolları")
    p.add_argument("--threads", type=int, nargs="+", default=[1, 2, 4],
                   help="Sınanacaq thread sayları (0 = avtomatik)")
    p.add_argument("--runs", type=int, default=30, help="Ölçülən işə salma sayı")
    p.add_argument("--warmup", type=int, default=5, help="İsinmə işə salmaları (ölçülmür)")
    p.add_argument("--csv", default="onnx_latency.csv", help="Nəticə CSV faylı")
    args = p.parse_args()

    print(f"Cihaz: {platform.processor() or platform.machine()} | məntiqi nüvə sayı: {os.cpu_count()} | "
          f"onnxruntime {ort.__version__}\n")

    rows = []
    print(f"{'model':<28}{'MB':>7}{'threads':>9}{'median ms':>11}{'p95 ms':>9}{'FPS':>8}")
    for path in args.models:
        size = os.path.getsize(path) / 1e6
        for t in args.threads:
            med, p95, shape = ort_latency(path, t, args.runs, args.warmup)
            fps = 1000 / med
            print(f"{os.path.basename(path):<28}{size:>7.1f}{t:>9}{med:>11.1f}{p95:>9.1f}{fps:>8.2f}")
            rows.append([os.path.basename(path), round(size, 2), t, round(med, 2), round(p95, 2), round(fps, 2)])

    with open(args.csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["model", "size_MB", "threads", "median_ms", "p95_ms", "FPS"])
        w.writerows(rows)
    print(f"\nYazıldı: {args.csv}")


if __name__ == "__main__":
    main()
