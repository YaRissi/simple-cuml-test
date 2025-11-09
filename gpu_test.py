# gpu_test.py
import time
import numpy as np
import cupy as cp
import cudf
import pandas as pd
from cuml.cluster import KMeans as cuKMeans
from sklearn.cluster import KMeans as skKMeans
from cuml.datasets import make_blobs

print("=" * 80)
print("GPU vs CPU Performance Test")
print("=" * 80)

# Test 1: Array Operations
print("\nTest 1: Large Array Operations (100M elements)")
print("-" * 80)

size = 100_000_000

# CPU with NumPy
print("Running on CPU (NumPy)...")
start = time.time()
cpu_array = np.random.rand(size)
cpu_result = np.sqrt(cpu_array) * np.sin(cpu_array) + np.cos(cpu_array)
cpu_mean = np.mean(cpu_result)
cpu_time = time.time() - start
print(f"  CPU Time: {cpu_time:.3f} seconds")
print(f"  Result mean: {cpu_mean:.6f}")

# GPU with CuPy
print("\nRunning on GPU (CuPy)...")
start = time.time()
gpu_array = cp.random.rand(size)
gpu_result = cp.sqrt(gpu_array) * cp.sin(gpu_array) + cp.cos(gpu_array)
gpu_mean = float(cp.mean(gpu_result))
gpu_time = time.time() - start
print(f"  GPU Time: {gpu_time:.3f} seconds")
print(f"  Result mean: {gpu_mean:.6f}")

speedup = cpu_time / gpu_time
print(f"\nSpeedup: {speedup:.2f}x faster on GPU")

# Test 2: DataFrame Operations
print("\n" + "=" * 80)
print("Test 2: DataFrame Operations (10M rows)")
print("-" * 80)

n_rows = 10_000_000

# CPU with Pandas
print("Running on CPU (Pandas)...")
start = time.time()
df_cpu = pd.DataFrame({
    'a': np.random.rand(n_rows),
    'b': np.random.rand(n_rows),
    'c': np.random.randint(0, 100, n_rows)
})
df_cpu['d'] = df_cpu['a'] * df_cpu['b']
df_cpu['e'] = df_cpu['c'] ** 2
grouped = df_cpu.groupby('c')['d'].mean()
cpu_time = time.time() - start
print(f"  CPU Time: {cpu_time:.3f} seconds")
print(f"  Groups processed: {len(grouped)}")

# GPU with cuDF
print("\nRunning on GPU (cuDF)...")
start = time.time()
df_gpu = cudf.DataFrame({
    'a': cp.random.rand(n_rows),
    'b': cp.random.rand(n_rows),
    'c': cp.random.randint(0, 100, n_rows)
})
df_gpu['d'] = df_gpu['a'] * df_gpu['b']
df_gpu['e'] = df_gpu['c'] ** 2
grouped = df_gpu.groupby('c')['d'].mean()
gpu_time = time.time() - start
print(f"  GPU Time: {gpu_time:.3f} seconds")
print(f"  Groups processed: {len(grouped)}")

speedup = cpu_time / gpu_time
print(f"\nSpeedup: {speedup:.2f}x faster on GPU")

# Test 3: Machine Learning - K-Means Clustering
print("\n" + "=" * 80)
print("Test 3: K-Means Clustering (1M samples, 50 features)")
print("-" * 80)

n_samples = 1_000_000
n_features = 50
n_clusters = 10

print("Generating dataset...")
X_gpu, _ = make_blobs(
    n_samples=n_samples,
    n_features=n_features,
    centers=n_clusters,
    random_state=42
)
# Convert CuPy array to NumPy for CPU
X_cpu = cp.asnumpy(X_gpu)

# CPU with scikit-learn
print("\nRunning on CPU (scikit-learn)...")
start = time.time()
kmeans_cpu = skKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans_cpu.fit(X_cpu)
cpu_time = time.time() - start
print(f"  CPU Time: {cpu_time:.3f} seconds")
print(f"  Inertia: {kmeans_cpu.inertia_:.2f}")

# GPU with cuML
print("\nRunning on GPU (cuML)...")
start = time.time()
kmeans_gpu = cuKMeans(n_clusters=n_clusters, random_state=42, n_init=10)
kmeans_gpu.fit(X_gpu)
gpu_time = time.time() - start
print(f"  GPU Time: {gpu_time:.3f} seconds")
print(f"  Inertia: {float(kmeans_gpu.inertia_):.2f}")

speedup = cpu_time / gpu_time
print(f"\nSpeedup: {speedup:.2f}x faster on GPU")

# Test 4: Matrix Operations
print("\n" + "=" * 80)
print("Test 4: Large Matrix Multiplication (5000x5000)")
print("-" * 80)

size = 5000

# CPU with NumPy
print("Running on CPU (NumPy)...")
start = time.time()
A_cpu = np.random.rand(size, size)
B_cpu = np.random.rand(size, size)
C_cpu = np.dot(A_cpu, B_cpu)
cpu_time = time.time() - start
print(f"  CPU Time: {cpu_time:.3f} seconds")

# GPU with CuPy
print("\nRunning on GPU (CuPy)...")
start = time.time()
A_gpu = cp.random.rand(size, size)
B_gpu = cp.random.rand(size, size)
C_gpu = cp.dot(A_gpu, B_gpu)
cp.cuda.Stream.null.synchronize()  # Ensure GPU finishes
gpu_time = time.time() - start
print(f"  GPU Time: {gpu_time:.3f} seconds")

speedup = cpu_time / gpu_time
print(f"\nSpeedup: {speedup:.2f}x faster on GPU")

# GPU Info
print("\n" + "=" * 80)
print("GPU Information")
print("=" * 80)
print(f"GPU Name: {cp.cuda.Device()}")
print(f"CUDA Version: {cp.__version__}")
total_mem = cp.cuda.Device().mem_info[1] / 1024**3
free_mem = cp.cuda.Device().mem_info[0] / 1024**3
used_mem = total_mem - free_mem
print(f"Total GPU Memory: {total_mem:.2f} GB")
print(f"Used GPU Memory: {used_mem:.2f} GB")
print(f"Free GPU Memory: {free_mem:.2f} GB")

print("\n" + "=" * 80)
print("All tests completed successfully!")
print("=" * 80)