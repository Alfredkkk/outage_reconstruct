import numpy as np
import ruptures as rpt
import time
import concurrent.futures as cf


def gen_one_signal(length=1000):
    signal = [0] * length
    for i in range(1, length):
        if np.random.rand() < 0.5:
            signal[i] = np.random.normal(0, 1)
        else:
            signal[i] = signal[i - 1]
            
    signal = np.array(signal)
    config = {
        "PELF": {
            "algo": rpt.Pelt(model="rbf", jump=1, min_size=1),
            "time": None,
            "bkps": []
        },
        "KernelCPD": {
            "algo": rpt.KernelCPD(kernel="rbf", min_size=1),
            "time": None,
            "bkps": []
        }
    }
    print("rbfing")
    for name in config:
        start = time.time()
        config[name]["bkps"] = config[name]["algo"].fit(signal).predict(pen=100)
        config[name]["time"] = time.time() - start

    # Verify results match
    python_bkps = config["PELF"]["bkps"]
    c_bkps = config["KernelCPD"]["bkps"]
    exact_match = python_bkps == c_bkps

    print(f"Python breakpoints: {python_bkps}")
    print(f"C breakpoints: {c_bkps}")
    print(f"Exact match: {exact_match}")
    print("\nPerformance:")
    for name in config:
        print(f"{name}: {config[name]['time']:.6f} seconds")
    return exact_match


def main():
# Generate reproducible test signal
    np.random.seed(42)
    # samples = generate_signal(n_samples=100)

    # Algorithm configuration
    # n_bkps = 3  # Request 3 breakpoints (4 segments)
    

    # Run detection and measure performance
    futures = []
    with cf.ProcessPoolExecutor(max_workers=16) as executor:
        
        for i in range(10):
            futures.append(executor.submit(gen_one_signal, np.random.randint(500, 2000)))
        
        for future in cf.as_completed(futures):
            exact_match = future.result()

        # Run each algorithm on the signal
            
            assert exact_match, "Breakpoints do not match!"

        
if __name__ == "__main__":
    main()