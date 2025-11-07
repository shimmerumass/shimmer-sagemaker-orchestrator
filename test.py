import h5py
import numpy as np

def compare_mats_h5py(file1, file2, tol=1e-6):
    with h5py.File(file1, 'r') as f1, h5py.File(file2, 'r') as f2:
        keys1 = set(f1.keys())
        keys2 = set(f2.keys())

        print(f"Variables in {file1} not in {file2}: {keys1 - keys2}")
        print(f"Variables in {file2} not in {file1}: {keys2 - keys1}")

        common_keys = keys1.intersection(keys2)
        for key in common_keys:
            data1 = f1[key][()]
            data2 = f2[key][()]
            print(f"\nComparing variable '{key}':")
            print(f"  Shape in {file1}: {data1.shape if hasattr(data1, 'shape') else 'N/A'}")
            print(f"  Shape in {file2}: {data2.shape if hasattr(data2, 'shape') else 'N/A'}")
            try:
                if np.allclose(data1, data2, atol=tol):
                    print(f"  Variable '{key}' matches.")
                else:
                    print(f"  Variable '{key}' differs.")
            except Exception as e:
                # For non-numeric data, fallback to simple comparison
                if data1 == data2:
                    print(f"  Variable '{key}' matches.")
                else:
                    print(f"  Variable '{key}' differs.")

# Example usage
file1 = 'touch_detection_results.mat'
file2 = 'touchEvent.mat'
compare_mats_h5py(file1, file2)
