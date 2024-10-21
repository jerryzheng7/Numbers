import numpy as np
import json
import pickle
import pandas as pd
import os
import time

#Store Array Function
def store_array(N, M, rnd, dtype, filename, filetype):
    #Define N, M, and rnd
    shape = tuple([N] * M)
    A = rnd(low=0, high=10, size=shape).astype(dtype)
    start_save_time = time.time()
    #Define Filetypes
    if filetype == 'numpy':
        np.save(filename, A)
    elif filetype == 'txt':
        with open(filename, 'w') as f:
            if A.ndim > 2:
                for i, sub_array in enumerate(A):
                    f.write(f"# Slice {i+1}\n")
                    np.savetxt(f, sub_array, fmt='%d' if np.issubdtype(dtype, np.integer) else '%.2f')
                    f.write("\n")
            else:
                np.savetxt(f, A, fmt='%d' if np.issubdtype(dtype, np.integer) else '%.2f')
    elif filetype == 'json':
        with open(filename, 'w') as f:
            json.dump(A.tolist(), f)
    elif filetype == 'pickle':
        with open(filename, 'wb') as f:
            pickle.dump(A, f)
    elif filetype == 'csv':
        df = pd.DataFrame(A.reshape(-1, A.shape[-1]))
        df.to_csv(filename, index=False)

    end_save_time = time.time()
    save = end_save_time - start_save_time
    size = os.path.getsize(filename)
    start_load_time = time.time()

    #Load Filetypes
    if filetype == 'numpy':
        Aretrieved = np.load(filename)
    elif filetype == 'txt':
        with open(filename, 'r') as f:
            Aretrieved = np.loadtxt(f).reshape(shape)
    elif filetype == 'json':
        with open(filename, 'r') as f:
            Aretrieved = np.array(json.load(f))
    elif filetype == 'pickle':
        with open(filename, 'rb') as f:
            Aretrieved = pickle.load(f)
    elif filetype == 'csv':
        df = pd.read_csv(filename)
        Aretrieved = df.values.reshape(shape)

    end_load_time = time.time()
    load = end_load_time - start_load_time
    mse = float(np.mean((save - load) ** 2)) #Calculates MSE
    os.remove(filename)
    metrics = {'save': save, 'load': load, 'size': size, 'MSE': mse}
    return A, Aretrieved, metrics

arrays, result, metrics = store_array(150, 1, np.random.randint, np.float32, 'test', 'csv')
print("Save:\n", arrays)
print("\nLoad:\n", result)
print("\nPerformance:\n", metrics)
