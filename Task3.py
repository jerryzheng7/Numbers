import numpy as np
import time
import matplotlib.pyplot as plt
import pandas as pd

#data types
ints = [np.int8, np.int16, np.int32, np.int64]
floats = [np.float16, np.float32, np.float64]

#function for timing operations
def times(func, *args, n=10):
    start_time = time.time()
    for _ in range(n):
        func(*args)
    end_time = time.time()
    return (end_time-start_time)/n

#add array operation
def add(array1,array2):
    return array1+array2

#multiply array operation
def multiply(array1,array2):
    return array1*array2

#add array using for loops
def py_add(array1,array2):
    result=[]
    for i in range(len(array1)):
        result.append(array1[i]+array2[i])
    return result

#multiply array using for loops
def py_multiply(array1,array2):
    result=[]
    for i in range(len(array1)):
        result.append(array1[i] * array2[i])
    return result

#generates arrays with random numbers for each data type
size = 10
np_arrays = {dtype: np.random.randint(0, 10, size=size).astype(dtype) for dtype in ints}
np_arrays.update({dtype: np.random.rand(size).astype(dtype) for dtype in floats})
python_int_array = list(np.random.randint(0, 10, size=size))
python_float_array = list(np.random.rand(size))

#populates each array with each data type
results = {}
for dtype, array in np_arrays.items():
    array2 = array.copy()
    results[dtype] = {}
    results[dtype]['add'] = times(add, array, array2)
    results[dtype]['multiply'] = times(multiply, array, array2)
results['python_int'] = {'add': times(py_add, python_int_array, python_int_array),
    'multiply': times(py_multiply, python_int_array, python_int_array)}
results['python_float'] = {'add': times(py_add, python_float_array, python_float_array),
    'multiply': times(py_multiply, python_float_array, python_float_array)}

# Function to plot bar results
def plot_results(results,title):
    fig, ax = plt.subplots(layout='constrained')
    dtypes = list(results.keys())
    indices = np.arange(len(dtypes))
    bar_width = 0.2
    add_times = [results[dtype]['add'] for dtype in dtypes]
    multiply_times = [results[dtype]['multiply'] for dtype in dtypes]
    ax.bar(indices, add_times, bar_width, label='Add')
    ax.bar(indices+bar_width, multiply_times,bar_width,label='Multiply')
    ax.set_ylabel('Time(ms)')
    ax.set_title(title)
    ax.set_xticks(indices+bar_width/2)
    ax.set_xticklabels([str(dtype) for dtype in dtypes], rotation=90)
    ax.legend()
    plt.show()

#table generation
def table(results):
    data = []
    for dtype, ops in results.items():
        data.append([str(dtype), ops['add'], ops['multiply']])
    df = pd.DataFrame(data, columns=['Data Type', 'Add(ms)', 'Multiply(ms)'])
    return df
plot_results(results, 'Comparison of Operations Across Data Types')
df = table(results)
print(df)