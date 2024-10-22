import numpy as np
import sys
import time
import pandas as pd
from multiprocessing import Pool
from numba import njit
from joblib import Parallel, delayed
from math import sqrt
import matplotlib.pyplot as plt
import json
import pickle
import os




##############################################
################### Task 1 ################### (Thomas)
##############################################

######## Part A ########

#Function to return number of bytes
def num_bytes(integer):
    return integer.nbytes

#Return maximum value representable by number of bits
def max_value(num_bits):
    return 2**num_bits-1

#Return the time to count to the max number
def time_to_count(number):

    diff=0

    if number>28:
        diff=number-28
        number=28


    start=time.perf_counter_ns()

    for i in range(2**number):
        pass
        
    end=time.perf_counter_ns()

    return (end-start)*10**-9*2**diff


#Declare a variable of each type
a=np.uint8()
b=np.uint16()
c=np.uint32()
d=np.uint64()
e=np.int8()
f=np.int16()
g=np.int32()
h=np.int64()

#Array of type names
types=["uint8","uint16","uint32","uint64","int8","int16","int32","uint64"]

#Size in bytes
sizes=[num_bytes(x) for x in [a,b,c,d,e,f,g,h]]

#Max value
maxes=[max_value(x*8) for x in sizes]
for i in range(4,8):
    maxes[i]=max_value(sizes[i]*8-1)

#Time to count to max value
times=[time_to_count(x*8) for x in sizes]
for i in range(4,8):
    times[i]=time_to_count(sizes[i]*8-1)

#Convert to years
times_years=[x/(3.154*10**7) for x in times]

#Create Data Frame
int_info=pd.DataFrame([types,sizes,maxes, times, times_years],index=["Type","Size (Bytes)","Max Values","Time to Count to Max Value (s)", "Time to Count to Max Value (years)"]).T
print(int_info)

int_info.to_excel('IntegerInformation.xlsx')



######## Part B ########

#Determines datatype and generates overflow
def overflow_int(x):
    if(np.dtype(x)=="uint8"):
        return np.uint8(np.iinfo(np.dtype(x)).max+1)
    if(np.dtype(x)=="uint16"):
        return np.uint16(np.iinfo(np.dtype(x)).max+1)
    if(np.dtype(x)=="uint32"):
        return np.uint32(np.iinfo(np.dtype(x)).max+1)
    if(np.dtype(x)=="uint64"):
        return np.uint64(np.iinfo(np.dtype(x)).max+1)

    if(np.dtype(x)=="int8"):
        return np.int8(np.iinfo(np.dtype(x)).max+1)
    if(np.dtype(x)=="int16"):
        return np.int16(np.iinfo(np.dtype(x)).max+1)
    if(np.dtype(x)=="int32"):
        return np.int32(np.iinfo(np.dtype(x)).max+1)
    if(np.dtype(x)=="int64"):
        return np.int64(np.iinfo(np.dtype(x)).max+1)


#Determines datatype and generates under
def underflow_int(x):
    if(np.dtype(x)=="uint8"):
        return np.uint8(np.iinfo(np.dtype(x)).min-1)
    if(np.dtype(x)=="uint16"):
        return np.uint16(np.iinfo(np.dtype(x)).min-1)
    if(np.dtype(x)=="uint32"):
        return np.uint32(np.iinfo(np.dtype(x)).min-1)
    if(np.dtype(x)=="uint64"):
        return np.uint64(np.iinfo(np.dtype(x)).min-1)

    if(np.dtype(x)=="int8"):
        return np.int8(np.iinfo(np.dtype(x)).min-1)
    if(np.dtype(x)=="int16"):
        return np.int16(np.iinfo(np.dtype(x)).min-1)
    if(np.dtype(x)=="int32"):
        return np.int32(np.iinfo(np.dtype(x)).min-1)
    if(np.dtype(x)=="int64"):
        return np.int64(np.iinfo(np.dtype(x)).min-1)


####### Part C ########

#Returns number of bytes
def num_bytes_float(num):
    return num.nbytes

#Initialize one of each type
a=np.float16(0)
b=np.float32(0)
c=np.float64(0)

#Names
types=["float16", "float32", "float64"]

#Number of Bytes
sizes=[num_bytes_float(x) for x in [a,b,c]]

#Largest Possible Value
maxes=[float(np.finfo(np.dtype(x)).max) for x in [a,b,c]]


#Smallest Positive Possible Value
mins=[np.finfo(np.dtype(x)).smallest_subnormal for x in [a,b,c]]

#Smallest Integer Not Representable
#This is given by 2^(number of mantissa bits+1)+1
maxes_plus_one=[2**11+1, 2**24+1, 2**54+1]

#The precision is approximately the number of digits of the max value of the mantissa
#Mantissa is 10 bits, 23 bits, 53 bits in these representations
precision=[len(str(2**10)),len(str(2**23)),len(str(2**53))]

#Create Data Frame
float_info=pd.DataFrame([types,sizes, maxes, mins, maxes_plus_one, precision],index=["Type","Size (Bytes)","Max","Min", "Max Plus One", "Approx Precision"]).T
print(float_info)
float_info.to_excel('FloatInformation.xlsx')


##############################################
################### Task 2 ################### (Jerry)
##############################################

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

##############################################
################### Task 3 ###################
##############################################

######## Array Scaling ######## (Jerry)

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



################## Speedup ################## (Thomas)

#This function takes in an array, loops through it, and cubes all the odds values
def cube_odds(x):
    for i in range(len(x)):
        if x[i] % 2 ==1:
            x[i]=x[i]**3

#This is the same as the previous function, but it utilizes no-python just-in-time compilation
@njit
def speed_cube_odds(x):
    for i in range(len(x)):
        if x[i] % 2 ==1.0:
            x[i]=x[i]**3

#This function conditionally multiples the elements of one array by another
def multiply_arrays(arr1, arr2):
    arr3=arr1
    for i in range(len(arr1)):
        if arr1[i]%3==1 or arr1[i]%5==1:
            arr3[i]=arr1[i]*arr2[i]
    

#This is the same as the previous function, but it utilizes no-python just-in-time compilation
@njit
def speed_multiply_arrays(arr1, arr2):
    arr3=arr1
    for i in range(len(arr1)):
        if arr1[i]%3==1 or arr1[i]%5==1:
            arr3[i]=arr1[i]*arr2[i]

#Create 3d array to store timing data
rows, cols = (8, 6)
speedup_cube = [[[0 for _ in range(2)] for _ in range(cols)] for _ in range(rows)]

#These are the types and sizes that we will loop through
np_types=[np.int8, np.int16, np.int32, np.int64, np.float32, np.float64] #(np.float16 not implemented)
py_types=[int,float]
sizes=[10**2, 10**3, 10**4, 10**5, 10**6, 10**7]

for type in np_types:
    for size in sizes:

        #Run without speedup
        start=time.perf_counter_ns()
        a=cube_odds(np.arange(size, dtype=type))
        end=time.perf_counter_ns()
        no_speedup= end - start
        speedup_cube[np_types.index(type)][sizes.index(size)][0]=no_speedup*10**-9
        
        #Run with speedup
        start=time.perf_counter_ns()
        a=speed_cube_odds(np.arange(size, dtype=type))
        end=time.perf_counter_ns()
        speedup_cube[np_types.index(type)][sizes.index(size)][1]= (end - start)*10**-9

for type in py_types:
    for size in sizes:
        
        #Run without speedup
        start=time.perf_counter_ns()
        a=cube_odds(np.arange(size, dtype=type))
        end=time.perf_counter_ns()
        speedup_cube[py_types.index(type)+6][sizes.index(size)][0]= (end-start)*10**-9

        #Run with speedup
        start=time.perf_counter_ns()
        a=speed_cube_odds(np.arange(size, dtype=type))
        end=time.perf_counter_ns()
        speedup_cube[py_types.index(type)+6][sizes.index(size)][1]= (end-start)*10**-9


#Plot For Integers
data_int8=[speedup_cube[np_types.index(np.int8)][j][0] for j in range(6)]
data_int16=[speedup_cube[np_types.index(np.int16)][j][0] for j in range(6)]
data_int32=[speedup_cube[np_types.index(np.int32)][j][0] for j in range(6)]
data_int64=[speedup_cube[np_types.index(np.int64)][j][0] for j in range(6)]
data_py_int=[speedup_cube[6][j][0] for j in range(6)]

data_int8_speed=[speedup_cube[np_types.index(np.int8)][j][1] for j in range(6)]
data_int16_speed=[speedup_cube[np_types.index(np.int16)][j][1] for j in range(6)]
data_int32_speed=[speedup_cube[np_types.index(np.int32)][j][1] for j in range(6)]
data_int64_speed=[speedup_cube[np_types.index(np.int64)][j][1] for j in range(6)]
data_py_int_speed=[speedup_cube[6][j][1] for j in range(6)]

plt.scatter(sizes, data_int8, label='np.int8',color='red',s=20)
plt.scatter(sizes, data_int16, label='np.int16',color='blue',s=20)
plt.scatter(sizes, data_int32, label='np.int32',color='green',s=20)
plt.scatter(sizes, data_int64, label='np.int64',color='orange',s=20)
plt.scatter(sizes, data_py_int, label='python int',color='yellow',s=20)

plt.scatter(sizes, data_int8_speed, label='np.int8 with speedup',color='red',marker='x')
plt.scatter(sizes, data_int16_speed, label='np.int16 with speedup',color='blue',marker='x')
plt.scatter(sizes, data_int32_speed, label='np.int32 with speedup',color='green',marker='x')
plt.scatter(sizes, data_int64_speed, label='np.int64 with speedup',color='orange',marker='x')
plt.scatter(sizes, data_py_int_speed, label='python int with speedup',color='yellow',marker='x')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Array Size')
plt.ylabel('Execution Time (s)')
plt.title("Processing Times of Conditonal Cube Function for Variable Sized Integer Arrays")
plt.legend()
plt.show()

plt.clf()

#Plot For Floats
data_float32=[speedup_cube[np_types.index(np.float32)][j][0] for j in range(6)]
data_float64=[speedup_cube[np_types.index(np.float64)][j][0] for j in range(6)]
data_py_float=[speedup_cube[7][j][0] for j in range(6)]

data_float32_speed=[speedup_cube[np_types.index(np.float32)][j][1] for j in range(6)]
data_float64_speed=[speedup_cube[np_types.index(np.float64)][j][1] for j in range(6)]
data_py_float_speed=[speedup_cube[7][j][1] for j in range(6)]


plt.scatter(sizes, data_float32, label='np.float32',color='green',s=20)
plt.scatter(sizes, data_float64, label='np.float64',color='orange',s=20)
plt.scatter(sizes, data_py_float, label='python float',color='yellow',s=20)


plt.scatter(sizes, data_float32_speed, label='np.float32 with speedup',color='green',marker='x')
plt.scatter(sizes, data_float64_speed, label='np.float64 with speedup',color='orange',marker='x')
plt.scatter(sizes, data_py_float_speed, label='python float with speedup',color='yellow',marker='x')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Array Size')
plt.ylabel('Execution Time (s)')
plt.title("Processing Times of Conditonal Cube Function for Variable Sized Float Arrays")
plt.legend()
plt.show()


################# Parallelism ################## (Thomas)

#This function condtionally adds to an element
def add_one(x):
    if x %5==1 and x%3==0 and x+8!=8:
        x=x+1
    if x %4==1 and x%2==0 and x+8!=8:
        x=x+1
    if x %90==1 and x%3==0 and x+8!=8:
        x=x+1
    if x %91==1 and x%3==0 and x+8!=8:
        x=x+1
    for i in range(1000):
        if x==i:
            x=x-1
    return(x)


#Run without speedup
start=time.perf_counter_ns()
x=Parallel(n_jobs=1)(delayed(add_one)(i) for i in range(10000))
end=time.perf_counter_ns()
print(end-start)
        
 #Run with speedup
start=time.perf_counter_ns()
x=Parallel(n_jobs=4)(delayed(add_one)(i) for i in range(10000))
end=time.perf_counter_ns()
print(end-start)


#Create 3d array to store timing data
rows, cols = (8, 4)
speedup_cube = [[[0 for _ in range(2)] for _ in range(cols)] for _ in range(rows)]

#These are the types and sizes that we will loop through
np_types=[np.int8, np.int16, np.int32, np.int64, np.float32, np.float64] #(np.float16 not implemented)
py_types=[int,float]
sizes=[10**2, 10**3, 10**4, 10**5]

for type in np_types:
    for size in sizes:
        print(type)
        print(size)
        print()
        #Run without speedup
        start=time.perf_counter_ns()
        x=Parallel(n_jobs=1)(delayed(add_one)(i) for i in range(size))
        end=time.perf_counter_ns()
        speedup_cube[np_types.index(type)][sizes.index(size)][0]=(end-start)*10**-9
        
        #Run with speedup
        start=time.perf_counter_ns()
        x=Parallel(n_jobs=4)(delayed(add_one)(i) for i in range(size))
        end=time.perf_counter_ns()
        speedup_cube[np_types.index(type)][sizes.index(size)][1]=(end-start)*10**-9

for type in py_types:
    for size in sizes:
        
        #Run without speedup
        start=time.perf_counter_ns()
        x=Parallel(n_jobs=1)(delayed(add_one)(i) for i in range(size))
        end=time.perf_counter_ns()
        speedup_cube[py_types.index(type)+6][sizes.index(size)][0]=(end-start)*10**-9
        
        #Run with speedup
        start=time.perf_counter_ns()
        x=Parallel(n_jobs=4)(delayed(add_one)(i) for i in range(size))
        end=time.perf_counter_ns()
        speedup_cube[py_types.index(type)+6][sizes.index(size)][1]=(end-start)*10**-9


#Plot For Integers
data_int8=[speedup_cube[np_types.index(np.int8)][j][0] for j in range(4)]
data_int16=[speedup_cube[np_types.index(np.int16)][j][0] for j in range(4)]
data_int32=[speedup_cube[np_types.index(np.int32)][j][0] for j in range(4)]
data_int64=[speedup_cube[np_types.index(np.int64)][j][0] for j in range(4)]
data_py_int=[speedup_cube[6][j][0] for j in range(4)]

data_int8_speed=[speedup_cube[np_types.index(np.int8)][j][1] for j in range(4)]
data_int16_speed=[speedup_cube[np_types.index(np.int16)][j][1] for j in range(4)]
data_int32_speed=[speedup_cube[np_types.index(np.int32)][j][1] for j in range(4)]
data_int64_speed=[speedup_cube[np_types.index(np.int64)][j][1] for j in range(4)]
data_py_int_speed=[speedup_cube[6][j][1] for j in range(4)]

plt.scatter(sizes, data_int8, label='np.int8',color='red',s=20)
plt.scatter(sizes, data_int16, label='np.int16',color='blue',s=20)
plt.scatter(sizes, data_int32, label='np.int32',color='green',s=20)
plt.scatter(sizes, data_int64, label='np.int64',color='orange',s=20)
plt.scatter(sizes, data_py_int, label='python int',color='yellow',s=20)

plt.scatter(sizes, data_int8_speed, label='np.int8 with 4 jobs',color='red',marker='x')
plt.scatter(sizes, data_int16_speed, label='np.int16 with 4 jobs',color='blue',marker='x')
plt.scatter(sizes, data_int32_speed, label='np.int32 with 4 jobs',color='green',marker='x')
plt.scatter(sizes, data_int64_speed, label='np.int64 with 4 jobs',color='orange',marker='x')
plt.scatter(sizes, data_py_int_speed, label='python int with 4 jobs',color='yellow',marker='x')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Array Size')
plt.ylabel('Execution Time (s)')
plt.title("Processing Times of Parallel Jobs for Variable Sized Integer Arrays")
plt.legend()
plt.show()

plt.clf()

#Plot For Floats
data_float32=[speedup_cube[np_types.index(np.float32)][j][0] for j in range(4)]
data_float64=[speedup_cube[np_types.index(np.float64)][j][0] for j in range(4)]
data_py_float=[speedup_cube[7][j][0] for j in range(4)]

data_float32_speed=[speedup_cube[np_types.index(np.float32)][j][1] for j in range(4)]
data_float64_speed=[speedup_cube[np_types.index(np.float64)][j][1] for j in range(4)]
data_py_float_speed=[speedup_cube[7][j][1] for j in range(4)]


plt.scatter(sizes, data_float32, label='np.float32',color='green',s=20)
plt.scatter(sizes, data_float64, label='np.float64',color='orange',s=20)
plt.scatter(sizes, data_py_float, label='python float',color='yellow',s=20)


plt.scatter(sizes, data_float32_speed, label='np.float32 with 4 jobs',color='green',marker='x')
plt.scatter(sizes, data_float64_speed, label='np.float64 with 4 jobs',color='orange',marker='x')
plt.scatter(sizes, data_py_float_speed, label='python float with 4 jobs',color='yellow',marker='x')

plt.xscale('log')
plt.yscale('log')
plt.xlabel('Array Size')
plt.ylabel('Execution Time (s)')
plt.title("Processing Times of Parallel Jobs for Variable Sized Float Arrays")
plt.legend()
plt.show()


############ Error Checking ############# (Thomas)

#This function takes in an array, loops through it, and cubes all the odds values
def cube_odds_check(x):
    for i in range(len(x)):
        if x[i] % 2 ==1:

            try:
                x[i]=x[i]**3

            except OverflowError as e:
                print("Overflow")


#Create 3d array to store timing data
rows, cols = (8, 3)
speedup_cube = [[[0 for _ in range(2)] for _ in range(cols)] for _ in range(rows)]

#These are the types and sizes that we will loop through
np_types=[np.int8, np.int16, np.int32, np.int64, np.float32, np.float64] #(np.float16 not implemented)
py_types=[int,float]
sizes=[10**4]

for type in np_types:
    for size in sizes:

        #Run without speedup
        start=time.perf_counter_ns()
        a=cube_odds(np.arange(size, dtype=type))
        end=time.perf_counter_ns()
        no_speedup= end - start
        speedup_cube[np_types.index(type)][sizes.index(size)][0]=no_speedup*10**-9
        
        #Run with speedup
        start=time.perf_counter_ns()
        a=cube_odds_check(np.arange(size, dtype=type))
        end=time.perf_counter_ns()
        speedup_cube[np_types.index(type)][sizes.index(size)][1]= (end - start)*10**-9

for type in py_types:
    for size in sizes:
        
        #Run without speedup
        start=time.perf_counter_ns()
        a=cube_odds(np.arange(size, dtype=type))
        end=time.perf_counter_ns()
        speedup_cube[py_types.index(type)+6][sizes.index(size)][0]= (end-start)*10**-9

        #Run with speedup
        start=time.perf_counter_ns()
        a=cube_odds_check(np.arange(size, dtype=type))
        end=time.perf_counter_ns()
        speedup_cube[py_types.index(type)+6][sizes.index(size)][1]= (end-start)*10**-9


#Plot For Integers
data_int8=[speedup_cube[np_types.index(np.int8)][j][0] for j in range(1)]
data_int16=[speedup_cube[np_types.index(np.int16)][j][0] for j in range(1)]
data_int32=[speedup_cube[np_types.index(np.int32)][j][0] for j in range(1)]
data_int64=[speedup_cube[np_types.index(np.int64)][j][0] for j in range(1)]
data_py_int=[speedup_cube[6][j][0] for j in range(1)]

data_int8_speed=[speedup_cube[np_types.index(np.int8)][j][1] for j in range(1)]
data_int16_speed=[speedup_cube[np_types.index(np.int16)][j][1] for j in range(1)]
data_int32_speed=[speedup_cube[np_types.index(np.int32)][j][1] for j in range(1)]
data_int64_speed=[speedup_cube[np_types.index(np.int64)][j][1] for j in range(1)]
data_py_int_speed=[speedup_cube[6][j][1] for j in range(1)]

plt.scatter(sizes, data_int8, label='np.int8',color='red',s=20)
plt.scatter(sizes, data_int16, label='np.int16',color='blue',s=20)
plt.scatter(sizes, data_int32, label='np.int32',color='green',s=20)
plt.scatter(sizes, data_int64, label='np.int64',color='orange',s=20)
plt.scatter(sizes, data_py_int, label='python int',color='yellow',s=20)

plt.scatter(sizes, data_int8_speed, label='np.int8 with overflow check',color='red',marker='x')
plt.scatter(sizes, data_int16_speed, label='np.int16 with overflow check',color='blue',marker='x')
plt.scatter(sizes, data_int32_speed, label='np.int32 with overflow check',color='green',marker='x')
plt.scatter(sizes, data_int64_speed, label='np.int64 with overflow check',color='orange',marker='x')
plt.scatter(sizes, data_py_int_speed, label='python int with overflow check',color='yellow',marker='x')

plt.xlabel('Array Size')
plt.ylabel('Execution Time (s)')
plt.title("Processing Times of Conditonal Cube Function With and Without an Integer Overflow Check")
plt.legend()
plt.show()

plt.clf()

#Plot For Floats
data_float32=[speedup_cube[np_types.index(np.float32)][j][0] for j in range(1)]
data_float64=[speedup_cube[np_types.index(np.float64)][j][0] for j in range(1)]
data_py_float=[speedup_cube[7][j][0] for j in range(1)]

data_float32_speed=[speedup_cube[np_types.index(np.float32)][j][1] for j in range(1)]
data_float64_speed=[speedup_cube[np_types.index(np.float64)][j][1] for j in range(1)]
data_py_float_speed=[speedup_cube[7][j][1] for j in range(1)]


plt.scatter(sizes, data_float32, label='np.float32',color='green',s=20)
plt.scatter(sizes, data_float64, label='np.float64',color='orange',s=20)
plt.scatter(sizes, data_py_float, label='python float',color='yellow',s=20)


plt.scatter(sizes, data_float32_speed, label='np.float32 with overflow check',color='green',marker='x')
plt.scatter(sizes, data_float64_speed, label='np.float64 with overflow check',color='orange',marker='x')
plt.scatter(sizes, data_py_float_speed, label='python float with overflow check',color='yellow',marker='x')

plt.xlabel('Array Size')
plt.ylabel('Execution Time (s)')
plt.title("Processing Times of Conditonal Cube Function With and Without an Float Overflow Check")
plt.legend()
plt.show()


# These last few plots show the impacts of speedup tools, parallization techniques, and error checking on computational
# efficiency.  For the speedup plots, I used njit from the module numba to demonstarte speedup to an arbitrary function.
# Then, I showed speedup to a function by dividing element-by-element operatiosn into multiple jobs with joblib.  Lastly, I
# showed the impact of error checking on computation time, which does show a bit of slowdown.