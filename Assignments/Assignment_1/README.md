# EECS E4750: Heterogeneous Computing for Signal and Data Processing (Fall 2020)

## Assignment-1: Introduction to Memory Access in PyCUDA & PyOpenCL

Due date: ~~25th September, 2020~~ 1st October, 2020 11:59 PM

Total points: 100

### Primer

The goal of the assignment is to discover the most efficient method(s) of host-to-device memory allocation. The assignment is divided into a programming section, and a theory section. The programming section contains two tasks - one for CUDA, the other for OpenCL. 

### Relevant Documentation

For PyOpenCL:
1. [OpenCL Runtime: Platforms, Devices & Contexts](https://documen.tician.de/pyopencl/runtime_platform.html)
2. [pyopencl.array](https://documen.tician.de/pyopencl/array.html#the-array-class)
3. [pyopencl.Buffer](https://documen.tician.de/pyopencl/runtime_memory.html#buffer)

For PyCUDA:
1. [Documentation Root](https://documen.tician.de/pycuda/index.html)
2. [Memory tools](https://documen.tician.de/pycuda/util.html#memory-pools)
3. [gpuarrays](https://documen.tician.de/pycuda/array.html)


## Programming Problem (80 points)

### Problem set up
Consider two 1D vectors *A* and *B* of length **N**. The task is to write code for PyOpenCL and for PyCUDA which performs vector addition on those two vectors in multiple ways, which are differentiated in how they interact with device memory. The programming problem is divided into two tasks, one each for OpenCL and CUDA. Make sure you complete both by following the instructions exactly. 

#### Task-1: PyOpenCL (30 points)

For PyOpenCL, you have been provided with the kernel code along with the assignment template. Your task is to build this kernel and use it as the basis for two methods of vector addition. The methods involve different ways of interacting with device and host memory. Read the instructions below and follow them exactly to complete task-1 of this assignment:

1. The kernel code for OpenCL is provided in the template at the end of this README file. 

2. *(10 points)* Write a function to perform vector addition on *A* and *B* by using `pyopencl.array` to load your vectors to device memory. Time the execution of the operation only, excluding the memory transfer steps. 

3. *(10 points)* Write a function to perform vector addition on *A* and *B* by using `pyopencl.Buffer` to load your vectors to device memory. Time the execution of the operation, excluding the memory transfer steps.  

4. *(5 points)* Iteratively increase the length of the array by a factor, say **L**, *(L = 1,2,3,...,20)*. Start with an array size of 100,000 or more. Call the functions which you wrote one-by-one to perform vector addition on the two arrays and record the **average** running time for each function call.

5. *(5 points)* Plot the **average** execution times against the increasing array size (in orders of **L**)


#### Task-2: PyCUDA (50 points)

For PyCUDA, the coding problem will involve your first practical encounter with kernel codes, host-to-device memory transfers (and vice-versa), and certain key classes that PyCUDA provides for them. Read the instructions below carefully and complete your assignment as outlined:

1. *(10 points)* Write the kernel code for vector addition. Keep in mind that you will be working with arrays much larger than the maximum blocksize accepted by modern Nvidia dGPUs (usually 1024). Start with an array size of 100,000 or more.

2. *(8 points)* Write a function to perform vector addition on *A* and *B* taking advantage of explicit memory allocation using `pycuda.driver.mem_alloc()`. Do not forget to retrieve the result from device memory using the appropriate PyCUDA function. Use `SourceModule` to compile the kernel which you defined earlier. Time the execution.

3. *(8 points)* Write a function to perform vector addition on *A* and *B* **without** explicit memory allocation. Use `SourceModule` to compile the kernel which you defined earlier. Time the execution.

4. *(5 points)* Write a function to perform vector addition on *A* and *B* using the `gpuarray` class instead of allocating with `mem_alloc`. Use standard algebraic syntax to perform addition. Time the execution. 

5. *(7 points)* Write a function to perform vector addition on *A* and *B* using the `gpuarray` class instead of allocating with `mem_alloc`. Use `SourceModule` to compile the kernel which you defined earlier. Time the execution.

6. *(7 points)* Iteratively increase the length of the array by a factor, say **L**, *(L = 1,2,3,...,20)*. Call the functions you previously wrote one-by-one to perform vector addition on the two arrays, and record the **average** running time for each function call. 

7. *(5 points)* Plot the **average** execution times against the increasing array size (in orders of **L**)

## Theory Problems (20 points)

1. *(3 points)* What is the difference between a thread, a task and a process?

2. *(3 points)* What are the differences between concurrency and parallelism?

3. *(7 points)* Out of the two approaches explored in task-1 (PyOpenCL), which proved to be faster? Explore the PyOpenCL docs and source code to support your conclusions about the differences in execution time.

4. *(7 points)* Of the different approaches explored in task-2 (PyCUDA), which method(s) proved the fastest? Explore the PyCUDA docs and source code and explain how/why: (a) Normal python syntax can be used to perform operations on gpuarrays; (b) gpuarray execution (non-naive method) is comparable to using `mem_alloc`.


## Submission Instructions

The assignment submission must strictly follow the following directory tree:

```
.
├── 4750HW1_mem_pocl.py
├── 4750HW1_mem_pycu.py
├── output_logs
│   ├── 4750HW1_pocl.out
│   └── 4750HW1_pycu.out
├── plots
│   ├── HW1_time_cuda.png
│   └── HW1_time_pocl.png
└── README.md

2 directories, 7 files
```
## Code Templates

You **must** adhere to the template given in the starter code below - this is essential for all assignments to be graded fairly and equally. 

#### PyOpenCL Starter Code *(with kernel code)*

**Note:** The kernel code is only provided for OpenCL, and for the sole reason that this is the first assignment of the course.  

```python
import relevant.libraries

class clModule:
    def __init__(self, a, b, length):
        """
        Attributes for instance of clModule
        Includes OpenCL context, command queue, kernel code
        and input variables.
        """
        
        # Get platform and device property
        NAME = 'NVIDIA CUDA'
        platforms = cl.get_platforms()
        devs = None
        for platform in platforms:
        	if platform.name == NAME:
        		devs = platform.get_devices()       
        
        # Set up a command queue:
        self.ctx = cl.Context(devs)
        
        # Enable profiling property to time event execution
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

        # host variables
        # 
        #

        # kernel - will not be provided for future assignments!
        kernel_code = """__kernel void sum(__global float* c, __global float* a, __global float* b, const unsigned int n)
                        {
                            unsigned int i = get_global_id(0);
                            if (i < n) {
                                c[i] = a[i] + b[i];
                                //printf("c[i] %f" , c[i]);
                                //printf(" ");
                                //Use printf to help debug kernel code
                            }
                        }""" 
        
        # Build kernel code
        self.prg = cl.Program(self.ctx, kernel_code).build()


    def deviceAdd(self):
        """
        Function to perform vector addition using the cl.array class    
        Returns:
            c       :   vector sum of arguments a and b
            time_   :   execution time for pocl function 
        """
                
        # Device memory allocation
        #
        #
        
        # Invoke kernel program and time execution using event.profile()
        # Wait for program execution to complete
        # Remember: OpenCL event profiling returns times in nanoseconds. 
        
        # Fetch result from device to host

        return c, time_

    
    def bufferAdd(self):
        """
        Function to perform vector addition using the cl.Buffer class
        Returns:
            c               :    vector sum of arguments a and b
            end - start     :    execution time for pocl function 
        """
        # Create three buffers (plans for areas of memory on the device)
        # Invoke kernel program, time execution
        # Fetch result
        
        return c, time_


if __name__ == "__main__":
    # Main code to create arrays, call all functions, calculate average
    # execution and plot
```

#### PyCUDA Starter Code

```python
import relevant.libraries

class deviceAdd:
    def __init__(self, a, b, length):
        """
        Attributes for instance of deviceAdd module
        Includes kernel code and input variables.
        """
        # If you are using any helper function to make 
        # blocksize or gridsize calculations, you may define them
        # here as lambda functions. 

        # host variables
        #

        # define block and grid dimensions
        #

        # kernel code wrapper
        #

        # Compile the kernel code when an instance
        # of this class is made. This way it only
        # needs to be done once for the 4 functions
        # you will call from this class.
        
    
    def explicitAdd(self):
        """
        Function to perform on-device parallel vector addition
        by explicitly allocating device memory for host variables.
        Returns
            c                               :   addition result
            e_start.time_till(e_end)*(1e-3) :   execution time
        """

        # Note: Use cuda.Event to time your executions

        # Device memory allocation for input and output arrays
        #
        
        # Copy data from host to device
        #

        # Call the kernel function from the compiled module
        #
        # Record execution time and call the kernel loaded to the device
        
        # Wait for the event to complete
        #
        # Copy result from device to the host
        #

        return c, e_start.time_till(e_end)*(1e-3)

    
    def implicitAdd(self):
        """
        Function to perform on-device parallel vector addition
        without explicitly allocating device memory for 
        host variables.
        Returns
            c                               :   addition result
            i_start.time_till(i_end)*(1e-3) :   execution time
        """
        # Call the kernel function from the compiled module
        #
        
        # Record execution time and call the kernel loaded to the device
        #
        
        # Wait for the event to complete
        #
        
        return c, i_start.time_till(i_end)*(1e-3)


    def gpuarrayAdd_np(self):
        """
        Function to perform on-device parallel vector addition
        without explicitly allocating device memory for 
        host variables and WITHOUT calling the kernel. The operation
        is defined using numpy-like syntax. 
        Returns
            c                               :   addition result
            i_start.time_till(i_end)*(1e-3) :   execution time
        """
        # Allocate device memory using gpuarray class        
        #

        # Record execution time and execute operation with numpy syntax
        #

        # Wait for the event to complete
        #
        
        # Fetch result from device to host
        
        return c, g_start.time_till(g_end)*(1e-3)
        
    
    def gpuarrayAdd(self):
        """
        Function to perform on-device parallel vector addition
        using the gpuarray class for memory allocation, along 
        with SourceModule to invoke the kernel.
        Returns
            c                               :   addition result
            i_start.time_till(i_end)*(1e-3) :   execution time
        """
        # Allocate device memory using gpuarray class        
        #

        # Record execution time and execute operation with numpy syntax
        #
        # Wait for the event to complete
        #
    
        # Fetch result from device to host
        #
    
        return c, g_start.time_till(g_end)*(1e-3)

if __name__ == "__main__":
    # Main code to create arrays, call all functions, calculate average
    # execution and plot
```
