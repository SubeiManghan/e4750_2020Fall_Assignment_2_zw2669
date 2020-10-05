# EECS E4750: Heterogeneous Computing for Signal and Data Processing (Fall 2020)

## Assignment-2: Deciphering Text and an Introduction to Profiling

Due date: 12th October, 2020 at *11:59 pm.*

Total points: 100

### To-do: GUI Installation and Introduction to CUDA Profiling

Before attempting this assignment, please follow the two tutorials posted to the Github wiki:
* [GUI Installation](https://github.com/eecse4750/e4750_2020Fall_students_repo/wiki/VM-GUI-Setup)
* [Intro to CUDA Profiling](https://github.com/eecse4750/e4750_2020Fall_students_repo/wiki/Introduction-to-Profiling)

Two demo videos for both topics have been posted to the Video Library on Courseworks.
You will need the GUI installation and Profiling know-how to complete Tasks 2.6 and 2.7, which are worth a total of 20% of this assignment. 

### Primer

The goal of this assignment is to introduce complexity to the 1D array manipulation kernel in OpenCL and CUDA. Additionally, you are expected to profile your PyCUDA code and form some conclusions from the visual profiling output you get. The assignment is divided into a programming section, and a theory section. The former contains two tasks - one for CUDA, the other for OpenCL. Information on CUDA Profiling will be introduced this week.

### Relevant Documentation

You may refer to the following links for this assignment to get started:

* [ROT-13 Cipher](https://en.wikipedia.org/wiki/ROT13)
* [ROT-13 converter](https://rot13.com/)


## Report (5 points)

Starting with Assignment-2, you are expected to prepare a markdown report to go with your code submission. A demo report markdown has been provided with this assignment, with detailed instructions on how to populate it, along with some useful markdown syntax if you are unfamiliar with its use. 

The GitHub wiki has also been updated with instructions for the homework report. You may check the sidebar to look for them.

## Programming Problem (75 points)

For this assignment, you have been given an encoded text file. Your task is to correctly decode it using appropriate CUDA and OpenCL kernels, along with a naive function using basic string manipulation in Python. While you will still be dealing with 1D arrays, the kernels you write will not be as primitive as they were in the first assignment. 

Your task is simplified several ways:
1. You will only deal with lower case characters of the English alphabet. Numbers, punctuation are left untouched by the cipher. 
2. The cipher scheme is known to you. 

### ROT-N Ciphers

Rotation ciphers are some of the most popular ciphers. In this assignment we will deal with the most well known of them all - ROT-13, popularly known as the Caesar Cipher. 

If you are intercepting a coded message, it is unlikely you immediately know the cipher used. The larger the possible number of ciphers, the more computational power you need. (See: Enigma). If you had to write a code to iteratively try out every possible ROT-N cipher to decode some text, parallel compuation would be a great boon to have. This assignment is set in a similar vein. 

### Problem set up

Consider the file `deciphertext.txt`. It contains text coded in ROT-13. All characters are lower case, and none of the numbers or punctuation are encoded. You must write code to read this file, and decipher it on a per-sentence basis. The PyOpenCL and PyCUDA kernels must only convert lowercase characters, and ignore all other ASCII values. 

#### Task-1: PyOpenCL (30 points)

Read the given text file and preprocess it according to the requirements in the steps below:

1. *(7 points)* Write the kernel code for per-character conversion.

2. *(8 points)* Write a function to decipher an input string using the kernel code. Time the execition of the entire operation (including memory transfers).

3. *(5 points)* Write a function to decipher an input string using native python string manipulation. Time the execution of the entire operation. 

4. *(5 points )* Call the kernel function and python function iteratively for every sentence in the input text. You can stitch the decrypted sentences back together into a unified output for each method. Finally, use Python's exception handling (`try` and `except`) to compare the two decryption results. If they are equal, only then write the decryption to a file in the assignment directory in `./results/decrypted_pocl.txt`

5. *(5 points)* Save a dot-plot the of per-sentence execution time for both methods. (There are 23 sentences in total). Use `nohup` to write the output to file in `./output_logs/4750HW2_rot13_pocl.out` You can do this by issuing the following command in the terminal:
    ```bash
    $ nohup python 4750HW2_rot13_pocl.py > ./output_logs/4750HW2_rot13_pocl.out
    ```

#### Task-2: PyCUDA & Profiling (40 points)

Read the given text file and preprocess it according to the requirements in the steps below:

1. *(5 points)* Write the kernel code for per-character conversion.

2. *(5 points)* Write a function to decipher an input string using the kernel code. Time the execition of the entire operation (including memory transfers).

3. Use the same naive python function to decipher sentence strings that you wrote for task-1. Time the execution of the entire operation. 

4. *(5 points )* Call the kernel function and python function iteratively for every sentence in the input text. You can stitch the decrypted sentences back together into a unified output for each method. Finally, use Python's exception handling (`try` and `except`) to compare the two decryption results. If they are equal, only then write the decryption to a file in the assignment directory in `./results/decrypted_pycu.txt`

5. *(5 points)* Save a dot-plot the of per-sentence execution time for both methods. (There are 23 sentences in total). Use `nohup` to write the output to file in `./output_logs/4750HW2_rot13_pycu.out` You can do this by issuing the following command in the terminal:
    ```bash
    $ nohup python 4750HW2_rot13_pycu.py > ./output_logs/4750HW2_rot13_pycu.out
    ```

6. *(10 points)* Use the Nvidia Visual Profiler (NVVP) to profile your CUDA kernel. You can do this by calling the first command below. The second command loads the profiling log to NVVP for viewing. 
    ```bash
    $ nvprof -o ./results/4750HW2_rot13_prof.nvprof python 4750HW2_rot13_pycu.py
    $ viewprofile ./results/4750HW2_rot13_prof.nvprof
    ```
    Take screenshots of what you see, and note your observations in the report.

7. *(10 points)* Based on the dot-plot, are all sentences deciphered in (roughly) equal time? If not, reason out why. Use the profiling output to explain the anomaly in execution time. 

### Deciphered Text (5 points)
The conditions for a full score are:
* The deciphered text matches the source text exactly, down to the spaces and punctuation. (i.e. total character count needs to be the same)
* All words in every sentence are correctly decoded. 

No points for identifying where the coded text is from; but if you happen to know - go ahead and write that in your report!

## Theory Problems (20 points)

1. *(5 points)* What is code profiling? How might profiling prove useful for CUDA development?

2. *(5 points)* If we need to use each thread to calculate one output element of a vector addition, what would be the expression for mapping the thread/block indices to data index: \
(A) i=threadIdx.x + threadIdx.y; \
(B) i=blockIdx.x + threadIdx.x; \
(C) i=blockIdx.x*blockDim.x + threadIdx.x; \
(D) i=blockIdx.x * threadIdx.x.

3. *(5 points)* We want to use each thread to calculate two (adjacent) output elements of a vector addition. Assume that variable i should be the index for the first element to be processed by a thread. What would be the expression for mapping the thread/block indices to data index of the first element? \
(A) i=blockIdx.xblockDim.x + threadIdx.x +2; \
(B) i=blockIdx.xthreadIdx.x2; \
(C) i=(blockIdx.xblockDim.x + threadIdx.x)2; \
(D) i=blockIdx.xblockDim.x*2 + threadIdx.x.

4. *(5 points)* For a vector addition, assume that the vector length is 8000, each thread calculates one output element, and the thread block size is 1024 threads. The programmer configures the kernel launch to have a minimal number of thread blocks to cover all output elements. How many threads will be in the grid? \
(A) 8000 \
(B) 8196 \
(C) 8192 \
(D) 8200

## Code Template

### PyOpenCL
```python
import relevant.libraries


class clCipher:
    def __init__(self, plain_text):
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
        self.queue = cl.CommandQueue(self.ctx, properties=cl.command_queue_properties.PROFILING_ENABLE)

        # host variables
        
        # kernel
        
        # Build kernel code
        

    def devCipher(self):
        """
        Function to perform on-device parallel ROT-13 encrypt/decrypt
        by explicitly allocating device memory for host variables using
        gpuarray.
        Returns
            decrypted :   decrypted/encrypted result
            time_     :   execution time in milliseconds
        """

        # Text pre-processing/list comprehension (if required)
        # Depends on how you approach the problem
        
        # device memory allocation
        
        # Call the kernel function and time event execution
        
        # OpenCL event profiling returns times in nanoseconds. 
        # Hence, 1e-6 will provide the time in milliseconds, 
        # making your plots easier to read.

        # Copy result to host memory
        
        return decrypted, time_

    
    def pyCipher(self):
        """
        Function to perform parallel ROT-13 encrypt/decrypt using 
        vanilla python. (String manipulation and list comprehension
        will prove useful.)

        Returns
            decrypted                  :   decrypted/encrypted result
            (end_ - start_)**(1e-3)    :   execution time in milliseconds
        """
        
        return decrypted, (end_ - start_)**(1e-3)


if __name__ == "__main__":
    # Main code

    # Open text file to be deciphered.
    # Preprocess the file to separate sentences


    # Loop over each sentence in the list
    for _ in _______:
        
    # Stitch decrypted sentences together
    
    print("OpenCL output cracked in ", tc, " milliseconds per sentence.")
    print("Python output cracked in ", tp, " milliseconds per sentence.")

    # Error check
    try:
        print("Checkpoint: Do python and kernel decryption match? Checking...")
        # compare outputs
    except ______:
        print("Checkpoint failed: Python and OpenCL kernel decryption do not match. Try Again!")
        # dump bad output to file for debugging
        

    # If ciphers agree, proceed to write decrypted text to file
    # and plot execution times

    if #conditions met: 

        # Write cuda output to file
        
        # Scatter plot the  per-sentence execution times    
```

### PyCUDA

```python
import relevant.libraries


class cudaCipher:
    def __init__(self, plain_text):
        """
        Attributes for instance of deviceAdd module
        Includes kernel code and input variables.
        """
        # If you are using any helper function to make 
        # blocksize or gridsize calculations, you may define them
        # here as lambda functions. 
        # Quick lambda function to calculate grid dimensions

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

    
    def devCipher(self):
        """
        Function to perform on-device parallel ROT-13 encrypt/decrypt
        by explicitly allocating device memory for host variables using
        gpuarray.
        Returns
            out                             :   encrypted/decrypted result
            e_start.time_till(e_end)*(1e-3) :   execution time in milliseconds
        """
        
        # Note: Use cuda.Event to time your executions
        # Event objects to mark the start and end points
        
        # Get kernel function

        # Device memory allocation for input and output array(s)
        
        
        # Record execution time and execute operation with numpy syntax
       

        # Wait for the event to complete

        # Fetch result from device to host

        # Convert output array back to string

        return decrypted, e_start.time_till(e_end)

    
    def pyCipher(self):
        """
        Function to perform parallel ROT-13 encrypt/decrypt using 
        vanilla python.

        Returns
            decrypted                       :   encrypted/decrypted result
            (end_ - start_)**(1e-3)         :   execution time in milliseconds
        """

        return decrypted, (end_ - start_)**(1e-3)


if __name__ == "__main__":
    # Main code

    # Open text file to be deciphered.
    # Preprocess the file to separate sentences
    
    # Split string into list populated with '.' as delimiter.

    # Empty lists to hold deciphered sentences, execution times


    # Loop over each sentence in the list
    for _ in _______:
        
    # post process the string(s) if required
        
    # Execution time
    print("CUDA output cracked in ", tc, " milliseconds per sentence.")
    print("Python output cracked in ", tp, " milliseconds per sentence.")

    # Error check
    try:
        print("Checkpoint: Do python and kernel decryption match? Checking...")
        assert # something
        
    except _________:
        print("Checkpoint failed: Python and CUDA kernel decryption do not match. Try Again!")
        
        # dump bad output to file for debugging
        

    # If ciphers agree, proceed to write decrypted text to file
    # and plot execution times
    
    if #conditions met:
        print("Checkpoint passed!")
        print("Writing decrypted text to file...")

        # Write cuda output to file
        
        # Dot plot the  per-sentence execution times
```
