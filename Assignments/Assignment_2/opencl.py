import numpy as np
import pyopencl as cl
import pyopencl.array as array
import time
import sys

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
        self.plain_text = plain_text
        self.d = dict()
        # ord("a") = 97
        for i in range(26):
            self.d[chr(i + 97)] = chr((i + 13) % 26 + 97)
        
        # kernel
        kernel_code = """__kernel void rot13(__global char* dest, __global char* src, const unsigned int n)
                        {
                            unsigned int i = get_global_id(0);
                            if (i < n) {
                                // ord('a') = 97 ord('z') = 122
                                if(src[i] < 97 || src[i] > 122) dest[i] = src[i];
                                else dest[i] = (src[i] - 97 + 13) % 26 + 97;
                            }
                        }""" 
        
        # Build kernel code
        self.prg = cl.Program(self.ctx, kernel_code).build()
        

    def devCipher(self):
        """
        Function to perform on-device parallel ROT-13 encrypt/decrypt
        by explicitly allocating device memory for host variables.
        Returns
            decrypted :   decrypted/encrypted result
            time_     :   execution time in milliseconds
        """

        # Text pre-processing/list comprehension (if required)
        # Depends on how you approach the problem
        src = np.frombuffer(self.plain_text.encode("utf8"), np.int8)
        # device memory allocation
        mf = cl.mem_flags
        src_g = cl.Buffer(self.ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=src)
        dest_g = cl.Buffer(self.ctx, mf.WRITE_ONLY, src.nbytes)
        
        # Call the kernel function and time event execution
        event = self.prg.rot13(self.queue, src.shape, None, dest_g, src_g, np.uint32(src.size))
        event.wait()
        time_ = (event.profile.end-event.profile.start) * 1e-6
        
        # OpenCL event profiling returns times in nanoseconds. 
        # Hence, 1e-6 will provide the time in milliseconds, 
        # making your plots easier to read.

        # Copy result to host memory
        dest = np.empty_like(src)
        cl.enqueue_copy(self.queue, dest, dest_g)
        decrypted = dest.tobytes()
        return decrypted.decode("utf8"), time_

    
    def pyCipher(self):
        """
        Function to perform parallel ROT-13 encrypt/decrypt using 
        vanilla python. (String manipulation and list comprehension
        will prove useful.)
        Returns
            decrypted                  :   decrypted/encrypted result
            end_ - start_             :   execution time in milliseconds
        """
        start_ = time.time()
        decrpted = "".join([self.d.get(c, c) for c in self.plain_text])
        end_ = time.time()
        return decrypted, (end_ - start_) * 1e3


if __name__ == "__main__":
    # Main code

    # Open text file to be deciphered.
    # Preprocess the file to separate sentences
    with open(sys.argv[1]) as fp:
      txt = fp.read()
    # Split string into list populated with '.' as delimiter.
    src = txt.split(".")
    # Empty lists to hold deciphered sentences, execution times
    dest_dev = []  
    dest_py = []  
    t_ds = []
    t_pys = []
    # Loop over each sentence in the list
    for l in src:
        c = clCipher(l)
        res_d, t_d = c.devCipher()
        res_py, t_py = c.devCipher()
        dest_dev.append(res_d) 
        dest_py.append(res_py)
        t_ds.append(t_d)
        t_pys.append(t_py)
    # post process the string(s) if required
    tc = sum(t_ds) / len(t_ds)
    tp = sum(t_pys) / len(t_pys)
    print("OpenCL output cracked in ", tc, " milliseconds per sentence.")
    print("Python output cracked in ", tp, " milliseconds per sentence.")

    # Error check
    try:
        print("Checkpoint: Do python and kernel decryption match? Checking...")
        assert "".join(dest_dev) == "".join(dest_py)# something
        # compare outputs
    except:
        print("Checkpoint failed: Python and OpenCL kernel decryption do not match. Try Again!")
        # dump bad output to file for debugging
        

    # If ciphers agree, proceed to write decrypted text to file
    # and plot execution times

    #if #conditions met: 

        # Write opencl output to file
        
        # Dot plot the  per-sentence execution times

