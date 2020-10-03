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
        
