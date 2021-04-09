import numpy as np
import netCDF4

class awsTestData():
    """
    CLass for the AWS test data.

    """
    def __init__(self, path, inChannels, option):
        """
        Create instance of the dataset from a given file path.

        Args:
            path: Path to the NetCDF4 containing the data.
            batch_size: If positive, data is provided in batches of this size

        """


        path0 = path.replace("_noise", "")

        self.file = netCDF4.Dataset(path, mode = "r")
        self.file0 = netCDF4.Dataset(path0, mode = "r")
        
        TB = self.file.variables["TB_noise"][:]
        TB0 = self.file0.variables["TB"][:]
        channels = self.file.variables["channels"][:]

 #       print (channels)
        self.channels = inChannels
        self.option = option
#       find index for input channels
        cases = TB.shape[2]

        self.index = []
        for c in self.channels:
            self.index.append(np.argwhere(channels == c)[0,0])
       
        print (self.index)
        C = []
        
        for i in range(len(self.channels)):
            C.append(TB[self.index[i], 1, :])

        x = np.float32(np.stack(C, axis = 1))
        
        #store mean and std to normalise data  
#        x_noise = self.add_noise(self.x)
        self.std = np.std(x, axis = 0)
        self.mean = np.mean(x, axis = 0)    
        
        self.y = np.float32(TB[self.index[0], 0, :])
        
        self.y0 = np.float32(TB0[self.index[0], 0, :])
        
        self.x  = x.data
        self.y  = self.y.data
        self.y0 = self.y0.data
        
        
 
    def normalise(self, x):
        """
        normalise the input data wit mean and standard deviation
        Args:
            x
        Returns :
            x_norm
        """            
        x_norm = (x - self.mean)/self.std   
            
        return x_norm 


