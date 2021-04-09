import numpy as np
from artssat.sensor import PassiveSensor, ActiveSensor

class AWS(PassiveSensor):
    def __init__(self):
        self.sensor_pos = np.array
        f_grid = np.concatenate([band_89, band_166, band_183, band_229, band_325l, band_325u])
        f_grid *=1e9
        super().__init__("aws", f_grid, stokes_dimension=2)
        self.sensor_line_of_sight = np.arange(135, 181, 5).reshape(-1, 1)
        self.sensor_position = np.broadcast_to(np.array([[600e3]]),
                                               self.sensor_line_of_sight.shape)

class ATMS(PassiveSensor):

    def __init__(self):
        f_grid = np.array([166.5,])
        f_grid = np.concatenate([f_grid, 183.31 + np.array([-7,-4.5,-3.0,-1.8,-1.0])])
        f_grid *= 1e9
        super().__init__("atms", f_grid, stokes_dimension=2)
        self.sensor_line_of_sight = np.arange(135, 181, 5).reshape(-1, 1)
        self.sensor_position = np.broadcast_to(np.array([[830e3]]),
                                               self.sensor_line_of_sight.shape)

class CloudSat(ActiveSensor):
    def __init__(self):
        range_bins = np.arange(500.0, 18e3 + 1.0, 500.0)
        super().__init__("cloudsat",
                         f_grid = np.array([94.1e9]),
                         stokes_dimension = 1)
        self.instrument_pol       = [1]
        self.instrument_pol_array = [[1]]
        self.extinction_scaling   = 0.0
        self.y_min = -30.0
        self.sensor_position = np.array([[600e3]])
        self.sensor_line_of_sight = np.array([[180.0]])
        self.nedt = 1.0

class ICI(PassiveSensor):
    def __init__(self):
#        f_grid = np.array([1.749100000000000e+11,
#                         1.799100000000000e+11,
#                         1.813100000000000e+11,
#                         2.407000000000000e+11,
#                         3.156500000000000e+11,
#                         3.216500000000000e+11,
#                         3.236500000000000e+11,
#                         4.408000000000000e+11,
#                         4.450000000000000e+11,
#                         4.466000000000000e+11,
#                         6.598000000000000e+11])

        f_grid = np.concatenate([183.31 + np.array([-7.0, -3.4, -2.0, 2.0, 3.4, 7.0]),
                                243.20 + np.array([-2.5, 2.5]),
                                325.15 + np.array([-9.5, -3.5, -1.5, 1.5, 3.5, 9.5]),
                                448.00 + np.array([-7.2, -3.0, -1.4, 1.4, 3.0, 7.2]),
                                664.00 + np.array([-4.2, 4.2])])
        f_grid *= 1e9
        nedt = np.array([0.8, 0.8, 0.8,   # 183 GHz
                     0.7 * np.sqrt(0.5),  # 243 GHz
                     1.2, 1.3, 1.5,       # 325 GHz
                     1.4, 1.6, 2.0,       # 448 GHz
                     1.6 * np.sqrt(0.5)]) # 664 GHz
        
        super().__init__("ici", f_grid, stokes_dimension = 2)
        self.sensor_position = np.array([[832e3]])
        self.sensor_line_of_sight = np.array([[135.2]])


#-----24 frequencies---------
band_89 = np.array([89.0000])
band_166 = np.array([165.5000])
band_183 = np.array([175.3110, 176.7110, 178.1110, 179.5110, 180.9110, 182.3110, 182.5610])
band_229 = np.array([229.000])
band_325l = np.array([316.1500, 317.7500, 319.3500, 320.9500, 322.5500, 324.1500, 324.4000])
band_325u = np.array([325.9000, 326.1500, 327.7500, 329.3500, 330.9500, 332.5500, 334.1500])

#------18 frequencies-------
#band_89 = np.array([89.0000])
#band_166 = np.array([165.5000])
#band_183 = np.array([175.8110, 177.4360, 179.0610, 180.6860, 182.3110])
#band_229 = np.array([229.000])
#band_325l = np.array([316.6500, 318.5250, 320.4000, 322.2750, 324.1500])
#band_325u = np.array([326.1500, 328.0250, 329.9000, 331.7750, 333.6500])

#band_166  = 166.50e9 + np.array([-0.8, 0.8]) * 1e9
#band_183l = 183.31e9 + np.arange(-8.0, -0.8, 8.0) * 1e9;
#band_229  = 229.00e9 + np.array([-1.0, 1.0]) * 1e9;
#band_325l = 325.15e9 + np.arange(-9.0, -0.75, 8.0) * 1e9;
#band_325u = 325.15e9 + np.arange(0.75, 9.0, 8.0) * 1e989.0000
