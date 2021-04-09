import os
import numpy as np
import re
import glob
from h5py import File
from artssat.data_provider import DataProviderBase

class Profiles(DataProviderBase):
    """
    Data provider class that provides the artssat interface for an
    input data file in HDF5 (matlab) format.

    Attributes:
        self.path: The filename of the input data
        self.data: HDF5 group object containing the data

    """
    def __init__(self, path):
        """
        Create data provider from given input file.

        Args:
            path(:code:`str`): The path to the input file.
        """
        super().__init__()
        path = os.path.expanduser(path)
        if not os.path.exists(path):
            raise ValueError("Could not find file {}.".format(path))
        self.path = path
        self.file = File(path, mode="r")

    def __del__(self):
        self.file.close()

    @property
    def n_profiles(self):
        return self.file["C"]["dBZ"].shape[0]

    def get_y_cloudsat(self, i, **kwargs):
        r = self.file["C"]["dBZ"][i][0]
        data = self.file[r][0, :]
        data = np.maximum(data, -30.0)
        return data

    def get_cloudsat_range_bins(self, i, **kwargs):
        r = self.file["C"]["z_field"][i][0]
        z = self.file[r][0, :]
        range_bins = np.zeros(z.size + 1)
        range_bins[1:-1] = 0.5 * (z[1:] + z[:-1])
        range_bins[0] = 2 * range_bins[1] - range_bins[2]
        range_bins[-1] = 2 * range_bins[-2] - range_bins[-3]
        return range_bins

    def get_pressure(self, i, **kwargs):
        r = self.file["C"]["p_grid"][i][0]
        data = self.file[r][0, :]
        return data

    def get_temperature(self, i, **kwargs):
        r = self.file["C"]["t_field"][i][0]
        data = self.file[r][0, :]
        return data

    def get_temperature_field(self):
        t = []
        for i in range(self.n_profiles):
            t += [self.get_temperature(i)]
        return np.stack(t)

    def get_surface_temperature(self, i, **kwargs):
        r = self.file["C"]["t_surface"][i][0]
        data = self.file[r][0, 0]
        return data

    def get_altitude(self, i, **kwargs):
        r = self.file["C"]["z_field"][i][0]
        data = self.file[r][0, :]
        return data

    def get_altitude_field(self):
        z = []
        for i in range(self.n_profiles):
            z += [self.get_altitude(i)]
        return np.stack(z)

    def get_H2O(self, i, **kwargs):
        r = self.file["C"]["h2o"][i][0]
        data = self.file[r][0, :]
        return data

    def get_N2(self, i, **kwargs):
        z = self.get_altitude(i)
        return 0.781 * np.ones(z.shape)

    def get_O2(self, i, **kwargs):
        z = self.get_altitude(i)
        return 0.209 * np.ones(z.shape)

    def get_cloud_water(self, i, **kwargs):
        r = self.file["C"]["lwc"][i][0]
        data = self.file[r][0, :]
        return data

    def get_surface_altitude(self, i, **kwargs):
        r = self.file["C"]["z_field"][i][0]
        data = self.file[r][0, 0]
        return data

    def get_surface_wind_speed(self, i, **kwargs):
        r = self.file["C"]["wind_speed"][i][0]
        data = self.file[r][0, :]
        return data

    def get_surface_wind_direction(self, i, **kwargs):
        r = self.file["C"]["wind_dir"][i][0]
        data = self.file[r][0, :]
        return data

    def get_surface_type(self, i, **kwargs):
        r = self.file["C"]["i_surface"][i][0]
        data = np.round(self.file[r][0, :])
        return data

    def get_latitude(self, i, **kwargs):
        r = self.file["C"]["lat"][i][0]
        data = self.file[r][0, :]
        return data

    def get_latitudes(self, **kwargs):
        lat = []
        for i in range(self.n_profiles):
            lat += [self.get_latitude(i)]
        return np.stack(lat)

    def get_longitude(self, i, **kwargs):
        r = self.file["C"]["lon"][i][0]
        data = self.file[r][0, :]
        return data

    def get_ice_water_content_field(self):
        iwc = []
        for i in range(self.n_profiles):
            r = self.file["C"]["iwc"][i][0]
            iwc += [self.file[r][0, :]]
        return np.stack(iwc)

    def get_rain_water_content_field(self):
        iwc = []
        for i in range(self.n_profiles):
            r = self.file["C"]["rwc"][i][0]
            iwc += [self.file[r][0, :]]
        return np.stack(iwc)

pattern = re.compile(".*c_of_([\d]*)_([\d]*)_([\d]*).mat")

class RandomProfile(DataProviderBase):
    """
    Data provider returning a random profile from a folder of orbit files.

    Attributes:
        self.path: The folder containing the input orbits.
    """
    def __init__(self, path):
        """
        """
        path = os.path.expanduser(path)
        super().__init__()
        files = glob.glob(os.path.join(path, "*.mat"))
        self.path = path
        self.files = [f for f in files if pattern.match(f)]

        self.cycle = 1000000
        np.random.seed(666)
        self.file_indices = np.random.randint(0, len(files), size=self.cycle)
        self.profile_indices = np.random.randint(0, len(files), size=self.cycle)

        def make_getter(name):
            def getter(self, i, **kwargs):
                profiles = self._get_random_file(i)
                ri = self._get_random_index(profiles, i)
                fget = getattr(profiles, "get_" + name)
                return fget(ri, **kwargs)
            return getter

        getters = ["y_cloudsat",
                   "cloudsat_range_bins",
                   "pressure",
                   "temperature",
                   "altitude",
                   "surface_temperature",
                   "surface_type",
                   "surface_wind_speed",
                   "surface_wind_direction",
                   "H2O",
                   "N2",
                   "O2",
                   "cloud_water",
                   "y_cloudsat",
                   "cloudsat_range_bins",
                   "surface_altitude",
                   "latitude",
                   "longitude"]

        for g in getters:
            self.__dict__["get_" + g] = make_getter(g).__get__(self)

    def _get_random_file(self, i):
        ind = i % self.cycle
        ri = self.file_indices[ind]
        return Profiles(self.files[ri])

    def _get_random_index(self, profiles, i):
        ind = i % self.cycle
        ri = self.profile_indices[i] % profiles.n_profiles
        return ri

    def get_filename(self, i, **kwargs):
        ind = i % self.cycle
        ri = self.file_indices[i]
        return os.path.basename(self.files[ri])

    def get_profile_index(self, i, **kwargs):
        profiles = self._get_random_file(i)
        ri = self._get_random_index(profiles, i)
        return ri
