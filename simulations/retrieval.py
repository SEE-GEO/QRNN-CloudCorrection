import numpy as np

from aws.simulation import Simulation
from aws.sensor import CloudSat
from artssat.jacobian import Log10
from artssat.atmosphere.surface import Tessem
from artssat.data_provider import DataProviderBase
from artssat.atmosphere.absorption import O2

class RainAPriori(DataProviderBase):
    def __init__(self):
        super().__init__()

    def get_rain_water_content_xa(self, *args, **kwargs):
        t = self.owner.get_temperature(*args, **kwargs)
        xa = np.zeros(t.shape)
        xa[:] = 1e-5
        xa[t < 273.15] = 1e-12
        return np.log10(xa)

    def get_rain_water_content_covariance(self, *args, **kwargs):
        t = self.owner.get_temperature(*args, **kwargs)
        diag = 2.0 * np.ones(t.size)
        diag[t < 273.15] = 1e-24
        covmat = np.diag(diag)
        return covmat

    def get_rain_n0(self, *args, **kwargs):
        t = self.owner.get_temperature(*args, **kwargs)
        return 1e7 * np.ones(t.shape)

class IceAPriori(DataProviderBase):
    def __init__(self):
        super().__init__()

    def get_ice_water_content_xa(self, *args, **kwargs):
        t = self.owner.get_temperature(*args, **kwargs)
        xa = np.zeros(t.shape)
        xa[:] = 1e-5
        xa[t >= 273.15] = 1e-12
        return np.log10(xa)

    def get_ice_water_content_covariance(self, *args, **kwargs):
        t = self.owner.get_temperature(*args, **kwargs)
        diag = 2.0 * np.ones(t.size)
        diag[t >= 273.15] = 1e-12
        covmat = np.diag(diag)
        return covmat

    def get_ice_n0(self, *args, **kwargs):
        t = self.owner.get_temperature(*args, **kwargs)
        t = t - 273.15
        return np.exp(-0.076586 * t + 17.948)

class ObservationErrors(DataProviderBase):
    def __init__(self, nedt=1.0):
        super().__init__()
        self.nedt = 1.0

    def get_observation_error_covariance(self, *args, **kwargs):
        range_bins = self.owner.get_cloudsat_range_bins(*args, **kwargs)
        return np.diag(self.nedt * np.ones(range_bins.size - 1))


class Retrieval(Simulation, DataProviderBase):
    def __init__(self,
                 data_provider,
                 ice_shape = "8-ColumnAggregate"):
        sensor = CloudSat()

        DataProviderBase.__init__(self)
        self.add(IceAPriori())
        self.add(RainAPriori())
        self.add(ObservationErrors())
        self.add(data_provider)

        Simulation.__init__(self,
                            sensor,
                            self,
                            ice_shape=ice_shape)

        self.atmosphere.absorbers += [O2(model="PWR98")]

        scatterers = self.atmosphere.scatterers
        self.atmosphere._surface = Tessem()
        # Add first moment ice to retrieval and set transform
        ice = [s for s in scatterers if s.name == "ice"][0]
        self.retrieval.add(ice.moments[0])
        ice.moments[0].transformation = Log10()
        self._ice_water_content = ice.moments[0]

        # Add first moment of rain PSD to retrieval and set transform
        rain = [s for s in scatterers if s.name == "rain"][0]
        self.retrieval.add(rain.moments[0])
        rain.moments[0].transformation = Log10()
        self._rain_water_content = rain.moments[0]

        self.retrieval.settings["stop_dx"] = 1e-6
        self.cache_index = None

        self.iwc = None
        self.rwc = None

    def run(self, i, silent=False):
        if not self._setup:
            self.setup()

        self.retrieval.settings["display_progress"] = int(not silent)
        Simulation.run(self, i)
        self.cache_index = i
        self.iwc = self.retrieval.results.get_result(self._ice_water_content,
                                                     transform_back=True)
        self.rwc = self.retrieval.results.get_result(self._rain_water_content,
                                                     transform_back=True)

    def get_ice_water_content(self, i, clearsky=False):
        if clearsky:
            z = self.data_provider.get_altitude(i)
            return np.zeros(z.shape)
        if not self.cache_index == i:
            self.run(i, silent=True)
        return self.iwc

    def get_rain_water_content(self, i, clearsky=False):
        if clearsky:
            z = self.data_provider.get_altitude(i)
            return np.zeros(z.shape)
        if not self.cache_index == i:
            self.run(i, silent=True)
        return self.rwc
