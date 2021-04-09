import os

from artssat.scattering import ScatteringSpecies
from artssat.scattering.psd import D14MN, AB12

try:
    dendrite_path = os.environ["DENDRITE_PATH"]
except:
    home = os.environ["HOME"]
    dendrite_path = os.path.join(home, "Dendrite")

try:
    data_path = os.path.join(os.path.dirname(__file__), "..", "data")
except:
    data_path = os.path.join(".", "..", "data")

particle_names = ["6-BulletRosette",
                  "8-ColumnAggregate",
                  "ColumnType1",
                  "EvansSnowAggregate",
                  "Flat3-BulletRosette",
                  "GemGraupel",
                  "IconCloudIce",
                  "IconHail",
                  "IconSnow",
                  "LargeBlockAggregate",
                  "LargeColumnAggregate",
                  "LargePlateAggregate",
                  "LiquidSphere",
                  "Perpendicular3BulletRosette",
                  "PlateType1",
                  "SectorSnowflake"]

class D14(D14MN):
    """
    Specialized class implementing a normalized modified gamma distribution
    parametrized using mass density and mass weighted mean diameter (D_m).
    The shape is the same as the one used for the DARDAR v3 retrievals.
    """
    def __init__(self, alpha, beta, rho):
        super().__init__(alpha, beta, rho)
        self.name = "d14"

    @property
    def moment_names(self):
        return ["water_content", "n0"]

class Abel(AB12):
    """
    Specialized class implementing a normalized modified gamma distribution
    parametrized using mass density and mass weighted mean diameter (D_m).
    The shape is the same as the one used for the DARDAR v3 retrievals.
    """
    def __init__(self):
        super().__init__()
        self.name = "ab12"

    @property
    def moment_names(self):
        return ["water_content"]

class Ice(ScatteringSpecies):
    def __init__(self,
                 shape):
        # PSD, same as DARDAR V3
        alpha = -0.262
        beta = 1.754
        psd = D14(alpha, beta, 917.0)
        psd.t_max = 280.0
        # Look up particle name
        if shape in particle_names:
            name_data = shape + ".xml"
            name_meta = shape + ".meta.xml"
        else:
            raise ValueError("{} is not a known shape. Available shapes are {}".
                             format(shape, particle_names))

        scattering_data = os.path.join(data_path,
                                       "StandardHabits_small",
                                       name_data)
        scattering_meta_data = os.path.join(data_path,
                                            "StandardHabits_small",
                                            name_meta)
        super().__init__("ice", psd, scattering_data, scattering_meta_data)

class Rain(ScatteringSpecies):
    def __init__(self):
        # PSD, same as DARDAR V3
        alpha = 0.0
        beta = 1.0
        psd = Abel()
        psd.t_min = 270.0

        scattering_data = os.path.join(data_path,
                                       "StandardHabits_small",
                                       "LiquidSphere.xml")
        scattering_meta_data = os.path.join(data_path,
                                            "StandardHabits_small",
                                            "LiquidSphere.meta.xml")
        super().__init__("rain", psd, scattering_data, scattering_meta_data)
