import os

from artssat import ArtsSimulation
from artssat.atmosphere import Atmosphere1D
from artssat.atmosphere.absorption import H2O, CloudWater, O2, N2
from artssat.atmosphere.surface import CombinedSurface, Tessem, Telsem
from artssat.scattering.solvers import RT4
from artssat.atmosphere.catalogs import LineCatalog
from pyarts.workspace.api import include_path_push, arts_include_path

from aws.hydrometeors import Ice, Rain
from aws.sensor import AWS
from aws import aws_path


include_path_push(os.path.join(aws_path, "aws", "include"))
arts_include_path += [os.path.join(aws_path, "aws", "include")]


class Simulation(ArtsSimulation):
    def __init__(self,
                 sensor,
                 data_provider,
                 ice_shape = "8Column-Aggregate"):
        scatterers = [Ice(ice_shape), Rain()]
        absorbers = [H2O(from_catalog=True,
                         model="MPM89"),
                     CloudWater(from_catalog=True,
                                model="ELL07"),
                     N2()]
        telsem = Telsem(os.path.join(aws_path, "data"))
        telsem.d_max = 100e3
        surface = CombinedSurface(Tessem(), telsem)
        scattering_solver = RT4(nstreams=16)
        atmosphere = Atmosphere1D(absorbers=absorbers,
                                  scatterers=scatterers,
                                  surface=surface,
#                                  catalog=LineCatalog("abs_lines_h2o_rttov_below340ghz.xml"))
                                  catalog=LineCatalog("abs_lines_h2o_rttov.xml"))


        super().__init__(atmosphere=atmosphere,
                         data_provider=data_provider,
                         sensors=[sensor],
                         scattering_solver=scattering_solver)

        self.includes = ["general/general.arts",
                         "include_mpm89_cont.arts",
                         "general/agendas.arts",
                         "general/planet_earth.arts"]


