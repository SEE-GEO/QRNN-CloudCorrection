#!/bin/bash
#odb sql "SELECT lat,lon, zenith, LSM, vertco_reference_1, obsvalue, biascorr_fg, tbclear, fg_depar  FROM '/home/inderpreet/Dendrite/Projects/AWS-325GHz/MWHS/mwhs_06_2020.odb'" > output_06_2020.txt
odb sql "SELECT lat,lon, zenith, LSM, vertco_reference_1, obsvalue, biascorr_fg, tbclear, fg_depar  FROM '/home/inderpreet/Dendrite/Projects/AWS-325GHz/MWHS/mwhs_05_2020.odb'" > output_05_2020.txt
