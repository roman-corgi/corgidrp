
from corgidrp.mocks import create_synthesized_master_dark_calib
from corgidrp.data import DetectorParams
from corgidrp.darks import calibrate_darks_lsq
from corgidrp.mocks import detector_areas_test as dat


dataset = create_synthesized_master_dark_calib(dat)
detector_params = DetectorParams({})
weighting = True

noise_maps = calibrate_darks_lsq(dataset, detector_params, weighting, dat)

print("noise_maps:", noise_maps)

