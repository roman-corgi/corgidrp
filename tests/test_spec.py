import os
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from corgidrp.data import Dataset, SpectroscopyCentroidPSF, Image, DispersionModel, WavelengthZeropoint, WaveCal
import corgidrp.spec as steps
from corgidrp.mocks import create_default_L1_headers
from corgidrp.spec import get_template_dataset

datadir = os.path.join(os.path.dirname(__file__), "test_data", "spectroscopy")
spec_datadir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "corgidrp", "data", "spectroscopy"))
template_dir = os.path.join(spec_datadir, "templates")
output_dir = os.path.join(os.path.dirname(__file__), "testcalib")
os.makedirs(output_dir, exist_ok=True)


def convert_tvac_to_dataset():
    """
    for me to convert the tvac data once.
    """
    file_path = [os.path.join(datadir, "g0v_vmag6_spc-spec_band3_unocc_CFAM3d_NOSLIT_PRISM3_offset_array.fits"), 
                 os.path.join(datadir, "g0v_vmag6_spc-spec_band3_unocc_CFAM3d_R1C2SLIT_PRISM3_offset_array.fits")]
    pri_hdr, ext_hdr = create_default_L1_headers()
    for k, file in enumerate(file_path):
        with fits.open(file) as hdul:
            psf_array = hdul[0].data
            psf_table = Table(hdul[1].data)

        initial_cent = {
            "xcent": np.array(psf_table["xcent"]),
            "ycent": np.array(psf_table["ycent"]),
            "xoffset": np.array(psf_table["xoffset"]),
            "yoffset": np.array(psf_table["yoffset"])
        }
        assert len(initial_cent.get('xcent')) == psf_array.shape[0]
        assert len(initial_cent.get('xoffset')) == psf_array.shape[0]
    
        psf_images = []
        file_names = []
        for i in range(psf_array.shape[0]):
            data_2d = np.copy(psf_array[i])
            err = np.zeros_like(data_2d)
            dq = np.zeros_like(data_2d, dtype=int)
            image = Image(
                data_or_filepath=data_2d,
                pri_hdr=pri_hdr.copy(),
                ext_hdr=ext_hdr.copy(),
                err=err,
                dq=dq
            )
            image.ext_hdr['CFAMNAME'] = '3d'
            image.ext_hdr['DPAMNAME'] = 'PRISM3'
            if k == 0:
                image.ext_hdr['FSAMNAME'] = 'OPEN'
            else:
                image.ext_hdr['FSAMNAME'] = 'R1C2'
            image.ext_hdr['xcent']= initial_cent.get('xcent')[i]
            image.ext_hdr['ycent']= initial_cent.get('ycent')[i]
            image.ext_hdr['xoffset']= initial_cent.get('xoffset')[i]
            image.ext_hdr['yoffset']= initial_cent.get('yoffset')[i]
            psf_images.append(image)
            if i > 0 and i <10:
                num = "0"+str(i)
            else:
                num = str(i)
            if k == 0:
                file_names.append("spec_unocc_noslit_offset_prism3_3d_" +num+".fits")
            else:
                file_names.append("spec_unocc_r1c2slit_offset_prism3_3d_" +num+".fits")

        dataset = Dataset(psf_images)
        dataset.save(filedir=template_dir, filenames = file_names)
    
    file_path_filtersweep = os.path.join(datadir, "g0v_vmag6_spc-spec_band3_unocc_NOSLIT_PRISM3_filtersweep_withoffsets.fits")
    psf_array = fits.getdata(file_path_filtersweep, ext = 0)
    psf_table = Table(fits.getdata(file_path_filtersweep, ext = 1))
    psf_header = fits.getheader(file_path_filtersweep, ext = 0)
    psf_table_header = fits.getheader(file_path_filtersweep, ext = 1)
    
    initial_cent = {
        "xcent": np.array(psf_table["xcent"]),
        "ycent": np.array(psf_table["ycent"]),
        "xoffset": np.array(psf_table["xoffset"]),
        "yoffset": np.array(psf_table["yoffset"])
    }
    assert len(initial_cent.get('xcent')) == psf_array.shape[0]
    assert len(initial_cent.get('xoffset')) == psf_array.shape[0]
    
    psf_images = []
    file_names = []
    for i in range(psf_array.shape[0]):
        data_2d = np.copy(psf_array[i])
        err = np.zeros_like(data_2d)
        dq = np.zeros_like(data_2d, dtype=int)
        image = Image(
            data_or_filepath=data_2d,
            pri_hdr=pri_hdr.copy(),
            ext_hdr=ext_hdr.copy(),
            err=err,
            dq=dq
        )
        image.ext_hdr['CFAMNAME'] = psf_table['CFAM'][i]
        image.ext_hdr['DPAMNAME'] = 'PRISM3'
        image.ext_hdr['FSAMNAME'] = 'OPEN'
        image.ext_hdr['xcent']= initial_cent.get('xcent')[i]
        image.ext_hdr['ycent']= initial_cent.get('ycent')[i]
        image.ext_hdr['xoffset']= initial_cent.get('xoffset')[i]
        image.ext_hdr['yoffset']= initial_cent.get('yoffset')[i]
        psf_images.append(image)
        if i>0 and i <10:
            num = "0"+str(i)
        else:
            num = str(i)
        file_names.append("spec_unocc_noslit_offset_prism3_filtersweep_" +num+".fits")
    dataset = Dataset(psf_images)
    dataset.save(filedir=template_dir, filenames = file_names)

def test_psf_centroid():
    """
    Test PSF centroid computation with mock data and assert correctness of output FITS structure.
    """
    errortol_pix = 0.01
    file_path = os.path.join(datadir, "g0v_vmag6_spc-spec_band3_unocc_CFAM3d_NOSLIT_PRISM3_offset_array.fits")
    assert os.path.exists(file_path), f"Test FITS file not found: {file_path}"
    
    pri_hdr, ext_hdr = create_default_L1_headers()
    
    with fits.open(file_path) as hdul:
        psf_array = hdul[0].data
        psf_table = Table(hdul[1].data)
        #pri_hdr = hdul[0].header
        #ext_hdr = hdul[1].header

    assert psf_array.ndim == 3, "Expected 3D PSF array"
    assert "xcent" in psf_table.colnames and "ycent" in psf_table.colnames, "Missing centroid columns"

    initial_cent = {
        "xcent": np.array(psf_table["xcent"]),
        "ycent": np.array(psf_table["ycent"])
    }
    ext_hdr['DPAMNAME'] = 'PRISM3'
    ext_hdr['FSAMNAME'] = 'OPEN'
    psf_images = []
    for i in range(psf_array.shape[0]):
        data_2d = np.copy(psf_array[i])
        err = np.zeros_like(data_2d)
        dq = np.zeros_like(data_2d, dtype=int)
        image = Image(
            data_or_filepath=data_2d,
            pri_hdr=pri_hdr,
            ext_hdr=ext_hdr,
            err=err,
            dq=dq
        )
        image.ext_hdr['CFAMNAME'] = '3d'
        psf_images.append(image)

    dataset = Dataset(psf_images)

    calibration = steps.compute_psf_centroid(
        dataset=dataset,
        initial_cent=initial_cent
    )
    
    assert calibration.xfit.ndim == 1
    assert calibration.yfit.ndim == 1
    assert calibration.xfit_err.ndim == 1
    assert calibration.yfit_err.ndim == 1
    # Manually assign filedir and filename before saving
    calibration.filename = "centroid_calibration.fits"
    calibration.filedir = output_dir
    calibration.save()

    output_file = os.path.join(output_dir, calibration.filename)
    assert os.path.exists(output_file), "Calibration file not created"

    # Validate calibration file structure and contents
    with fits.open(output_file) as hdul:
        assert len(hdul) >= 2, "Expected at least 2 HDUs (primary + extension)"
        assert isinstance(hdul[0].header, fits.Header), "Missing primary header"
        assert isinstance(hdul[1].header, fits.Header), "Missing extension header"
        assert "EXTNAME" in hdul[1].header, "Extension header missing EXTNAME"
        assert hdul[1].header["EXTNAME"] == "CENTROIDS", "EXTNAME should be 'CENTROIDS'"

        centroid_data = hdul[1].data
        assert centroid_data is not None, "Centroid data missing"
        assert centroid_data.shape[1] == 2, "Centroid data should be shape (N, 2)"
        centroid_error = hdul[2].data
        assert centroid_error is not None, "Centroid error missing"
        assert centroid_error.shape[2] == 2, "Centroid error should be shape (N, 2)"

        print(f"Centroid FITS file validated: {centroid_data.shape[0]} rows")
        
    calibration_2 = steps.compute_psf_centroid(
        dataset=dataset
    )
    
    assert np.all(np.abs(calibration.xfit - initial_cent["xcent"]) < errortol_pix)
    assert np.all(np.abs(calibration.yfit - initial_cent["ycent"]) < errortol_pix)
    assert np.all(calibration.xfit_err < errortol_pix)
    assert np.all(calibration.yfit_err < errortol_pix)
    assert np.all(calibration_2.xfit_err < errortol_pix)
    assert np.all(calibration_2.yfit_err < errortol_pix)
    #accuracy lower without initial guess
    assert np.all(np.abs(calibration_2.xfit - initial_cent["xcent"]) < 1)
    assert np.all(np.abs(calibration_2.yfit - initial_cent["ycent"]) < 3)
    
    #use the default template file as input
    temp_dataset, filtersweep = get_template_dataset(dataset)
    calibration_3 = steps.compute_psf_centroid(
        dataset=dataset, template_dataset = temp_dataset, filtersweep = filtersweep
    )
    assert np.all(calibration_2.xfit == calibration_3.xfit)
    assert np.all(calibration_2.yfit == calibration_3.yfit)
    assert np.all(calibration_2.xfit_err == calibration_3.xfit_err)
    assert np.all(calibration_2.yfit_err == calibration_3.yfit_err)
    
def test_dispersion_model():
    global disp_dict
    prhdr, exthdr = create_default_L1_headers()
    disp_file_path = os.path.join(datadir, "TVAC_PRISM3_dispersion_profile.npz")
    assert os.path.exists(disp_file_path), f"Test file not found: {disp_file_path}"
    disp_params = np.load(disp_file_path)
    disp_dict = {'clocking_angle': disp_params['clocking_angle'],
                'clocking_angle_uncertainty': disp_params['clocking_angle_uncertainty'],
                'pos_vs_wavlen_polycoeff': disp_params['pos_vs_wavlen_polycoeff'],
                'pos_vs_wavlen_cov' : disp_params['pos_vs_wavlen_cov'],
                'wavlen_vs_pos_polycoeff': disp_params['wavlen_vs_pos_polycoeff'],
                'wavlen_vs_pos_cov': disp_params['wavlen_vs_pos_cov']}
    
    disp_model = DispersionModel(disp_dict, pri_hdr = prhdr, ext_hdr = exthdr)
    assert disp_model.clocking_angle == disp_dict.get('clocking_angle')
    assert disp_model.clocking_angle_uncertainty == disp_dict.get('clocking_angle_uncertainty')
    assert np.array_equal(disp_model.pos_vs_wavlen_polycoeff, disp_dict.get('pos_vs_wavlen_polycoeff'))
    assert np.array_equal(disp_model.pos_vs_wavlen_cov, disp_dict.get('pos_vs_wavlen_cov'))
    assert np.array_equal(disp_model.wavlen_vs_pos_polycoeff, disp_dict.get('wavlen_vs_pos_polycoeff'))
    assert np.array_equal(disp_model.wavlen_vs_pos_cov, disp_dict.get('wavlen_vs_pos_cov'))
    
    disp_model.save(output_dir, disp_model.filename)
    load_disp = DispersionModel(os.path.join(output_dir, disp_model.filename))
    assert load_disp.clocking_angle == disp_dict.get('clocking_angle')
    assert load_disp.clocking_angle_uncertainty == disp_dict.get('clocking_angle_uncertainty')
    assert np.array_equal(load_disp.pos_vs_wavlen_polycoeff, disp_dict.get('pos_vs_wavlen_polycoeff'))
    assert np.array_equal(load_disp.pos_vs_wavlen_cov, disp_dict.get('pos_vs_wavlen_cov'))
    assert np.array_equal(load_disp.wavlen_vs_pos_polycoeff, disp_dict.get('wavlen_vs_pos_polycoeff'))
    assert np.array_equal(load_disp.wavlen_vs_pos_cov, disp_dict.get('wavlen_vs_pos_cov'))

def test_read_cent_wave():
    band_file = os.path.join(spec_datadir, 'CGI_bandpass_centers.csv')
    cen_wave = steps.read_cent_wave(band_file, '3C')
    assert cen_wave == 726.0
    cen_wave = steps.read_cent_wave(band_file, '3G')
    assert cen_wave == 752.5
    with pytest.raises(ValueError):
        cen_wave = steps.read_cent_wave(band_file, 'X')
    
def test_calibrate_dispersion_model():    
    """
    Test PSF dispersion computation with mock data and assert correctness of output FITS structure.
    """
    
    global disp_model
    file_path = os.path.join(datadir, "g0v_vmag6_spc-spec_band3_unocc_NOSLIT_PRISM3_filtersweep_withoffsets.fits")
    assert os.path.exists(file_path), f"Test FITS file not found: {file_path}"
    
    prihdr, exthdr = create_default_L1_headers()
    exthdr["DPAMNAME"] = 'PRISM3'
    exthdr["FSAMNAME"] = 'OPEN'
    psf_array = fits.getdata(file_path, ext = 0)
    psf_table = Table(fits.getdata(file_path, ext = 1))
    psf_header = fits.getheader(file_path, ext = 0)
    psf_table_header = fits.getheader(file_path, ext = 1)
    
    assert psf_array.ndim == 3, "Expected 3D PSF array"
    assert "xcent" in psf_table.colnames and "ycent" in psf_table.colnames, "Missing centroid columns"

    # Add random noise to the filter sweep template images to serve as fake data
    np.random.seed(5)
    read_noise = 200
    noisy_data_array = (np.random.poisson(np.abs(psf_array) / 2) + 
                        np.random.normal(loc=0, scale=read_noise, size=psf_array.shape))
    psf_images = []
    for i in range(noisy_data_array.shape[0]):
        data_2d = np.copy(noisy_data_array[i])
        err = np.zeros_like(data_2d)
        dq = np.zeros_like(data_2d, dtype=int)
        image = Image(
            data_or_filepath=data_2d,
            pri_hdr=prihdr.copy(),
            ext_hdr=exthdr.copy(),
            err=err,
            dq=dq
        )
        image.ext_hdr['CFAMNAME'] = psf_table['CFAM'][i]
        psf_images.append(image)

    dataset = Dataset(psf_images)

    psf_centroid = steps.compute_psf_centroid(
        dataset=dataset
    )
    #calibrate dispersion without the broad band fit
    psf_centroid.xfit = psf_centroid.xfit[:-1] - (np.array(psf_table['xoffset'])[:-1] - np.array(psf_table['xoffset'])[-1])
    psf_centroid.yfit = psf_centroid.yfit[:-1] - (np.array(psf_table['yoffset'])[:-1] - np.array(psf_table['yoffset'])[-1])
    psf_centroid.xfit_err = psf_centroid.xfit_err[:-1]
    psf_centroid.yfit_err = psf_centroid.yfit_err[:-1]
    
    disp_model = steps.calibrate_dispersion_model(psf_centroid, prism = 'PRISM3')
    disp_model.save(output_dir, disp_model.filename)
    assert disp_model.filename.startswith("DispersionModel")
    assert disp_model.clocking_angle == pytest.approx(psf_header["PRISMANG"], abs = 2 * disp_model.clocking_angle_uncertainty) 
    
    pos_func_wavlen = np.poly1d(disp_model.pos_vs_wavlen_polycoeff)
    wavlen_func_pos = np.poly1d(disp_model.wavlen_vs_pos_polycoeff)
    
    #read the TVAC result of PRISM3 and compare
    ref_wavlen = 730.0
    bandpass = [675, 785]  
    tvac_pos_vs_wavlen_polycoeff = disp_dict.get('pos_vs_wavlen_polycoeff')
    tvac_pos_vs_wavlen_cov = disp_dict.get('pos_vs_wavlen_cov')
    tvac_wavlen_vs_pos_polycoeff = disp_dict.get('wavlen_vs_pos_polycoeff')
    tvac_wavlen_vs_pos_cov = disp_dict.get('wavlen_vs_pos_cov')
    tvac_pos_func_wavlen = np.poly1d(tvac_pos_vs_wavlen_polycoeff)
    tvac_wavlen_func_pos = np.poly1d(tvac_wavlen_vs_pos_polycoeff)
    
    (xtest_min, xtest_max) = (tvac_pos_func_wavlen((bandpass[0] - ref_wavlen)/ref_wavlen),
                              tvac_pos_func_wavlen((bandpass[1] - ref_wavlen)/ref_wavlen))
    xtest = np.linspace(xtest_min, xtest_max, 1000)
    tvac_model_wavlens = tvac_wavlen_func_pos(xtest)
    corgi_model_wavlens = wavlen_func_pos(xtest)
    wavlen_model_error = corgi_model_wavlens - tvac_model_wavlens
    worst_case_wavlen_error = np.abs(wavlen_model_error).max()
    print(f"Worst case wavelength disagreement from the test input model: {worst_case_wavlen_error:.2f} nm")
    assert worst_case_wavlen_error == pytest.approx(0, abs=0.5)
    print("Dispersion profile fit test passed.")
    
def test_wave_zeropoint():
    """
    test the WavelengthZeropoint calibration class
    """
    wave_0 = {"prism": "PRISM3",
              "wavlen": 730.,
              'x': 3.5,
              'xerr': 0.1,
              'y': 4.5,
              'yerr': 0.1,
              'shape0': 61,
              'shape1': 21}
    
    w_zero = WavelengthZeropoint(wave_0)
    
    assert w_zero.prism == wave_0.get("prism")
    assert w_zero.x == wave_0.get("x")
    assert w_zero.xerr == wave_0.get("xerr")
    assert w_zero.y == wave_0.get("y")
    assert w_zero.yerr == wave_0.get("yerr")
    assert w_zero.image_shape[0] == wave_0.get("shape0")
    assert w_zero.image_shape[1] == wave_0.get("shape1")
    
    w_zero.save(filedir=output_dir, filename = "mock_wave_zeropoint.fits")
    w_name = os.path.join(output_dir, "mock_wave_zeropoint.fits")
    w_zero_load = WavelengthZeropoint(w_name)
    
    assert w_zero_load.prism == wave_0.get("prism")
    assert w_zero_load.x == wave_0.get("x")
    assert w_zero_load.xerr == wave_0.get("xerr")
    assert w_zero_load.y == wave_0.get("y")
    assert w_zero_load.yerr == wave_0.get("yerr")
    assert w_zero_load.image_shape[0] == wave_0.get("shape0")
    assert w_zero_load.image_shape[1] == wave_0.get("shape1")

def test_create_wave_cal():
    """
    test the WaveCal calibration class and generation of wavelength map.
    """
    wave_zero_name = os.path.join(output_dir, "mock_wave_zeropoint.fits")
    wave_zero = WavelengthZeropoint(wave_zero_name)
    ref_wavlen = wave_zero.wavlen
    wavecal = steps.create_wave_cal(disp_model, wave_zero, ref_wavlen = ref_wavlen)
    assert wavecal.data.shape == wave_zero.image_shape
    # assert that the wavelengths are within bandpass 3
    assert ref_wavlen > np.min(wavecal.data) and ref_wavlen < np.max(wavecal.data)
    assert wavecal.err[0].shape == wave_zero.image_shape
    assert hasattr(wavecal, 'pos_lookup')
    assert len(wavecal.pos_lookup.colnames) == 5
    assert np.allclose(wavecal.pos_lookup.columns[0].data, ref_wavlen, atol = 65)
    
    #test the saving and loading of the cal file
    wavecal.save(filedir = output_dir, filename = "mock_wavecal.fits")
    wc_name = os.path.join(output_dir, "mock_wavecal.fits")
    wavecal_load = WaveCal(wc_name)
    assert np.array_equal(wavecal_load.data, wavecal.data)
    assert np.array_equal(wavecal_load.err, wavecal.err)
    assert np.array_equal(wavecal.pos_lookup, wavecal_load.pos_lookup)
    
    #test the WaveCal without saving the position lookup table
    wave_without = steps.create_wave_cal(disp_model, wave_zero, ref_wavlen = ref_wavlen, lookup_table = False)
    assert not hasattr(wave_without, 'pos_lookup')
    assert np.array_equal(wave_without.data, wavecal.data)
    assert np.allclose(wave_without.err, wavecal.err, atol = 0.15)
    
    wave_without.save(filedir = output_dir, filename = "mock_wavecal_without.fits")
    wc_name_wo = os.path.join(output_dir, "mock_wavecal_without.fits")
    wave_without_load = WaveCal(wc_name_wo)
    assert np.array_equal(wave_without_load.data, wave_without.data)
    assert np.array_equal(wave_without_load.err, wave_without.err)
    
if __name__ == "__main__":
    #convert_tvac_to_dataset()
    test_psf_centroid()
    test_dispersion_model()
    test_read_cent_wave()
    test_calibrate_dispersion_model()
    test_wave_zeropoint()
    test_create_wave_cal()
