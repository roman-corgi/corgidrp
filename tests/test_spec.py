import os
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from corgidrp.data import Dataset, SpectroscopyCentroidPSF, Image, DispersionModel
import corgidrp.spec as steps
from corgidrp.mocks import create_default_L1_headers
from corgidrp.spec import get_template_dataset
import corgidrp.l3_to_l4 as l3_to_l4

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
        
        #for now only one image needed as template
        dataset = Dataset([psf_images[12]])
        dataset.save(filedir=template_dir, filenames = [file_names[12]])
    
    file_path_filtersweep = os.path.join(datadir, "g0v_vmag6_spc-spec_band3_unocc_NOSLIT_PRISM3_filtersweep.fits")
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
        file_names.append("spec_unocc_noslit_prism3_filtersweep_" +num+".fits")
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
    #accuracy better without initial guess
    assert np.all(np.abs(calibration_2.xfit - initial_cent["xcent"]) < errortol_pix)
    assert np.all(np.abs(calibration_2.yfit - initial_cent["ycent"]) < errortol_pix)
    
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
    cen_wave = steps.read_cent_wave('3C')
    assert cen_wave == 726.0
    cen_wave = steps.read_cent_wave('3G')
    assert cen_wave == 752.5
    with pytest.raises(ValueError):
        cen_wave = steps.read_cent_wave('X')
    
    cen_wave, fwhm = steps.read_cent_wave('3')
    assert cen_wave == 729.3
    assert fwhm == 122.3
    
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
    
    disp_model = steps.calibrate_dispersion_model(psf_centroid)
    disp_model.save(output_dir, disp_model.filename)
    assert disp_model.filename.startswith("DispersionModel")
    assert disp_model.clocking_angle == pytest.approx(psf_header["PRISMANG"], abs = 2 * disp_model.clocking_angle_uncertainty) 
    
    pos_func_wavlen = np.poly1d(disp_model.pos_vs_wavlen_polycoeff)
    wavlen_func_pos = np.poly1d(disp_model.wavlen_vs_pos_polycoeff)
    
    #read the TVAC result of PRISM3 and compare
    ref_wavlen = 730.
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

def test_wave_cal():
    """
    test l3_to_l4.wave_cal(), the generation of the wavelength map extensions
    """
    # invented wavelength zero point
    wave_0 = {"wavlen": 753.83,
              'x': 3.5,
              'xerr': 0.1,
              'y': 4.5,
              'yerr': 0.1,
              'shapex': 61,
              'shapey': 21}
    
    ref_wavlen = disp_model.ext_hdr["REFWAVE"]
    filepath = os.path.join(spec_datadir, "templates", "spec_unocc_noslit_offset_prism3_3d_12.fits")
    image = Image(filepath)
    image.ext_hdr['wavlen0'] = wave_0.get('wavlen')
    image.ext_hdr['x0'] = wave_0.get('x')
    image.ext_hdr['x0err'] = wave_0.get('xerr')
    image.ext_hdr['y0'] = wave_0.get('y')
    image.ext_hdr['y0err'] = wave_0.get('yerr')
    image.ext_hdr['shapex0'] = wave_0.get('shapex')
    image.ext_hdr['shapey0'] = wave_0.get('shapey')
    dataset = Dataset([image])
    
    output_dataset = l3_to_l4.wave_cal(dataset, disp_model)
    
    out_im = output_dataset.frames[0]
    
    wave = out_im.hdu_list["wave"].data
    assert wave.shape == (wave_0.get('shapex'), wave_0.get('shapey'))
    wave_err = out_im.hdu_list["wave_err"].data
    assert wave_err.shape == (wave_0.get('shapex'), wave_0.get('shapey'))
    # assert that the wavelengths are within bandpass 3
    assert ref_wavlen > np.min(wave) and ref_wavlen < np.max(wave)
    assert wave_err.shape[0] == wave_0.get("shapex")
    assert wave_err.shape[1] == wave_0.get("shapey")
    #Worst case wavelength uncertainty should be smaller than 1 nm
    assert np.max(wave_err) == pytest.approx(0., abs = 1.)
    pos_lookup = Table(out_im.hdu_list["poslookup"].data)
    assert len(pos_lookup.colnames) == 5
    assert np.allclose(pos_lookup.columns[0].data, ref_wavlen, atol = 65)
    
    
if __name__ == "__main__":
    #convert_tvac_to_dataset()
    test_psf_centroid()
    test_dispersion_model()
    test_read_cent_wave()
    test_calibrate_dispersion_model()
    test_wave_cal()
