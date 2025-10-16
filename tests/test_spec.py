import os
import numpy as np
import pytest
import logging
from astropy.io import fits
from astropy.table import Table
from corgidrp.data import Dataset, Image, DispersionModel, LineSpread
import corgidrp.spec as steps
from corgidrp.mocks import create_default_L2b_headers, get_formatted_filename
from corgidrp.spec import get_template_dataset
import corgidrp.l3_to_l4 as l3_to_l4
from datetime import datetime, timedelta

spec_datadir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', "corgidrp", "data", "spectroscopy")) 
test_datadir = os.path.join(os.path.dirname(__file__), "test_data", "spectroscopy")
template_dir = os.path.join(spec_datadir, "templates")
output_dir = os.path.join(os.path.dirname(__file__), "testcalib")
os.makedirs(output_dir, exist_ok=True)

def convert_tvac_to_dataset():
    """
    for me to convert the tvac data once.
    """

    file_path = [os.path.join(test_datadir, "g0v_vmag6_spc-spec_band3_unocc_CFAM3d_NOSLIT_PRISM3_offset_array.fits"), 
                 os.path.join(test_datadir, "g0v_vmag6_spc-spec_band3_unocc_CFAM3d_R1C2SLIT_PRISM3_offset_array.fits")]
    pri_hdr, ext_hdr, errhdr, dqhdr, biashdr = create_default_L2b_headers()

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
            image.ext_hdr['CFAMNAME'] = '3D'
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
    
    file_path_filtersweep = os.path.join(test_datadir, "g0v_vmag6_spc-spec_band3_unocc_NOSLIT_PRISM3_filtersweep.fits")
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
        
        image.ext_hdr['CFAMNAME'] = psf_table['CFAM'][i].upper()
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
    file_path = os.path.join(test_datadir, "g0v_vmag6_spc-spec_band3_unocc_CFAM3d_NOSLIT_PRISM3_offset_array.fits")
    assert os.path.exists(file_path), f"Test FITS file not found: {file_path}"
    
    pri_hdr, ext_hdr, errhdr, dqhdr, biashdr = create_default_L2b_headers()
    
    with fits.open(file_path) as hdul:
        psf_array = hdul[0].data
        psf_table = Table(hdul[1].data)

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
        image.ext_hdr['CFAMNAME'] = '3D'
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

    pri_hdr, ext_hdr, errhdr, dqhdr, biashdr = create_default_L2b_headers()
    disp_file_path = os.path.join(spec_datadir, "TVAC_PRISM3_dispersion_profile.npz")

    assert os.path.exists(disp_file_path), f"Test file not found: {disp_file_path}"
    disp_params = np.load(disp_file_path)
    disp_dict = {'clocking_angle': disp_params['clocking_angle'],
                'clocking_angle_uncertainty': disp_params['clocking_angle_uncertainty'],
                'pos_vs_wavlen_polycoeff': disp_params['pos_vs_wavlen_polycoeff'],
                'pos_vs_wavlen_cov' : disp_params['pos_vs_wavlen_cov'],
                'wavlen_vs_pos_polycoeff': disp_params['wavlen_vs_pos_polycoeff'],
                'wavlen_vs_pos_cov': disp_params['wavlen_vs_pos_cov']}
    
    disp_model = DispersionModel(disp_dict, pri_hdr = pri_hdr, ext_hdr = ext_hdr)
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
    cen_wave = steps.read_cent_wave('3C')[0]
    assert cen_wave == 726.0
    cen_wave_list = steps.read_cent_wave('3G')
    assert cen_wave_list[0] == 752.5
    assert len(cen_wave_list) == 4
    with pytest.raises(ValueError):
        cen_wave = steps.read_cent_wave('X')[0]
    
    cen_wave_list = steps.read_cent_wave('3')
    assert len(cen_wave_list) == 4
    assert cen_wave_list[0] == 729.3
    assert cen_wave_list[1] == 122.3
    assert cen_wave_list[2] == 0.725909
    assert cen_wave_list[3] == -0.09398
    
    
def test_calibrate_dispersion_model():    
    """
    Test PSF dispersion computation with mock data and assert correctness of output FITS structure.
    """
    
    global disp_model
    file_path = os.path.join(test_datadir, "g0v_vmag6_spc-spec_band3_unocc_NOSLIT_PRISM3_filtersweep_withoffsets.fits")
    assert os.path.exists(file_path), f"Test FITS file not found: {file_path}"
    
    pri_hdr, ext_hdr, errhdr, dqhdr, biashdr = create_default_L2b_headers()
    ext_hdr["DPAMNAME"] = 'PRISM3'
    ext_hdr["FSAMNAME"] = 'OPEN'
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
            pri_hdr=pri_hdr.copy(),
            ext_hdr=ext_hdr.copy(),
            err=err,
            dq=dq
        )
        image.ext_hdr['CFAMNAME'] = psf_table['CFAM'][i].upper()
        psf_images.append(image)

    dataset = Dataset(psf_images)

    psf_centroid = steps.compute_psf_centroid(
        dataset=dataset
    )
    
    disp_model = steps.calibrate_dispersion_model(psf_centroid)
    disp_model.save(output_dir, disp_model.filename)
    assert disp_model.filename.endswith("dpm_cal.fits")
    assert disp_model.clocking_angle == pytest.approx(psf_header["PRISMANG"], abs = 2 * disp_model.clocking_angle_uncertainty) 
    
    wavlen_func_pos = np.poly1d(disp_model.wavlen_vs_pos_polycoeff)
    
    #read the TVAC result of PRISM3 and compare
    ref_wavlen = 730.
    bandpass = [675, 785]  
    tvac_pos_vs_wavlen_polycoeff = disp_dict.get('pos_vs_wavlen_polycoeff')
    tvac_wavlen_vs_pos_polycoeff = disp_dict.get('wavlen_vs_pos_polycoeff')
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

def test_add_wavelength_map():
    """
    test l3_to_l4.add_wavelength_map(), the generation of the wavelength map extensions
    this test requires running test_calibrate_dispersion_model() beforehand, since the function 
    needs the DispersionModel calibration file as input
    """
    # invented wavelength zero point
    wave_0 = {"wavlen": 753.83,
              'x': 40.,
              'xerr': 0.1,
              'y': 32.,
              'yerr': 0.1,
              'shapex': 81,
              'shapey': 81}
    
    ref_wavlen = disp_model.ext_hdr["REFWAVE"]
    filepath = os.path.join(spec_datadir, "templates", "spec_unocc_noslit_offset_prism3_3d_12.fits")
    image = Image(filepath)
    image.ext_hdr['CFAMNAME'] = '3D'
    image.ext_hdr['WAVLEN0'] = wave_0.get('wavlen')
    image.ext_hdr['WV0_X'] = wave_0.get('x')
    image.ext_hdr['WV0_XERR'] = wave_0.get('xerr')
    image.ext_hdr['WV0_Y'] = wave_0.get('y')
    image.ext_hdr['WV0_YERR'] = wave_0.get('yerr')
    image.ext_hdr['WV0_DIMX'] = wave_0.get('shapex')
    image.ext_hdr['WV0_DIMY'] = wave_0.get('shapey')
    dataset = Dataset([image])
    
    global output_dataset
    output_dataset = l3_to_l4.add_wavelength_map(dataset, disp_model)
    
    out_im = output_dataset.frames[0]
    wave = out_im.hdu_list["wave"].data
    assert wave.shape == (wave_0.get('shapex'), wave_0.get('shapey'))
    wave_err = out_im.hdu_list["wave_err"].data
    assert wave_err.shape == (wave_0.get('shapex'), wave_0.get('shapey'))
    # assert that the wavelengths are within bandpass 3
    assert ref_wavlen > np.min(wave) and ref_wavlen < np.max(wave)
    assert wave_0.get('wavlen') > np.min(wave) and  wave_0.get('wavlen') < np.max(wave)
    #show that the ref_wavlen is approx. in the center of the array and the narrow band center wavelength at the max of the psf
    c_max = np.where(image.data == np.max(image.data))
    wave_center_band = wave[c_max[0][0],c_max[1][0]]
    wave_ref_band = wave[40,40]
    assert wave_0.get('wavlen') == pytest.approx(wave_center_band, abs = 0.1) 
    assert ref_wavlen == pytest.approx(wave_ref_band, abs = 1.5)
    assert wave_err.shape[0] == wave_0.get("shapex")
    assert wave_err.shape[1] == wave_0.get("shapey")
    wave_hdr = out_im.hdu_list["wave"].header
    #position of ref wavelength should be in array center 
    assert wave_hdr["XREFWAV"] == pytest.approx(40, abs = 1.)
    assert wave_hdr["YREFWAV"] == pytest.approx(40, abs = 1.)
    assert wave_hdr["REFWAVE"] ==730.

    #Worst case wavelength uncertainty should be smaller than 1 nm
    assert np.max(wave_err) == pytest.approx(0., abs = 1.)
    pos_lookup = Table(out_im.hdu_list["poslookup"].data)
    assert len(pos_lookup.colnames) == 5
    assert np.allclose(pos_lookup.columns[0].data, ref_wavlen, atol = 65)
    
def test_determine_zeropoint():
    """
    test the calculation of the wavelength zeropoint position of narrowband/satspot data
    """
    errortol_pix = 0.5
    filepath = os.path.join(test_datadir, "g0v_vmag6_spc-spec_band3_unocc_CFAM3d_R1C2SLIT_PRISM3_offset_array.fits")
    pri_hdr, ext_hdr = create_default_L2b_headers()[:2]
    
    with fits.open(filepath) as hdul:
        psf_array = hdul[0].data
        psf_table = Table(hdul[1].data)
 
    assert psf_array.ndim == 3, "Expected 3D PSF array"
    assert "xcent" in psf_table.colnames and "ycent" in psf_table.colnames, "Missing centroid columns"

    initial_cent = {
        "xcent": np.array(psf_table["xcent"]),
        "ycent": np.array(psf_table["ycent"])
    }
    offset_cent = {
        "xoffset": np.array(psf_table["xoffset"]),
        "yoffset": np.array(psf_table["yoffset"])
    }
    # Use the position for the zero-offset PSF template to set the projected,
    # vertical slit position on the image array. This should be in the exact
    # center of the array. 
    # the zero offset template image is our fake satspot observation in the slit center
    assert(offset_cent.get("xoffset")[12] == 0.)
    assert(offset_cent.get("yoffset")[12] == 0.)
    slit_x = initial_cent.get("xcent")[12]
    slit_y = initial_cent.get("ycent")[12]

    ext_hdr['DPAMNAME'] = 'PRISM3'
    ext_hdr['FSAMNAME'] = 'R1C2'
    psf_images = []
    for i in range(psf_array.shape[0]):
        data_2d = np.copy(psf_array[i])
        ext_hdr["NAXIS1"] =np.shape(data_2d)[0]
        ext_hdr["NAXIS2"] =np.shape(data_2d)[1]
        ext_hdr['CFAMNAME'] = '3'
        if i == 12:
            pri_hdr["SATSPOTS"] = 1
            ext_hdr['CFAMNAME'] = '3D'
        else:
            pri_hdr["SATSPOTS"] = 0
        err = np.zeros_like(data_2d)
        dq = np.zeros_like(data_2d, dtype=int)
        image = Image(
            data_or_filepath=data_2d,
            pri_hdr=pri_hdr.copy(),
            ext_hdr=ext_hdr.copy(),
            err=err,
            dq=dq
        )
        psf_images.append(image)

    #test it with optional initial guess and with one satspot frame
    input_dataset = Dataset(psf_images)
    dataset_guess = l3_to_l4.determine_wave_zeropoint(input_dataset, xcent_guess = 40., ycent_guess = 32.)

    assert len(dataset_guess) < len(input_dataset)
    for frame in dataset_guess:
        assert frame.pri_hdr["SATSPOTS"] == 0
        assert frame.ext_hdr["WAVLEN0"] == 753.83
        assert "WV0_X" in frame.ext_hdr
        assert "WV0_Y" in frame.ext_hdr
        assert "WV0_XERR" in frame.ext_hdr
        assert "WV0_YERR" in frame.ext_hdr
        assert frame.ext_hdr["WV0_DIMX"] == 81
        assert frame.ext_hdr["WV0_DIMY"] == 81
        x0 = frame.ext_hdr["WV0_X"]
        y0 = frame.ext_hdr["WV0_Y"]
        x0err = frame.ext_hdr["WV0_XERR"]
        y0err = frame.ext_hdr["WV0_YERR"]
        assert x0 == pytest.approx(slit_x, abs = errortol_pix)
        assert y0 == pytest.approx(slit_y, abs = errortol_pix)
        assert x0err < errortol_pix
        assert y0err < errortol_pix
    
    psf_images = []
    for i in range(psf_array.shape[0]):
        data_2d = np.copy(psf_array[i])
        ext_hdr["NAXIS1"] =np.shape(data_2d)[0]
        ext_hdr["NAXIS2"] =np.shape(data_2d)[1]
        ext_hdr['CFAMNAME'] = '3D'
        pri_hdr["SATSPOTS"] = 0
        err = np.zeros_like(data_2d)
        dq = np.zeros_like(data_2d, dtype=int)
        image = Image(
            data_or_filepath=data_2d,
            pri_hdr=pri_hdr.copy(),
            ext_hdr=ext_hdr.copy(),
            err=err,
            dq=dq
        )
        psf_images.append(image)

    #test it as non-coronagraphic observation of only psf narrowband, so no science frames
    input_dataset2 = Dataset(psf_images)
    with pytest.raises(AttributeError):
        dataset = l3_to_l4.determine_wave_zeropoint(input_dataset2)
    
    #only 1 fake science dataset frame
    input_dataset2.frames[0].ext_hdr['CFAMNAME'] = '3'
    dataset = l3_to_l4.determine_wave_zeropoint(input_dataset2)
    assert len(dataset) == 1
    for frame in dataset:
        assert frame.pri_hdr["SATSPOTS"] == 0
        assert frame.ext_hdr["WAVLEN0"] == 753.83
        assert "WV0_X" in frame.ext_hdr
        assert "WV0_Y" in frame.ext_hdr
        assert "WV0_XERR" in frame.ext_hdr
        assert "WV0_YERR" in frame.ext_hdr
        assert frame.ext_hdr["WV0_DIMX"] == 81
        assert frame.ext_hdr["WV0_DIMY"] == 81
        x0 = frame.ext_hdr["WV0_X"]
        y0 = frame.ext_hdr["WV0_Y"]
        x0err = frame.ext_hdr["WV0_XERR"]
        y0err = frame.ext_hdr["WV0_YERR"]
        assert x0 == pytest.approx(slit_x, abs = errortol_pix)
        assert y0 == pytest.approx(slit_y, abs = errortol_pix)
        assert x0err < errortol_pix
        assert y0err < errortol_pix
    
    #to test the accuracy add noise to the dataset frames
    read_noise = 200
    np.random.seed(0)

    noise_dataset = input_dataset.copy()
    for frame in noise_dataset:
        frame.data = np.random.poisson(np.abs(frame.data)/3) + \
        np.random.normal(loc=0, scale=read_noise, size = frame.data.shape)
    noisci_dataset = l3_to_l4.determine_wave_zeropoint(noise_dataset)
    for i in range(len(noisci_dataset)):
        x0_noi = noisci_dataset[i].ext_hdr["WV0_X"]
        y0_noi = noisci_dataset[i].ext_hdr["WV0_Y"]
        x0err_noi = noisci_dataset[i].ext_hdr["WV0_XERR"]
        y0err_noi = noisci_dataset[i].ext_hdr["WV0_YERR"]
        x0 = dataset_guess[i].ext_hdr["WV0_X"]
        y0 = dataset_guess[i].ext_hdr["WV0_Y"]
        x0err = dataset_guess[i].ext_hdr["WV0_XERR"]
        y0err = dataset_guess[i].ext_hdr["WV0_YERR"]
        assert x0 == pytest.approx(x0_noi, abs = errortol_pix)
        assert y0 == pytest.approx(y0_noi, abs = errortol_pix)
        assert x0err_noi < errortol_pix
        assert y0err_noi < errortol_pix
    
def test_linespread_function():
    """
    test the fit of a linespread function to a narrowband observation and storing in a LineSpread calibration file
    using the output_dataset of the test of the wavelength map
    """
    line_spread = steps.fit_line_spread_function(output_dataset)
    xcent_round, ycent_round = (int(np.rint(output_dataset[0].ext_hdr["WV0_X"])), int(np.rint(output_dataset[0].ext_hdr["WV0_Y"])))
    image = output_dataset[0].data
    flux = np.sum(image, axis = 1)/np.sum(image)
    pos_max = np.argmax(flux)
    mean_wave = np.mean(output_dataset[0].hdu_list["WAVE"].data, axis = 1)
    assert line_spread.amplitude == pytest.approx(flux[pos_max], abs=0.04)
    assert line_spread.mean_wave == pytest.approx(mean_wave[pos_max], abs=2)
    ind_fwhm = np.where(flux >= flux[pos_max]/2.)[0]
    est_fwhm = mean_wave[ind_fwhm[0]] - mean_wave[ind_fwhm[-1]]
    assert est_fwhm == pytest.approx(line_spread.fwhm, abs = 3)
    assert np.min(mean_wave) < np.min(line_spread.wavlens)
    assert np.max(mean_wave) > np.max(line_spread.wavlens)
    assert np.min(flux) == pytest.approx(np.min(line_spread.flux_profile), abs = 0.001)
    assert np.max(flux) == pytest.approx(np.max(line_spread.flux_profile), abs = 0.035)
    line_spread.save(filedir = output_dir)
    
    #load the calibration fits file and check whether the content is unchanged
    line_spread_load = LineSpread(os.path.join(output_dir, line_spread.filename))
    assert np.array_equal(line_spread_load.gauss_par, np.array([line_spread.amplitude, line_spread.mean_wave, line_spread.fwhm, line_spread.amp_err, line_spread.wave_err, line_spread.fwhm_err]))
    assert np.array_equal(line_spread.flux_profile, line_spread_load.flux_profile)
    assert np.array_equal(line_spread.wavlens, line_spread_load.wavlens)
    assert line_spread_load.amplitude == line_spread.amplitude
    assert line_spread_load.fwhm == line_spread.fwhm
    assert line_spread_load.mean_wave == line_spread.mean_wave
    assert line_spread_load.amp_err == line_spread.amp_err
    assert line_spread_load.fwhm_err == line_spread.fwhm_err
    assert line_spread_load.wave_err == line_spread.wave_err
    
    #add a bad pixel and check the result
    bad_dataset = output_dataset.copy()
    for frame in bad_dataset:
        frame.dq[31, 40] = 1
        frame.data[10,10] = np.nan 
    line_spread_bad = steps.fit_line_spread_function(bad_dataset)
    assert line_spread_bad.fwhm == pytest.approx(line_spread.fwhm, rel = 0.1)
    assert line_spread_bad.amplitude == pytest.approx(line_spread.amplitude, rel = 0.1)

def test_spec_psf_subtraction():
    """
    test the spec PSF subtraction 
    """
    # first testing helper function 
    r=np.array([[0,0,1,2,3,0,0,0],[0,0,0,0,0,0,0,0]])
    s = np.roll(r,1)
    shift = steps.get_shift_correlation(r,s)
    shifted_s = np.roll(s, shift, axis=(0,1))
    assert np.array_equal(shifted_s, r)

    # now the PSF subtraction
    ref_filepath = os.path.join("tests", "test_data", "spectroscopy", "sim_rdi_L1", "spec_sim_rdi_reference_L1.fits")
    sci_filepath = os.path.join("tests", "test_data", "spectroscopy", "sim_rdi_L1", "spec_sim_rdi_target_L1.fits")
    input_dset = Dataset([sci_filepath, ref_filepath]) 
    input_dset[0].ext_hdr['PSFREF'] = False
    input_dset[1].ext_hdr['PSFREF'] = True
    for img in input_dset:
        img.data = img.data.astype(float)
        img.dq = np.zeros_like(img.data, dtype=int)
        # roughly the center of the imaged area
        img.ext_hdr['WV0_X'] = 1600
        img.ext_hdr['WV0_Y'] = 547
        np.random.seed(1039)
        img.err = np.random.randint(0,100, (1, img.data.shape[0],img.data.shape[1])).astype(float)
    input_dset[1].dq[533, 1600] = 1
    output = l3_to_l4.spec_psf_subtraction(input_dset)
    shift = steps.get_shift_correlation(input_dset[1].data, input_dset[0].data)
    # check that the subtraction actually decreased the residual because of the rescaling in each band
    #nanmean b/c mean-combining frames for ref frame exposed the DQ pixels as NaNs in the image
    assert np.mean(input_dset[0].data - input_dset[1].data) > np.nanmean(output[0].data) 
    assert output[1].dq[533,1600] == 1
    assert output[0].dq[533-shift[0], 1600-shift[1]] == 1
    # way outside the image region, no additional error added
    assert np.array_equal(output[0].err[0,0:100,0:100], input_dset[0].err[0,0:100,0:100])
    # example where we know the exact solution
    

if __name__ == "__main__":
    #convert_tvac_to_dataset()
    test_spec_psf_subtraction()
    test_determine_zeropoint()
    test_psf_centroid()
    test_dispersion_model()
    test_read_cent_wave()
    test_calibrate_dispersion_model()
    test_add_wavelength_map()
    test_linespread_function()
