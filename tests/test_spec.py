import os
import numpy as np
import pytest
from astropy.io import fits
from astropy.table import Table
from corgidrp.data import Dataset, Image, DispersionModel
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

    file_path = [os.path.join(spec_datadir, "g0v_vmag6_spc-spec_band3_unocc_CFAM3d_NOSLIT_PRISM3_offset_array.fits"), 
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
        basetime = datetime.now()
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
            
            # Generate timestamp for this file
            dt = basetime + timedelta(seconds=i)
            filename = get_formatted_filename(dt, pri_hdr['VISITID'])
            file_names.append(filename)
        
        # Sort by CFAMNAME for deterministic output
        sorted_indices = sorted(range(len(psf_images)), key=lambda x: psf_images[x].ext_hdr['CFAMNAME'])
        psf_images = [psf_images[i] for i in sorted_indices]
        file_names = [file_names[i] for i in sorted_indices]
        
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
    basetime = datetime.now()
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
        
        # Generate timestamp for this file
        dt = basetime + timedelta(seconds=i)
        filename = get_formatted_filename(dt, pri_hdr['VISITID'])
        file_names.append(filename)
    
    # Sort by CFAMNAME for deterministic output
    sorted_indices = sorted(range(len(psf_images)), key=lambda x: psf_images[x].ext_hdr['CFAMNAME'])
    psf_images = [psf_images[i] for i in sorted_indices]
    file_names = [file_names[i] for i in sorted_indices]
    
    dataset = Dataset(psf_images)
    dataset.save(filedir=template_dir, filenames = file_names)

def test_psf_centroid():
    """
    Test PSF centroid computation with mock data and assert correctness of output FITS structure.
    """
    errortol_pix = 0.01
    file_path = os.path.join(spec_datadir, "g0v_vmag6_spc-spec_band3_unocc_CFAM3d_NOSLIT_PRISM3_offset_array.fits")
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
        image.ext_hdr['CFAMNAME'] = psf_table['CFAM'][i]
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
    image.ext_hdr['WAVLEN0'] = wave_0.get('wavlen')
    image.ext_hdr['WV0_X'] = wave_0.get('x')
    image.ext_hdr['WV0_XERR'] = wave_0.get('xerr')
    image.ext_hdr['WV0_Y'] = wave_0.get('y')
    image.ext_hdr['WV0_YERR'] = wave_0.get('yerr')
    image.ext_hdr['WV0_DIMX'] = wave_0.get('shapex')
    image.ext_hdr['WV0_DIMY'] = wave_0.get('shapey')
    dataset = Dataset([image])
    
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
    test the calculation of the wavelength zeropoint position of satspot data
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
            ext_hdr['CFAMNAME'] = '3d'
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
        ext_hdr['CFAMNAME'] = '3d'
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

def test_star_spec_registration():
    """ Test the star spectrum registration """

    # Instrumental setup
    cfam_name = '3F'
    dpam_name = 'PRISM3'
    spam_name = 'SPEC'
    lsam_name = 'SPEC'
    fsam_name = ['OPEN', 'R1C2', 'R6C5', 'R3C1']
    fpam_name = ['OPEN', 'ND225', 'ND475']

    # Seeded random generator
    rng = np.random.default_rng(seed=0)
    # Choose (arbitrarily) which template will be the correct stellar spectrum
    slit_ref = 4
    # Loop over possible setup values (loop over FSAM and FPAM)
    for fsam in fsam_name:
        for fpam in fpam_name:
            # Create some mock data for the template with spectra
            file_path = os.path.join(test_datadir,
                    'g0v_vmag6_spc-spec_band3_unocc_CFAM3_R1C2SLIT_PRISM3_offset_array.fits')
            assert os.path.exists(file_path), f'Test FITS file not found: {file_path}'

            pri_hdr, ext_hdr = create_default_L2b_headers()[0:2]
            with fits.open(file_path) as hdul:
                psf_array = hdul[0].data
                psf_table = Table(hdul[1].data)

            assert psf_array.ndim == 3, 'Expected 3D PSF array'
            assert 'xcent' in psf_table.colnames and 'ycent' in psf_table.colnames, 'Missing centroid columns'

            # Add an initial guess of where the centroid is found
            initial_cent = {
                'xcent': np.array(psf_table['xcent']),
                'ycent': np.array(psf_table['ycent'])
            }

            # Add wavelength zero-point. In this test, we set it in a way that
            # matches one of the slices, so that we can predict which one is the
            # best image later
            ext_hdr['WV0_X'] = initial_cent['xcent'][slit_ref]
            ext_hdr['WV0_Y'] = initial_cent['ycent'][slit_ref]
        
            # Update Setup header key values
            ext_hdr['CFAMNAME'] = cfam_name
            ext_hdr['DPAMNAME'] = dpam_name
            ext_hdr['SPAMNAME'] = spam_name
            ext_hdr['LSAMNAME'] = lsam_name
            ext_hdr['FSAMNAME'] = fsam
            ext_hdr['FPAMNAME'] = fpam
            template_images = []
            data_images = []
            for i in range(psf_array.shape[0]):
                data_2d = np.copy(psf_array[i])
                err = np.zeros_like(data_2d)
                dq = np.zeros_like(data_2d, dtype=int)
                # Store template data as a DRP object
                image_template = Image(
                    data_or_filepath=data_2d,
                    pri_hdr=pri_hdr,
                    ext_hdr=ext_hdr,
                    err=err,
                    dq=dq
                )
                template_images.append(image_template)
                # Some noisy version for the simulated data without blowing it
                # unreasonably. The one with slit_ref has no additional
                # noise to test that this is the one outputted by star_spec_registration()
                # Collected data have different FSM values
                ext_hdr['FSMX'] = i // 5
                ext_hdr['FSMY'] = i - 5 * (i // 5)
                image_data = Image(
                    data_or_filepath=data_2d + rng.normal(0,
                        0.1*np.abs(i-slit_ref)*data_2d.std(), data_2d.shape),
                    pri_hdr=pri_hdr,
                    ext_hdr=ext_hdr,
                    err=err,
                    dq=dq
                )
                # Add some filename that will be used to check that the correct
                # spectrum is found
                image_data.filename = f'test_file_{i}.fits'
                data_images.append(image_data)
            dataset_template = Dataset(template_images)
            dataset_data     = Dataset(data_images)
        
            # Identify best image. Pass template and guessed centroids
            best_image = steps.star_spec_registration(
                dataset_data,
                dataset_template,
                xcent_template=initial_cent['xcent'],
                ycent_template=initial_cent['ycent'],
                slit_align_err=slit_ref)

            # Tests:
            # Test that the output corresponds with the expected best image
            assert best_image.filename == f'test_file_{slit_ref}.fits'
            assert np.all(best_image.data == dataset_data[slit_ref].data), 'Expected output data does not coincide with the input frame'
            # Check that all static header values in the primary header coincide
            # b/w I/O (all but those who depend on clock creation time)
            assert np.all([best_image.pri_hdr[key] == dataset_data[slit_ref].pri_hdr[key] 
                for key in dataset_data[slit_ref].pri_hdr]), 'Some keyword values between the expected output and input disagree'
            # Similarly for the extended header
            assert np.all([best_image.ext_hdr[key] == dataset_data[slit_ref].ext_hdr[key]
                for key in dataset_data[slit_ref].ext_hdr if key != 'DRPCTIME']), 'Some keyword values between the expected output and input disagree'
 

    # Expected failures
    pam_list = ['CFAMNAME', 'DPAMNAME', 'SPAMNAME', 'LSAMNAME', 'FSAMNAME',
        'FPAMNAME']

    for pam in pam_list:
        # Store current common value
        tmp = dataset_data[0].ext_hdr[pam]
        # Set it to some value that will disagree
        dataset_data[0].ext_hdr[pam] = ''
        with pytest.raises(ValueError):
            steps.star_spec_registration(
                dataset_data,
                dataset_template,
                xcent_template=initial_cent['xcent'],
                ycent_template=initial_cent['ycent'],
                slit_align_err=slit_ref)
        print(f'PAM failure test for {pam} passed')
        dataset_data[0].ext_hdr[pam] = tmp
    # Remove WV0_X/Y keywords
    del dataset_data[0].ext_hdr['WV0_X']
    with pytest.raises(ValueError):
        steps.star_spec_registration(
            dataset_data,
            dataset_template,
            xcent_template=initial_cent['xcent'],
            ycent_template=initial_cent['ycent'],
            slit_align_err=slit_ref)
    print('WV0_X failure test passed')
    dataset_data[0].ext_hdr['WV0_X'] = 30.
    del dataset_data[0].ext_hdr['WV0_Y']
    with pytest.raises(ValueError):
        steps.star_spec_registration(
            dataset_data,
            dataset_template,
            xcent_template=initial_cent['xcent'],
            ycent_template=initial_cent['ycent'],
            slit_align_err=slit_ref)
    print('WV0_Y failure test passed')
    dataset_data[0].ext_hdr['WV0_Y'] = 30.

    # Remove FSMX/Y keywords
    del dataset_data[0].ext_hdr['FSMX']
    with pytest.raises(AssertionError):
        steps.star_spec_registration(
            dataset_data,
            dataset_template,
            xcent_template=initial_cent['xcent'],
            ycent_template=initial_cent['ycent'],
            slit_align_err=slit_ref)
    print('FSMX failure test passed')
    dataset_data[0].ext_hdr['FSMX'] = 30.
    del dataset_data[0].ext_hdr['FSMY']
    with pytest.raises(AssertionError):
        steps.star_spec_registration(
            dataset_data,
            dataset_template,
            xcent_template=initial_cent['xcent'],
            ycent_template=initial_cent['ycent'],
            slit_align_err=slit_ref)
    print('FSMY failure test passed')
    dataset_data[0].ext_hdr['FSMY'] = 30.
    
    print('Star spectrum registration passed')
    
if __name__ == "__main__":
    #convert_tvac_to_dataset()
    test_psf_centroid()
    test_dispersion_model()
    test_read_cent_wave()
    test_calibrate_dispersion_model()
    test_determine_zeropoint()
    test_add_wavelength_map()
    test_star_spec_registration()
