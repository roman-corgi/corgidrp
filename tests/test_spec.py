import os
import numpy as np
import pytest
import logging
import warnings
from astropy.io import fits
from astropy.table import Table
from corgidrp.data import Dataset, Image, DispersionModel, LineSpread
import corgidrp.spec as steps
from corgidrp.mocks import create_default_L2b_headers, get_formatted_filename
from corgidrp.spec import get_template_dataset
import corgidrp.l3_to_l4 as l3_to_l4
from datetime import datetime, timedelta
# VAP testing
from corgidrp.check import check_filename_convention, verify_header_keywords

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
    
    # test None in x/ycent header of template file
    temp_dataset[0].ext_hdr['XCENT'] = None
    calibration_4 = steps.compute_psf_centroid(
        dataset=dataset, template_dataset = temp_dataset, filtersweep = filtersweep
    )
    assert np.all(calibration_2.xfit - calibration_4.xfit < errortol_pix)
    assert np.all(calibration_2.yfit - calibration_4.yfit < errortol_pix)
    assert np.all(calibration_4.xfit_err < errortol_pix)
    assert np.all(calibration_4.yfit_err < errortol_pix)
        
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

    # Load the filter-to-filter image offsets to correct for the location of the narrowband centroid
    # with respect to the broadband filter.
    (xoff_nb, yoff_nb) = (steps.read_cent_wave('3D')[2], steps.read_cent_wave('3D')[3])
    (xoff_bb, yoff_bb) = (steps.read_cent_wave('3')[2], steps.read_cent_wave('3')[3])

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
        assert x0 - (xoff_bb - xoff_nb) == pytest.approx(slit_x, abs = errortol_pix)
        assert y0 - (yoff_bb - yoff_nb) == pytest.approx(slit_y, abs = errortol_pix)
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
        assert x0 - (xoff_bb - xoff_nb) == pytest.approx(slit_x, abs = errortol_pix)
        assert y0 - (yoff_bb - yoff_nb) == pytest.approx(slit_y, abs = errortol_pix)
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

    # The tests are of two types:
    # 1/ UTs with mock data showing that the step function finds the expected best
    # match spectrum among all present ones
    # 2/ UTs showing that if the input parameters are invalid, the step function
    # raises an exception
    # 3/ VAP tests are performed along this test function too (https://github.com/roman-corgi/corgidrp/issues/545)

    # Directory to temporarily store the I/O of the test
    dir_test = os.path.join(os.path.dirname(__file__), 'simdata')
    os.makedirs(dir_test, exist_ok=True)

    log_file = os.path.join(dir_test, 'star_spec_registration.log')

    # Create a new logger specifically for this test, otherwise things have issues
    logger = logging.getLogger('star_spec_registration')
    logger.setLevel(logging.INFO)

    # Clear any existing handlers to avoid duplicates
    logger.handlers.clear()

    # Create file handler
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)

    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info('='*80)
    logger.info('CGI-REQT-5465: REGISTERED PRISM IMAGE OF STAR TEST')
    logger.info('='*80)
    logger.info("")
    logger.info('='*80)
    logger.info('Test Case 1: Input Image Data Format and Content')
    logger.info('='*80)

    # Instrumental setup
    cfam_name = '3F'
    dpam_name = 'PRISM3'
    spam_name = 'SPEC'
    lsam_name = 'SPEC'
    # Test all possible supported mode configurations
    fsam_name = ['OPEN', 'R1C2', 'R6C5', 'R3C1']
    fpam_name = ['OPEN', 'ND225', 'ND475']
    print(f'Considering CFAM={cfam_name:s}, DPAM={dpam_name:s}, ' +
        f'SPAM={spam_name}, LSAM={lsam_name:s}')

    # Data level of input data
    dt_lvl = 'L2b'

    # Create some L2b mock data: The step function must find the spectrum that
    # best matches one of the template spectra. There's a step in the step
    # function that derives a shift between the L2b data and the templates.
    # Hence, we create a mock L2b Dataset with a similar shape between template
    # and L2b data
    pri_hdr, ext_hdr = create_default_L2b_headers()[0:2]
    # Number of templates available
    n_temp = 5
    # Array of data to be used to generate L2b later on adding some noise
    psf_arr = []
    # y offsets (from FSAM slit vertical offset)
    yoffset_arr = []
    pathfiles_template = []
    # =================================================================
    # VAP Testing: Validate Input Images
    # =================================================================
    # Validate all input images
    logger.info('='*80)
    logger.info('Test Case 1: Input Image Data Format and Content')
    logger.info('='*80)
    logger.info('Template data')
    for idx_temp in range(n_temp):
        pathfile = os.path.join(test_datadir,
                f'spec_reg_fsm_offset_template_cfam3F_{idx_temp:02d}.fits')
        # Make sure the template exists before continuing
        assert os.path.exists(pathfile), f'Test FITS file not found: {pathfile}'
        with fits.open(pathfile) as hdul:
            # Get template data to create a noisy sim with different FSM positions later on
            psf_arr += [hdul[0].data]
            assert psf_arr[-1].ndim == 2, 'Expected 2D PSF array'
            # Make sure FSAM offset is present and record them
            try:
                yoffset_arr += [hdul[0].header['FSM_OFF']]
            except:
                logger.info(f'Alignment offsets relative to FSAM slit NOT present in template file {pathfile}. FAIL')
                raise ValueError(f'Missing FSM offset in file {pathfile:s}')
            # Make sure zero-points are present            
            try:
                wv0_x = hdul[0].header['WV0_X']
                wv0_y = hdul[0].header['WV0_Y']
            except:
                logger.info(f'Wavelength zero-point WV0_X, WV0_Y NOT present in template file {pathfile}. FAIL')
      
        # Add pathfilename to the list
        pathfiles_template += [pathfile]

    # At this step all individual tests above have passed
    logger.info('Alignment offsets relative to FSAM slit present in all template files. PASS')
    logger.info('WV0_X and WV0_Y present in all template files. PASS')

    # Define a slit alignment offset for the FSM data that is close to one of the
    # templates to be able to predict which offset best matches the templates
    slit_ref = n_temp // 2
    # Slight change
    slit_align_err = (np.array(yoffset_arr)
        + np.diff(np.array(yoffset_arr)).mean()*0.1)[slit_ref]

    # Start UTs showing that the step function works as expected
    # Some (arbitrary) number of frames per FSM position
    nframes = 3
    # Seeded random generator
    rng = np.random.default_rng(seed=0)
    # Loop over possible spectroscopy setup values (loop over FSAM and FPAM)
    for fsam in fsam_name:
        for fpam in fpam_name:
            # Update Setup header key values
            ext_hdr['CFAMNAME'] = cfam_name
            ext_hdr['DPAMNAME'] = dpam_name
            ext_hdr['SPAMNAME'] = spam_name
            ext_hdr['LSAMNAME'] = lsam_name
            ext_hdr['FSAMNAME'] = fsam
            ext_hdr['FPAMNAME'] = fpam
            data_images = []
            basetime = datetime.now()
            # Random inserts to test the cross-correlation functionality
            x0, y0 = 512+np.random.randint(300), 512+np.random.randint(300)
            for i in range(len(psf_arr)):
                data_l2b = np.zeros([1024, 1024])
                psf_tmp = np.copy(psf_arr[i])
                data_l2b[y0-psf_tmp.shape[0]//2:y0+psf_tmp.shape[0]//2 + 1,
                    x0-psf_tmp.shape[1]//2:x0+psf_tmp.shape[1]//2 + 1] = psf_tmp
                err = np.zeros_like(data_l2b)
                dq = np.zeros_like(data_l2b, dtype=int)
                ext_hdr_cp = ext_hdr.copy()
                # Only vertical FSM positions, along the narrower length of the
                # slit, need be explored. Values are irrelevant.
                ext_hdr_cp['FSMX'] = 0
                ext_hdr_cp['FSMY'] = i - 5 * (i // 5)
                # Produce NFRAMES for each FSM position:
                # Some noisy version for the simulated data without blowing it
                # unreasonably. The one with slit_ref has much less noise added
                for i_frame in range(nframes):
                    image_data = Image(
                        data_or_filepath=data_l2b + rng.normal(0,
                            np.abs(i-slit_ref+0.01)*data_l2b.std(),
                            data_l2b.shape),
                        pri_hdr=pri_hdr,
                        ext_hdr=ext_hdr_cp,
                        err=err,
                        dq=dq
                    )
                    # Append L2b filename 
                    image_data.filename = get_formatted_filename(
                        basetime + timedelta(seconds=nframes*i+i_frame),
                        '0000000000000000000')
                    data_images.append(image_data)

            dataset_fsm = Dataset(data_images)

            logger.info('FSM data')
            logger.info(f'SUBCASE FSAM={fsam:s}, FPAM={fpam:s}')
            for i, frame in enumerate(dataset_fsm):
                frame_info = f"Frame {i}"
                verify_header_keywords(frame.ext_hdr, {'DATALVL': dt_lvl},
                    frame_info, logger)
                check_filename_convention(getattr(frame, 'filename', None),
                    f'cgi_*_{dt_lvl.lower():s}.fits', frame_info, logger)
                verify_header_keywords(frame.ext_hdr, ['CFAMNAME'], frame_info,
                    logger)
                logger.info("")

            logger.info(f"Total input images validated: {len(dataset_fsm)}")
            logger.info("")
        
            # Identify best image
            list_of_best_fsm = steps.star_spec_registration(
                dataset_fsm,
                pathfiles_template,
                slit_align_err=slit_align_err)

            # Collect all input files
            fsm_filenames = []
            for image in dataset_fsm:
                fsm_filenames += [image.filename]
            # The best FSM position in the test is the one in these files
            list_of_expected_fsm = fsm_filenames[nframes*slit_ref:nframes*(slit_ref+1)]
            # Check they are the same set (not necessarily in the same order)
            assert len(list_of_expected_fsm) == len(list_of_best_fsm), 'List of FSM frames does not match expected set'
            # Save files (temporarily) to check data level
            dataset_fsm.save(filedir=dir_test) 

            # VAP testing: Check data level of best FSM files only
            # Test that the output corresponds with the expected best FSM position
            logger.info('='*80)
            logger.info('Test Case 2: Output Calibration Product Data Format and Content')
            logger.info('='*80)
            for i, frame in enumerate(dataset_fsm):
                if frame.filename in list_of_best_fsm:
                    frame_info = f"Frame {i}"
                    verify_header_keywords(frame.ext_hdr, {'DATALVL': dt_lvl},
                        frame_info, logger)
                    check_filename_convention(getattr(frame, 'filename', None),
                        f'cgi_*_{dt_lvl.lower():s}.fits', frame_info, logger)
                    logger.info("")

            logger.info(f"Total input images validated: {len(list_of_best_fsm)}")
            logger.info("")

            for file in list_of_best_fsm:
                # Verify all files are in the set of expected files in the test
                assert file in list_of_expected_fsm, f'File {file:s} is not in the list of expected best FSM frames'

            logger.info('='*80)
            logger.info('Test Case 3: Baseline Performance Checks')
            logger.info('Best-matching filenames')
            for file in list_of_best_fsm:
                logger.info(f'{file}')
            logger.info('='*80)

            # Delete temporary files
            for file in dataset_fsm:
                os.remove(file.filepath)

    # Make sure all temporary files are removed except the logger
    for file in os.listdir(dir_test):
        if file[-3:] != 'log':
            os.remove(os.path.join(dir_test, file))

    # End of tests testing proper functioning.

    # Expected failures
    # Wrong PAM setting
    pam_list = ['CFAMNAME', 'DPAMNAME', 'SPAMNAME', 'LSAMNAME', 'FSAMNAME',
        'FPAMNAME']

    for pam in pam_list:
        # Store current, common value to all images
        tmp = dataset_fsm[0].ext_hdr[pam]
        # Set PAM to some value that will disagree with the other images
        dataset_fsm[0].ext_hdr[pam] = ''
        with pytest.raises(ValueError):
            steps.star_spec_registration(
                dataset_fsm,
                pathfiles_template,
                slit_align_err=slit_align_err)
        print(f'PAM failure test for {pam} passed')
        # Restore original value, before moving to the next failure test
        dataset_fsm[0].ext_hdr[pam] = tmp

    # Remove FSMX/Y keywords from the observation data
    del dataset_fsm[0].ext_hdr['FSMX']
    with pytest.raises(AssertionError):
        steps.star_spec_registration(
            dataset_fsm,
            pathfiles_template,
            slit_align_err=slit_align_err)
    print('FSMX failure test passed')
    dataset_fsm[0].ext_hdr['FSMX'] = 30.
    del dataset_fsm[0].ext_hdr['FSMY']
    with pytest.raises(AssertionError):
        steps.star_spec_registration(
            dataset_fsm,
            pathfiles_template,
            slit_align_err=slit_align_err)
    print('FSMY failure test passed')
    dataset_fsm[0].ext_hdr['FSMY'] = 30.
    
    logger.info('='*80)
    logger.info('TEST COMPLETE')
    logger.info('='*80)
    
def test_linespread_function():
    """
    test the fit of a linespread function to a narrowband observation and storing in a LineSpread calibration file
    using the output_dataset of the test of the wavelength map
    """
    line_spread = steps.fit_line_spread_function(output_dataset)
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
        frame.data[10, 10] = np.nan 
    line_spread_bad = steps.fit_line_spread_function(bad_dataset)
    assert line_spread_bad.fwhm == pytest.approx(line_spread.fwhm, rel = 0.1)
    assert line_spread_bad.amplitude == pytest.approx(line_spread.amplitude, rel = 0.1)

def test_extract_spec():
    """
    test the l3_to_l4.extract_spec() function, that extracts the 1D spectrum 
    with corresponding wavelengths, error, and dq.
    """
    spec_dataset = l3_to_l4.extract_spec(output_dataset)
    image = spec_dataset[0]
    #halfwidth = 9, size: 2 * 9 + 1
    assert np.shape(image.data) == (19,)
    assert np.shape(image.dq) == (19,)
    assert np.shape(image.err) == (1,19)
    assert np.shape(image.hdu_list["WAVE"]) == (19,)
    assert np.shape(image.hdu_list["WAVE_ERR"]) == (19,)
    
    err_im = output_dataset[0].copy()
    #equal error for all pixels => equal weights, should not change the sum
    err_im.err[0,:,:] = 3.
    input_dataset = Dataset([err_im])
    err_ext = l3_to_l4.extract_spec(input_dataset, apply_weights = True)
    out_im = err_ext[0]
    assert np.allclose(out_im.data, image.data)
    #estimate the resulting value at the position of the maximum
    ind_x = np.argmax(err_im.data[:, 40])
    assert np.sum(err_im.data[ind_x, 38:43]) == pytest.approx(np.max(out_im.data))
    
    #estimate error as Poisson like
    err_im.err[0,:,:] = np.sqrt(err_im.data)
    dataset = Dataset([err_im])
    err_ext = l3_to_l4.extract_spec(dataset, apply_weights = True)
    out_im = err_ext[0]
    #estimate the resulting value at the position of the maximum, 
    #there might be a great different in the maxima due to the rough weighting with signal noise
    ind_x = np.argmax(err_im.data[:, 40])
    assert np.sum(err_im.data[ind_x, 38:43]) == pytest.approx(np.max(out_im.data), rel = 2)
    max_ind = np.argmax(out_im.data)
    assert max_ind == np.argmax(image.data)
    err_wht = err_im.err[0]
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        whts = 1./np.square(err_wht)
    err_expect = 1./np.sqrt(np.nansum(whts[:, 38:43], axis = 1))
    assert np.max(err_expect) == np.max(out_im.err)
    assert "extraction" and "half width" and "weights" in str(out_im.ext_hdr["HISTORY"])

if __name__ == "__main__":
    #convert_tvac_to_dataset()
    test_psf_centroid()
    test_dispersion_model()
    test_read_cent_wave()
    test_calibrate_dispersion_model()
    test_determine_zeropoint()
    test_add_wavelength_map()
    test_star_spec_registration()
    test_linespread_function()
    test_extract_spec()
