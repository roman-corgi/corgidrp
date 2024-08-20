import os  
import glob  
import argparse
import numpy as np
import astropy.time as time
from astropy.io import fits  
import matplotlib.pyplot as plt

import corgidrp
import corgidrp.data as data
import corgidrp.mocks as mocks
import corgidrp.walker as walker

corgidrp_dir = os.path.join(os.path.dirname(corgidrp.__file__), '..') # basedir of entire corgidrp github repo

nonlin_tvac = os.path.join(corgidrp_dir, '../e2e_tests_corgidrp/nonlin_table_240322.txt')
nonlin_l1_datadir = os.path.join(corgidrp_dir, '../e2e_tests_corgidrp/')
output_dir = './l1_to_l2a_output/'

def set_obstype_for_tvac(
    list_of_fits,
    ):
    """ Adds proper values to OBSTYPE for the non-linearity calibration: NONLIN,
        (data used to calibrate the non-linearity) or MNFRAME (data used to build
        a mean frame).

        This function is unnecessary with future data because data will have
        the proper values in OBSTYPE. The TVAC data used must be the
        following 382 files with IDs: 51841-51870 (30: mean frame). And NL:
        51731-51840 (110), 51941-51984 (44), 51986-52051 (66), 55122-55187 (66),
        55191-55256 (66)  

        Args:
  
        list_of_fits (list): list of FITS files that need to be updated.

        Returns:

        FITS files with updated value of OBSTYPE.

    """
    # Folder with files
    nonlin_dir = list_of_fits[0][0:len(list_of_fits[0]) - list_of_fits[0][::-1].find('/')]
    # TVAC files                                                                    
    tvac_file_0 = [
        'CGI_EXCAM_L1_0000051841.fits',                                                 
        'CGI_EXCAM_L1_0000051731.fits',                                                 
        'CGI_EXCAM_L1_0000051941.fits',                                                 
        'CGI_EXCAM_L1_0000051986.fits',                                                 
        'CGI_EXCAM_L1_0000055122.fits',                                                 
        'CGI_EXCAM_L1_0000055191.fits',
        ]                                                 

    n_files = [30, 110, 44, 66, 66, 66]
    if len(tvac_file_0) != len(n_files):
        raise Exception('Inconsistent number of files and stacks')

    for i_group, file in enumerate(tvac_file_0):
        l1_number = int(file[file.find('L1_')+3:file.find('L1_')+13])
        print(f'Group of {n_files[i_group]} files starting with {file}')
        for i_file in range(n_files[i_group]):
            file_name = f'CGI_EXCAM_L1_00000{l1_number+i_file}.fits'
            # Additional check
            if np.any([nonlin_dir+file_name == file for file in list_of_fits]) is False:
                raise Exception(f'The file {nonlin_dir+file} is not part of the calibration data')
            fits_file = fits.open(nonlin_dir+file_name)
            prihdr = fits_file[0].header 
            exthdr = fits_file[1].header
            # Adjust OBSTYPE
            if n_files[i_group] == 30:
                prihdr['OBSTYPE'] = 'MNFRAME'
            else:
                prihdr['OBSTYPE'] = 'NONLIN'
            # Update FITS file    
            fits_file.writeto(nonlin_dir+file_name, overwrite=True)

def get_first_nonlin_file(
    list_of_fits,
    ):
    """ Returns the first FITS file with the NONLIN value on OBSTYPE in a list
        of FITS files.

        Remember that FITS files used for NL calibration must have DATETIME in
        ascending order.

        Args:                                                              
                                                                                
        list_of_fits (list): list of FITS files that need to be updated.        
                                                                                
        Returns:                                                                
                                                                                
        First FITS file with OBSTYPE set to NONLIN.

    """
    first_fits_file = 'NONLIN not found'
    for file in list_of_fits:
        fits_file = fits.open(file)                         
        if fits_file[0].header['OBSTYPE'] == 'NONLIN':
            first_fits_file = fits_file.filename()
            break
    return first_fits_file

def set_obstype_for_tvac(
    list_of_fits,
    ):
    """ Adds proper values to OBSTYPE for the non-linearity calibration: NONLIN,
        (data used to calibrate the non-linearity) or MNFRAME (data used to build
        a mean frame).

        This function is unnecessary with future data because data will have
        the proper values in OBSTYPE. The TVAC data used must be the
        following 382 files with IDs: 51841-51870 (30: mean frame). And NL:
        51731-51840 (110), 51941-51984 (44), 51986-52051 (66), 55122-55187 (66),
        55191-55256 (66)  

        Args:
  
        list_of_fits (list): list of FITS files that need to be updated.

        Returns:

        FITS files with updated value of OBSTYPE.

    """
    # Folder with files
    nonlin_dir = list_of_fits[0][0:len(list_of_fits[0]) - list_of_fits[0][::-1].find('/')]
    # TVAC files                                                                    
    tvac_file_0 = [
        'CGI_EXCAM_L1_0000051841.fits',                                                 
        'CGI_EXCAM_L1_0000051731.fits',                                                 
        'CGI_EXCAM_L1_0000051941.fits',                                                 
        'CGI_EXCAM_L1_0000051986.fits',                                                 
        'CGI_EXCAM_L1_0000055122.fits',                                                 
        'CGI_EXCAM_L1_0000055191.fits',
        ]                                                 

    n_files = [30, 110, 44, 66, 66, 66]
    if len(tvac_file_0) != len(n_files):
        raise Exception('Inconsistent number of files and stacks')

    for i_group, file in enumerate(tvac_file_0):
        l1_number = int(file[file.find('L1_')+3:file.find('L1_')+13])
        print(f'Group of {n_files[i_group]} files starting with {file}')
        for i_file in range(n_files[i_group]):
            file_name = f'CGI_EXCAM_L1_00000{l1_number+i_file}.fits'
            # Additional check
            if np.any([nonlin_dir+file_name == file for file in list_of_fits]) is False:
                raise Exception(f'The file {nonlin_dir+file} is not part of the calibration data')
            fits_file = fits.open(nonlin_dir+file_name)
            prihdr = fits_file[0].header 
            exthdr = fits_file[1].header
            # Adjust OBSTYPE
            if n_files[i_group] == 30:
                prihdr['OBSTYPE'] = 'MNFRAME'
            else:
                prihdr['OBSTYPE'] = 'NONLIN'
            # Update FITS file    
            fits_file.writeto(nonlin_dir+file_name, overwrite=True)

def get_first_nonlin_file(
    list_of_fits,
    ):
    """ Returns the first FITS file with the NONLIN value on OBSTYPE in a list
        of FITS files.

        Remember that FITS files used for NL calibration must have DATETIME in
        ascending order.

        Args:                                                              
                                                                                
        list_of_fits (list): list of FITS files that need to be updated.        
                                                                                
        Returns:                                                                
                                                                                
        First FITS file with OBSTYPE set to NONLIN.

    """
    first_fits_file = 'NONLIN not found'
    for file in list_of_fits:
        fits_file = fits.open(file)                         
        if fits_file[0].header['OBSTYPE'] == 'NONLIN':
            first_fits_file = fits_file.filename()
            break
    return first_fits_file

def set_obstype_for_tvac(
    list_of_fits,
    ):
    """ Adds proper values to OBSTYPE for the non-linearity calibration: NONLIN,
        (data used to calibrate the non-linearity) or MNFRAME (data used to build
        a mean frame).

        This function is unnecessary with future data because data will have
        the proper values in OBSTYPE. The TVAC data used must be the
        following 382 files with IDs: 51841-51870 (30: mean frame). And NL:
        51731-51840 (110), 51941-51984 (44), 51986-52051 (66), 55122-55187 (66),
        55191-55256 (66)  

        Args:
  
        list_of_fits (list): list of FITS files that need to be updated.

        Returns:

        FITS files with updated value of OBSTYPE.

    """
    # Folder with files
    nonlin_dir = list_of_fits[0][0:len(list_of_fits[0]) - list_of_fits[0][::-1].find('/')]
    # TVAC files                                                                    
    tvac_file_0 = [
        'CGI_EXCAM_L1_0000051841.fits',                                                 
        'CGI_EXCAM_L1_0000051731.fits',                                                 
        'CGI_EXCAM_L1_0000051941.fits',                                                 
        'CGI_EXCAM_L1_0000051986.fits',                                                 
        'CGI_EXCAM_L1_0000055122.fits',                                                 
        'CGI_EXCAM_L1_0000055191.fits',
        ]                                                 

    n_files = [30, 110, 44, 66, 66, 66]
    if len(tvac_file_0) != len(n_files):
        raise Exception('Inconsistent number of files and stacks')

    for i_group, file in enumerate(tvac_file_0):
        l1_number = int(file[file.find('L1_')+3:file.find('L1_')+13])
        print(f'Group of {n_files[i_group]} files starting with {file}')
        for i_file in range(n_files[i_group]):
            file_name = f'CGI_EXCAM_L1_00000{l1_number+i_file}.fits'
            # Additional check
            if np.any([nonlin_dir+file_name == file for file in list_of_fits]) is False:
                raise Exception(f'The file {nonlin_dir+file} is not part of the calibration data')
            fits_file = fits.open(nonlin_dir+file_name)
            prihdr = fits_file[0].header 
            exthdr = fits_file[1].header
            # Adjust OBSTYPE
            if n_files[i_group] == 30:
                prihdr['OBSTYPE'] = 'MNFRAME'
            else:
                prihdr['OBSTYPE'] = 'NONLIN'
            # Update FITS file    
            fits_file.writeto(nonlin_dir+file_name, overwrite=True)

def get_first_nonlin_file(
    list_of_fits,
    ):
    """ Returns the first FITS file with the NONLIN value on OBSTYPE in a list
        of FITS files.

        Remember that FITS files used for NL calibration must have DATETIME in
        ascending order.

        Args:                                                              
                                                                                
        list_of_fits (list): list of FITS files that need to be updated.        
                                                                                
        Returns:                                                                
                                                                                
        First FITS file with OBSTYPE set to NONLIN.

    """
    first_fits_file = 'NONLIN not found'
    for file in list_of_fits:
        fits_file = fits.open(file)                         
        if fits_file[0].header['OBSTYPE'] == 'NONLIN':
            first_fits_file = fits_file.filename()
            break
    return first_fits_file

def main():
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    # Define the raw science data to process
    nonlin_l1_list = glob.glob(os.path.join(nonlin_l1_datadir, "*.fits"))


    #TODO:
    # The data used to generate a mean frame can be taken at any time. Either there's
    # a directive about when to take them: first/last set or the CPGS file has
    # a tag that identifies which kind of calibration is being used similar to
    # OBSTYPE and set to NONLIN, meaning that all the collected data are meant to
    # be part of the non-linearity calibration.
    # For now, since the frames used to generate the mean frame were the first
    # ones, and walker opens the first file to identify the type of recipe, the
    # result may be confused with Kgain calibration. Thus, until there's a decision
    # on the above, I reverse the order of the files, which does not create any
    # issues in the calibration functions (they re-order inoput files themselves)
    nonlin_l1_list.sort()#reverse=True)   
    
    # Set TVAC OBSTYPE to MNFRAME/NONLIN (flight data should have these values)
    set_obstype_for_tvac(nonlin_l1_list)

    first_nonlin_file = get_first_nonlin_file(nonlin_l1_list)

    # Non-linearity calibration file used to compare the output from CORGIDRP:
    # We are going to make a new nonlinear calibration file using
    # a combination of the II&T nonlinearty file and the mock headers from
    # our unit test version of the NonLinearityCalibration
    nonlin_dat = np.genfromtxt(output_dir+'nonlin_table_240322.txt', delimiter=",")
    pri_hdr, ext_hdr = mocks.create_default_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] =  corgidrp.__version__
    mock_input_dataset = data.Dataset(nonlin_l1_list)
    nonlinear_cal = data.NonLinearityCalibration(nonlin_dat,
                                                 pri_hdr=pri_hdr,
                                                 ext_hdr=ext_hdr,
                                                 input_dataset=mock_input_dataset)
    nonlinear_cal.save(filedir=output_dir, filename="nonlin_tvac.fits" )

    # Run the walker on some test_data
    print('Running walker')
    walker.walk_corgidrp(nonlin_l1_list, '', output_dir)

    # Compare results
<<<<<<< HEAD
    print('Comparing the results with TVAC')
    ## NL from CORGIDRP
    nonlin_out_filename = first_nonlin_file[len(first_nonlin_file)-first_nonlin_file[::-1].find('/'):]
    if nonlin_out_filename.find('fits') == -1:
        raise Exception('Data files must be FITS files')
    nonlin_out_filename = nonlin_out_filename[0:nonlin_out_filename.find('fits')-1]
    nonlin_out_filename += '_NonLinearityCalibration.fits'
    nonlin_out = fits.open(output_dir+nonlin_out_filename)
    if nonlin_out[0].header['OBSTYPE'] != 'NONLIN':
        raise Exception('Calibration type is not NL')
    nonlin_out_table = nonlin_out[1].data
    
    ## NL from TVAC
    nonlin_tvac = fits.open(output_dir+'nonlin_tvac.fits')
    nonlin_tvac_table = nonlin_tvac[1].data

    # Check
    if nonlin_out_table.shape[0] != nonlin_tvac_table.shape[0] or nonlin_out_table.shape[1] != nonlin_tvac_table.shape[1]:
        raise Exception('Non-linearity table from CORGI DRP has a different format than the one from TVAC')   
    
    rel_out_tvac_perc = 100*(nonlin_out_table[1:]/nonlin_tvac_table[1:]-1)
    plt.figure(figsize=(10,6))
    em_list = nonlin_out_table[0,1:]
    for i_em, em in enumerate(em_list):
        plt.plot(nonlin_out_table[1:,0], rel_out_tvac_perc[:,i_em], label=f'EM={em:.1f}')
    plt.xlabel('DN value', fontsize=16)
    plt.ylabel('Percentage (%)', fontsize=16)
    plt.title('Relative difference for NL coefficient for a given DN and EM value', fontsize=16)
    plt.legend()
    plt.savefig(output_dir+nonlin_out_filename[:-5])
    print(f'Figure saved: {output_dir+nonlin_out_filename[:-5]}.png')
=======
    ## NL from CORGIDRP
    CGI_EXCAM_L1_0000051731_NonLinearityCalibration
    breakpoint()
    ## NL from TVAC
>>>>>>> aa805fb (walker working)

if __name__ == "__main__":
    # Use arguments to run the test. Users can then write their own scripts
    # that call this script with the correct arguments and they do not need
    # to edit the file. The arguments use the variables in this file as their
    # defaults allowing the use to edit the file if that is their preferred
    # workflow.
    ap = argparse.ArgumentParser(description="run the l1->l2a end-to-end test")
    ap.add_argument("-np", "--nonlin_tvac", default=nonlin_tvac,
                    help="text file containing the non-linear table from TVAC [%(default)s]")
    ap.add_argument("-l1", "--nonlin_l1_datadir", default=nonlin_l1_datadir,
                    help="directory that contains the L1 data files used for nonlinearity calibration [%(default)s]")
    ap.add_argument("-o", "--output_dir", default=output_dir,
                    help="directory to write results and it will be created if it does not exist [%(default)s]")
    args = ap.parse_args()
    nonlin_path = args.nonlin_tvac
    l1_datadir = args.nonlin_l1_datadir
    output_dir = args.output_dir
    main()
