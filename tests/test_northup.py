import os,math
from corgidrp import data, mocks, astrom
from corgidrp.l3_to_l4 import northup
from corgidrp.l2b_to_l3 import create_wcs
from astropy.wcs import WCS
from matplotlib import pyplot as plt
import numpy as np

def test_northup(save_mock_dataset=False,save_derot_dataset=False,save_comp_figure=False,test_offset=False):
    """
    unit test of the northup function

    Args:
        save_mock_dataset (optional): if you want to save the original mock files at the input directory, turn True
        save_derot_dataset (optional): if you want to save the derotated files at the input directory, turn True
        save_comp_figure (optional): if you want to save a comparison figure of the original mock data and the derotated data

    """

    # read mock file
    dirname = 'test_data/'
    filename = 'JWST_CALFIELD2020.csv'

    fieldpath = os.path.join(os.path.dirname(__file__),dirname,filename)
    if not fieldpath:
        raise FileNotFoundError(f"No filed data {filename} found")

    # running northup function
    ang_list = [0,45,90,135,180,270]
    north_angle = 30
    updated_datalist = []

    # make a mock dataset
    mock_dataset_ori = mocks.create_astrom_data(fieldpath, rotation=north_angle)
    # run the boresight calibration to get an AstrometricCalibration file
    astrom_cal = astrom.boresight_calibration(mock_dataset_ori, fieldpath, find_threshold=10)

    for ang in ang_list:
       mock_dataset =  mock_dataset_ori.copy()

       # add an angle offset
       mock_dataset[0].pri_hdr['ROLL']=(ang,'roll angle (deg)')

       # create the wcs
       updated_dataset = create_wcs(mock_dataset,astrom_cal)
       # inject fake sources for test
       updated_dataset[0].data[340:360,340:360]=5
       updated_dataset[0].ext_hdr['X_1VAL']=350
       updated_dataset[0].ext_hdr['Y_1VAL']=350
       updated_dataset[0].dq[340:360,340:360]=1
       if save_mock_dataset:
          outdir = os.path.join('./',dirname)
          os.makedirs(outdir, exist_ok=True)
          updated_dataset[0].save(filedir='./',filename=f'mock_offset{ang+north_angle}deg.fits')
       updated_datalist.append(updated_dataset[0])

    input_dataset = data.Dataset(updated_datalist)
    derot_dataset = northup(input_dataset)

    # save fits file
    if save_derot_dataset:
       outdir = os.path.join('./',dirname)
       os.makedirs(outdir, exist_ok=True)
       for derot_data in derot_dataset:
          derot_data.save(filedir='./',filename=f'derot_offset{ang+north_angle}deg.fits')

    for i, (input_data,derot_data) in enumerate(zip(input_dataset,derot_dataset)):
        # read the original mock file and derotated file
        sci_input = input_data.data
        sci_derot = derot_data.data
        dq_input = input_data.dq
        dq_derot = derot_data.dq

        sci_hd = input_data.ext_hdr
	# rotate around the center of the image
        ylen, xlen = sci_input.shape
        xcen, ycen = xlen/2, ylen/2 

        # check the angle offset
        astr_hdr = WCS(sci_hd)
        angle_offset = np.rad2deg(-np.arctan2(-astr_hdr.wcs.cd[0,1], astr_hdr.wcs.cd[1,1]))

        # the location for test
        x_value1 = input_dataset[0].ext_hdr['X_1VAL']
        y_value1 = input_dataset[0].ext_hdr['Y_1VAL']
        r = np.sqrt((x_value1-xcen)**2+(y_value1-ycen)**2)
        theta = np.rad2deg(np.arctan2(y_value1-ycen,x_value1-xcen))
        if theta<0:
           theta+=360
        
        # the location for test in the derotated images
        x_test = round(xcen + r*np.cos(np.deg2rad(angle_offset+theta)))
        y_test = round(ycen + r*np.sin(np.deg2rad(angle_offset+theta)))

        # check if rotation works properly
        assert(sci_input[y_value1,x_value1] != sci_derot[y_value1,x_value1])
        assert(dq_input[y_value1,x_value1] != dq_derot[y_value1,x_value1])

        assert(math.isclose(sci_derot[y_test,x_test],sci_input[y_value1,x_value1],rel_tol=0.01))
        assert(dq_input[y_value1,x_value1] == dq_derot[y_test,x_test])

        # check if the derotated DQ frame has no non-integer values (except NaN)
        non_integer_mask = (~np.isnan(dq_derot)) & (dq_derot % 1 != 0)
        non_integer_indices = np.argwhere(non_integer_mask)
        assert(len(non_integer_indices) == 0)

        # check if the north vector really faces up
        astr_hdr_new = WCS(derot_data.ext_hdr)
        north_pa = -np.rad2deg(np.arctan2(-astr_hdr_new.wcs.cd[0,1], astr_hdr_new.wcs.cd[1,1]))
        assert(math.isclose(north_pa,0,abs_tol=1e-3))

        # (optional) save comparison figure
        ang = ang_list[i]
        if save_comp_figure:
          center_x, center_y = 850,850 # location of the compass

          fig, (ax0, ax1) = plt.subplots(nrows=1, ncols=2, sharex=True, figsize=(8,5))

          ax0.set_title('Original Mock Data')
          ax0.imshow(sci_input,origin='lower',vmin=0,vmax=5)

          north_vector = np.array([astr_hdr.wcs.cd[0,1], astr_hdr.wcs.cd[1,1]])  # Points toward increasing Dec
          east_vector = np.array([astr_hdr.wcs.cd[0,0], astr_hdr.wcs.cd[1,0]])   # Points toward increasing RA
          # Normalize vectors for consistent display length
          north_vector /= np.linalg.norm(north_vector)
          east_vector /= np.linalg.norm(east_vector)

          ax0.arrow(center_x, center_y, 30 * north_vector[0], 30 * north_vector[1], head_width=5, head_length=10, fc='w', ec='w')
          ax0.text(center_x+50 * north_vector[0], center_y+50 * north_vector[1], 'N', c='w',fontsize=12)
          ax0.arrow(center_x, center_y, 30 * east_vector[0], 30 * east_vector[1], head_width=5, head_length=10, fc='w', ec='w')
          ax0.text(center_x+75 * east_vector[0], center_y+75 * east_vector[1], 'E', c='w',fontsize=12)

          ax1.set_title(f'Derotated Data\n by {ang+north_angle}deg counterclockwise')
          ax1.imshow(sci_derot,origin='lower',vmin=0,vmax=5)

          new_east_vector = np.array([astr_hdr_new.wcs.cd[0,0], astr_hdr_new.wcs.cd[1,0]])   # Points toward increasing RA
          new_north_vector = np.array([astr_hdr_new.wcs.cd[0,1], astr_hdr_new.wcs.cd[1,1]])

          # Normalize vectors for consistent display length
          new_east_vector /= np.linalg.norm(new_east_vector)
          new_north_vector /= np.linalg.norm(new_north_vector)

          # add N/E
          ax1.arrow(center_x, center_y, 30 * new_north_vector[0], 30 * new_north_vector[1], head_width=5, head_length=10, fc='w', ec='w')
          ax1.text(center_x+50 * new_north_vector[0], center_y+50 * new_north_vector[1], 'N', c='w',fontsize=12)
          ax1.arrow(center_x, center_y, 30 * new_east_vector[0], 30 * new_east_vector[1], head_width=5, head_length=10, fc='w', ec='w')
          ax1.text(center_x+75 * new_east_vector[0], center_y+75 * new_east_vector[1], 'E', c='w',fontsize=12)

          outdir = os.path.join('./',dirname)
          os.makedirs(outdir, exist_ok=True)
          outfilename = f'compare_northup_offset{ang+north_angle}deg.png'

          plt.savefig(os.path.join(outdir,outfilename))

          print(f"Comparison figure saved at {dirname+outfilename}")
          plt.close(fig)

    return

def test_wcs_and_offset(save_mock_dataset=False):
   """
   unit test of the create_wcs function and offset keyword 

    Args:
        save_mock_dataset (optional): if you want to save the original mock files at the input directory, turn True
        
   """
   # read mock file
   dirname = 'test_data/'
   filename = 'JWST_CALFIELD2020.csv'

   fieldpath = os.path.join(os.path.dirname(__file__),dirname,filename)
   if not fieldpath:
      raise FileNotFoundError(f"No filed data {filename} found")
   
   # running northup function
   ang = 0
   north_angle = 30
   updated_datalist = []

   # make a mock dataset
   mock_dataset_ori = mocks.create_astrom_data(fieldpath, rotation=north_angle)
   # run the boresight calibration to get an AstrometricCalibration file
   astrom_cal = astrom.boresight_calibration(mock_dataset_ori, fieldpath, find_threshold=10)

   mock_dataset =  mock_dataset_ori.copy()

   # add an angle offset
   mock_dataset[0].pri_hdr['ROLL']=(ang,'roll angle (deg)')

   # create the wcs
   test_offset = (3.3, 1.0)
   updated_dataset = create_wcs(mock_dataset,astrom_cal,offset=test_offset)

   im_data = updated_dataset[0].data
   image_y, image_x = im_data.shape
   center_pixel = [(image_x-1) // 2, (image_y-1) // 2]
   # ensure offset worked, test it doesn't equal previous center, and that it does equal center + offset
   assert updated_dataset[0].ext_hdr['CRPIX1'] != center_pixel[0] 
   assert updated_dataset[0].ext_hdr['CRPIX2'] == center_pixel[1]+test_offset[1]

   if save_mock_dataset:
      outdir = os.path.join('./',dirname)
      os.makedirs(outdir, exist_ok=True)
      updated_dataset[0].save(filedir='./',filename=f'mock_offset{ang+north_angle}deg_testoffset.fits')

if __name__ == '__main__':
   test_northup()
   test_wcs_and_offset()  
