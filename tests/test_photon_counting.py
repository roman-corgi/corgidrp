import pytest
import os
import astropy.time as time
import corgidrp.mocks as mocks
import corgidrp
from astropy.io import fits
import numpy as np
import corgidrp.walker as walker
import corgidrp.caldb as caldb
import corgidrp.data as data

dataset, ill_mean, dark_mean = mocks.create_photon_countable_frames(Nbrights=20, Ndarks=30, cosmic_rate=0)
thisfile_dir = os.path.dirname(__file__) # this file's folder
outputdir = os.path.join(thisfile_dir, 'simdata', 'pc_test_data')
if not os.path.exists(outputdir):
    os.mkdir(outputdir)
# empty out directory of any previous files
for f in os.listdir(outputdir):
    os.remove(os.path.join(outputdir,f))
dataset.save(outputdir, ['pc_frame_{0}.fits'.format(i) for i in range(len(dataset))])
l1_data_filelist = []
for f in os.listdir(outputdir):
    l1_data_filelist.append(os.path.join(outputdir, f))

def test_expected_results():
    '''Results are as expected theoretically.  Also runs raw frames through pre-processing pipeline.'''
    
    this_caldb = caldb.CalDB() # connection to cal DB
    # remove other KGain calibrations that may exist in case they don't have the added header keywords
    for i in range(len(this_caldb._db['Type'])):
        if this_caldb._db['Type'][i] == 'KGain':
            this_caldb._db = this_caldb._db.drop(i)
    this_caldb.save()
    # KGain
    kgain_val = 7 # default value used in mocks.create_photon_countable_frames()
    pri_hdr, ext_hdr = mocks.create_default_headers()
    ext_hdr["DRPCTIME"] = time.Time.now().isot
    ext_hdr['DRPVERSN'] =  corgidrp.__version__
    mock_input_dataset = data.Dataset(l1_data_filelist)
    kgain = data.KGain(np.array([[kgain_val]]), pri_hdr=pri_hdr, ext_hdr=ext_hdr, 
                    input_dataset=mock_input_dataset)
    # add in keywords that didn't make it into mock_kgain.fits, using values used in mocks.create_photon_countable_frames()
    kgain.ext_hdr['RN'] = 100
    kgain.ext_hdr['RN_ERR'] = 0
    kgain.save(filedir=outputdir, filename="mock_kgain.fits")
    this_caldb.create_entry(kgain)
    walker.walk_corgidrp(l1_data_filelist, '', outputdir, template="l1_to_l2b_pc.json")
    # get photon-counted frame
    for f in os.listdir(outputdir):
        if f.endswith('_pc.fits'):
            pc_filename = f
    pc_file = os.path.join(outputdir, pc_filename)
    pc_frame = fits.getdata(pc_file)
    pc_ext_hdr = fits.getheader(pc_file, 1)
    assert np.isclose(pc_frame.mean(), ill_mean - dark_mean)
    assert 'niter=2' in pc_ext_hdr["HISTORY"][-1]
    assert 'T_factor=5' in pc_ext_hdr["HISTORY"][-1]
    
    # empty out directory now
    for f in os.listdir(outputdir):
        os.remove(os.path.join(outputdir,f))

    #XXX negative value test, others from PhotonCount?
    #XXX change pc error: should come from doing pc on upper and lower bound of pixel values; same goes for trad and synthesized dark calibration
if __name__ == '__main__':
    test_expected_results()