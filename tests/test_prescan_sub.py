import glob
import os

import corgidrp.data as data
from corgidrp.l1_to_l2a import prescan_biassub
import corgidrp.mocks as mocks

# Expected output image shapes
shapes = {
    'SCI' : {
        True : (1200,2200),
        False : (1024,1024)
    },
    'ENG' : {
        True: (2200,2200),
        False : (1024,1024)
    }
}

def test_prescan_sub():
    """
    Generate mock input data and pass into prescan processing function

    TODO: 
    * handle 'CAL' observation types
    * check that output is consistent with what we expect from II&T prescan code
    """
    ###### create simulated data
    # check that simulated data folder exists, and create if not
    datadir = os.path.join(os.path.dirname(__file__), "simdata")
    if not os.path.exists(datadir):
        os.mkdir(datadir)

    for obstype in ['SCI', 'ENG']:
        # create simulated data
        dataset = mocks.create_prescan_files(filedir=datadir, obstype=obstype)

        filenames = glob.glob(os.path.join(datadir, f"sim_prescan_{obstype}*.fits"))

        dataset = data.Dataset(filenames)

        if len(dataset) != 2:
            raise Exception(f"Mock dataset is an unexpected length ({len(dataset)}).")
        
        for return_full_frame in [True, False]:
            output_dataset = prescan_biassub(dataset, return_full_frame=return_full_frame)

            output_shape = output_dataset[0].data.shape
            if output_shape != shapes[obstype][return_full_frame]:
                raise Exception(f"Shape of output frame for {obstype}, return_full_frame={return_full_frame} is {output_shape}, \nwhen {shapes[obstype][return_full_frame]} was expected.")
    
            # check that data, err, and dq arrays are consistently modified
            dataset.all_data[0, 0, 0] = 0.
            if dataset[0].data[0, 0] != 0. :
                raise Exception("Modifying dataset.all_data did not modify individual frame data.")

            dataset[0].data[0,0] = 1.
            if dataset.all_data[0,0,0] != 1. :
                raise Exception("Modifying individual frame data did not modify dataset.all_data.")

            dataset.all_err[0, 0, 0] = 0.
            if dataset[0].err[0, 0] != 0. :
                raise Exception("Modifying dataset.all_err did not modify individual frame err.")

            dataset[0].err[0,0] = 1.
            if dataset.all_err[0,0,0] != 1. :
                raise Exception("Modifying individual frame err did not modify dataset.all_err.")

            dataset.all_dq[0, 0, 0] = 0.
            if dataset[0].dq[0, 0] != 0. :
                raise Exception("Modifying dataset.all_dq did not modify individual frame dq.")

            dataset[0].dq[0,0] = 1.
            if dataset.all_dq[0,0,0] != 1. :
                raise Exception("Modifying individual frame dq did not modify dataset.all_dq.")

if __name__ == "__main__":
    test_prescan_sub()