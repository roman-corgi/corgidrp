import os
import glob
import corgidrp.data as data
import corgidrp.mocks as mocks

def test_dark_sub():
    """
    Generate mock input data and pass into dark subtraction function
    """
    # check that simulated data folder exists, and create if not
    datadir = os.path.join(os.path.dirname(__file__), "simdata")
    if not os.path.exists(datadir):
        os.mkdir(datadir)
    
    mocks.create_dark_calib_files(filedir=datadir)

    dark_filenames = glob.glob(os.path.join(datadir, "simcal_dark*.fits"))

    dark_dataset = data.Dataset(dark_filenames)

    assert(len(dark_dataset) == 10)
    print(dark_dataset.all_data.shape)


if __name__ == "__main__":
    test_dark_sub()