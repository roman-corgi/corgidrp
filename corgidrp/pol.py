




def generate_mueller_matrix_cal(input_dataset, path_to_pol_ref_file="./data/pol_ref_file.fits"):
    '''
    A step function that generates a MuellerMatrix calibration file from a dataset.
    If the dataset is an ND dataset, then it generates an ND MuellerMatrix calibration file.

    Args: 
        input_dataset (list): A list of CorgiDRP data objects that will be used to generate the Mueller Matrix.
            This should be a list of either all ND datasets or all non-ND datasets.
    Returns:
        mueller_matrix (MuellerMatrix or NDMuellerMatrix): The generated Mueller Matrix calibration file.
    '''

    dataset = input_dataset.copy()

    # check that all the data in the dataset is either ND or non-ND

    # split the datasets into different targest

    # measure the stokes vector for each target

    # generate the mueller matrix from the stokes vectors and the known properties

    # propagate errors

    # create the mueller matrix object

    # return the mueller matrix object (nd or non-nd)
    pass


    

    