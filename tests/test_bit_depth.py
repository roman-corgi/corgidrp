import configparser
import os
import shutil
import numpy as np
import corgidrp
from corgidrp.mocks import generate_coron_dataset_with_companions
from corgidrp.data import Image

def test_bit_depth():
    """
    Modifies the configuration file with different bit depth settings,
    generate and save mock image, check if image data is of the right type
    as specified.
    """
    # load config file
    config = configparser.ConfigParser()
    config.read(corgidrp.config_filepath)

    # save original information
    original_image_dtype = config.get("DATA", "image_dtype", fallback="64")
    original_dq_dtype = config.get("DATA", "dq_dtype", fallback="16")

    # test default settings
    config['DATA']['image_dtype'] = '64'
    config['DATA']['dq_dtype'] = '16'

    # write to file
    with open(corgidrp.config_filepath, 'w') as f:
        config.write(f)

    # update global variables
    corgidrp.update_pipeline_settings()

    # create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'bit_depth_test_data')
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.mkdir(output_dir)

    # generate mock image for testing
    dataset = generate_coron_dataset_with_companions(outdir=output_dir)
    
    # reset configuration file to its original state
    # reset the file before assert statements so that if the test fails the config file will not be affected
    config['DATA']['image_dtype'] = original_image_dtype
    config['DATA']['dq_dtype'] = original_dq_dtype
    with open(corgidrp.config_filepath, 'w') as f:
        config.write(f)
    corgidrp.update_pipeline_settings()

    # load image and check data and error is type float64, dq is type int16
    img = dataset[0]
    assert img.data.dtype.type == np.float64
    assert img.err.dtype.type == np.float64
    assert img.dq.dtype.type == np.uint16

    # change config file to use reduced bit-depth and test again
    config['DATA']['image_dtype'] = '32'
    config['DATA']['dq_dtype'] = '8'
    with open(corgidrp.config_filepath, 'w') as f:
        config.write(f)
    corgidrp.update_pipeline_settings()
    dataset2 = generate_coron_dataset_with_companions(outdir=output_dir)

    # reset config file
    config['DATA']['image_dtype'] = original_image_dtype
    config['DATA']['dq_dtype'] = original_dq_dtype
    with open(corgidrp.config_filepath, 'w') as f:
        config.write(f)
    corgidrp.update_pipeline_settings()

    # load image and check data and error is type float32, dq is type int8
    img = dataset2[0]
    assert img.data.dtype.type == np.float32
    assert img.err.dtype.type == np.float32
    assert img.dq.dtype.type == np.uint8

if __name__ == "__main__":
    test_bit_depth()