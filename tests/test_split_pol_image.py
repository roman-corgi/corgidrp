import numpy as np
import pytest
import corgidrp.mocks as mocks
import corgidrp.l2b_to_l3 as l2b_to_l3
import corgidrp.data as data
import matplotlib.pyplot as plt
import os

def test_image_splitting():
    """
    Create mock L2b polarimetric images, check that it is split correctly
    """
    image_WP1 = mocks.create_mock_l2b_polarimetric_image(dpamname='POL0', observing_mode='WFOV', left_image_value=1, right_image_value=2)
    image_WP2 = mocks.create_mock_l2b_polarimetric_image(dpamname='POL45', observing_mode='WFOV', left_image_value=1, right_image_value=2)
    input_dataset = data.Dataset([image_WP1, image_WP2])

    # test autocropping
    ## leave image_size parameter blank so the function automatically determines size
    output_dataset_autocrop = l2b_to_l3.split_image_by_polarization_state(input_dataset)
    ## create what the expected output data should look like
    radius = int(round((20.1 * ((0.8255 * 1e-6) / 2.363114) * 206265) / 0.0218))
    padding = 5
    img_size = 2 * (radius + padding)
    expected_output_autocrop = np.zeros(shape=(2, img_size, img_size))
    ## fill in expected values
    img_center = radius + padding
    for y in range(img_size):
        for x in range(img_size):
            if ((x-img_center)**2) + ((y-img_center)**2) <= radius**2:
                expected_output_autocrop[0,y,x] = 1
                expected_output_autocrop[1,y,x] = 2
    diff_0 = abs(output_dataset_autocrop.frames[1].data[0] - expected_output_autocrop[0])
    diff_1 = abs(output_dataset_autocrop.frames[1].data[1] - expected_output_autocrop[1])
    arrays = [image_WP1.data, image_WP2.data, output_dataset_autocrop.frames[1].data[0], output_dataset_autocrop.frames[1].data[1], expected_output_autocrop[0], expected_output_autocrop[1], diff_0, diff_1]

    save_dir = "plots"
    os.makedirs(save_dir, exist_ok=True)

    for i, arr in enumerate(arrays, start=1):
        plt.imshow(arr, origin="lower", cmap="viridis", aspect="auto")
        plt.colorbar(label="Value")
        plt.title(f"Array {i}")
        save_path = os.path.join(save_dir, f"array_{i}.png")
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

    ## check that actual output is as expected
    assert output_dataset_autocrop.frames[0].data == pytest.approx(expected_output_autocrop)
    assert output_dataset_autocrop.frames[1].data == pytest.approx(expected_output_autocrop)


    # test NaN pixels
    
if __name__ == "__main__":
    test_image_splitting()