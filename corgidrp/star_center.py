import numpy as np
from scipy.optimize import minimize

# Define default parameters for the satellite spot finding algorithm
# Spot separation in pixels
# separation in lambda/D and multiply by pixel per lambda/D
spot_separation_nfov = 6.5*(51.46*0.575/13) # 14.79
spot_separation_spec = 6.0*(51.46*0.730/13) # 17.34
spot_separation_wfov = 13*(51.46*0.825/13) # 42.4545

satellite_spot_parameters_defaults = {
    "NFOV": {
        "offset": {
            "spotSepPix": spot_separation_nfov,
            "roiRadiusPix": 4.5,
            "probeRotVecDeg": [0, 90],
            "nSubpixels": 100,
            "nSteps": 7,
            "stepSize": 1,
            "nIter": 6,
        },
        "separation": {
            "spotSepPix": spot_separation_nfov,
            "roiRadiusPix": 1.5,
            "probeRotVecDeg": [0, 90],
            "nSubpixels": 100,
            "nSteps": 21,
            "stepSize": 0.25,
            "nIter": 5,
        }
    },
    "SPEC660": {
        "offset": {
            "spotSepPix": spot_separation_spec,
            "roiRadiusPix": 6,
            "probeRotVecDeg": [0,],
            "nSubpixels": 100,
            "nSteps": 9,
            "stepSize": 1,
            "nIter": 6,
        },
        "separation": {
            "spotSepPix": spot_separation_spec,
            "roiRadiusPix": 4,
            "probeRotVecDeg": [0,],
            "nSubpixels": 100,
            "nSteps": 21,
            "stepSize": 0.25,
            "nIter": 5,
        }
    },
    "SPEC730": {
        "offset": {
            "spotSepPix": spot_separation_spec,
            "roiRadiusPix": 6,
            "probeRotVecDeg": [0,],
            "nSubpixels": 100,
            "nSteps": 9,
            "stepSize": 1,
            "nIter": 6,
        },
        "separation": {
            "spotSepPix": spot_separation_spec,
            "roiRadiusPix": 4,
            "probeRotVecDeg": [0,],
            "nSubpixels": 100,
            "nSteps": 21,
            "stepSize": 0.25,
            "nIter": 5,
        }
    },
    "WFOV": {
        "offset": {
            "spotSepPix": spot_separation_wfov,
            "roiRadiusPix": 4.5,
            "probeRotVecDeg": [0, 90],
            "nSubpixels": 100,
            "nSteps": 7,
            "stepSize": 1,
            "nIter": 6,
        },
        "separation": {
            "spotSepPix": spot_separation_wfov,
            "roiRadiusPix": 4.5,
            "probeRotVecDeg": [0, 90],
            "nSubpixels": 100,
            "nSteps": 21,
            "stepSize": 0.25,
            "nIter": 5,
        }
    }
}


def validate_satellite_spot_parameters(params):
    """
    Checks if a dictionary conforms to the required satellite spot parameters format.

    Args:
        params (dict): Dictionary to validate.

    Returns:
        bool: True if valid, False otherwise.
    """
    required_structure = {
        "offset": {
            "spotSepPix": float,
            "roiRadiusPix": float,
            "probeRotVecDeg": list,
            "nSubpixels": int,
            "nSteps": int,
            "stepSize": float,
            "nIter": int,
        },
        "separation": {
            "spotSepPix": float,
            "roiRadiusPix": float,
            "probeRotVecDeg": list,
            "nSubpixels": int,
            "nSteps": int,
            "stepSize": float,
            "nIter": int,
        }
    }

    if not isinstance(params, dict):
        return False

    for section in ['offset', 'separation']:
        if section not in params or not isinstance(params[section], dict):
            return False
        for key, expected_type in validate_satellite_spot_parameters.__annotations__[section].items():
            if key not in params[section]:
                return False
            if not isinstance(params[section][key], expected_type):
                return False

    return True


def update_parameters(params, new_values):
    """
    Updates a nested dictionary of parameters with new values.

    Args:
        params (dict):
            Original nested dictionary containing initial parameter values.
            The structure should be: {key1: {subkey1: val, subkey2: val}, key2: {...}}
        new_values (dict):
            Nested dictionary containing new parameter values to update.
            Must match the structure of `params`.

    Returns:
        dict: The updated nested dictionary.

    Raises:
        KeyError: If `new_values` contains keys or subkeys not present in `params`.

    Example:
        >>> params = {'a': {'x': 1, 'y': 2}, 'b': {'z': 3}}
        >>> new_values = {'a': {'x': 10}}
        >>> update_parameters(params, new_values)
        {'a': {'x': 10, 'y': 2}, 'b': {'z': 3}}

        >>> new_values = {'b': {'w': 5}}
        >>> update_parameters(params, new_values)
        KeyError: "Subkey 'w' is not a valid parameter under key 'b'."
    """
    for key, subdict in new_values.items():
        if key not in params:
            raise KeyError(f"Key '{key}' is not a valid parameter.")

        for subkey, value in subdict.items():
            if subkey not in params[key]:
                raise KeyError(f"Subkey '{subkey}' is not a valid parameter under key '{key}'.")

            params[key][subkey] = value

    return params


def circle(nx, ny, roiRadiusPix, xShear, yShear, nSubpixels=100):
    """
    Generates a circular aperture with an antialiased edge at specified offsets.

    This function is used as a software window for isolating a region of interest. 
    Grayscale edges are applied to account for low detector sampling, ensuring that 
    fractional values along the edges are preserved.

    Args:
        nx (int):  
            Width of the 2D array to create.  
        ny (int):  
            Height of the 2D array to create.  
        roiRadiusPix (float):  
            Radius of the circle in pixels.  
        xShear (float):  
            Lateral offset of the circle's center from the array's center pixel along the x-axis.  
        yShear (float):  
            Lateral offset of the circle's center from the array's center pixel along the y-axis.  
        nSubpixels (int, optional):  
            Number of subpixels per edge pixel used for antialiasing. Each edge pixel is subdivided 
            into a square subarray of `nSubpixels` across, where binary values are assigned and 
            averaged to produce grayscale edge values between 0 and 1.  
            Defaults to 100. Must be a positive integer.  

    Returns:
        numpy.ndarray:  
            A 2D array containing the generated circular aperture mask.  
    """

    if nx % 2 == 0:
        x = np.linspace(-nx / 2.0, nx / 2.0 - 1, nx) - xShear
    elif nx % 2 == 1:
        x = np.linspace(-(nx - 1) / 2.0, (nx - 1) / 2.0, nx) - xShear

    if ny % 2 == 0:
        y = np.linspace(-ny / 2.0, ny / 2.0 - 1, ny) - yShear
    elif ny % 2 == 1:
        y = np.linspace(-(ny - 1) / 2.0, (ny - 1) / 2.0, ny) - yShear

    dx = x[1] - x[0]
    [X, Y] = np.meshgrid(x, y)
    RHO = np.sqrt(X * X + Y * Y)

    halfWindowWidth = np.sqrt(2.0) * dx
    mask = -1 * np.ones(RHO.shape)
    mask[np.abs(RHO) < roiRadiusPix - halfWindowWidth] = 1
    mask[np.abs(RHO) > roiRadiusPix + halfWindowWidth] = 0
    grayInds = np.array(np.nonzero(mask == -1))
    # print('Number of grayscale points = %d' % grayInds.shape[1])

    dxHighRes = 1.0 / float(nSubpixels)
    xUp = (
        np.linspace(-(nSubpixels - 1) / 2.0, (nSubpixels - 1) / 2.0, nSubpixels)
        * dxHighRes
    )
    [Xup, Yup] = np.meshgrid(xUp, xUp)

    subpixelArray = np.zeros((nSubpixels, nSubpixels))
    # plt.figure(); plt.imshow(RHO); plt.colorbar(); plt.pause(0.1)

    # Compute the value between 0 and 1 of each edge pixel along the circle by
    # taking the mean of the binary subpixels.
    for iInterior in range(grayInds.shape[1]):
        subpixelArray = 0 * subpixelArray

        xCenter = X[grayInds[0, iInterior], grayInds[1, iInterior]]
        yCenter = Y[grayInds[0, iInterior], grayInds[1, iInterior]]
        RHOHighRes = np.sqrt((Xup + xCenter) ** 2 + (Yup + yCenter) ** 2)
        # plt.figure(); plt.imshow(RHOHighRes); plt.colorbar(); plt.pause(1/20)

        subpixelArray[RHOHighRes <= roiRadiusPix] = 1
        pixelValue = np.sum(subpixelArray) / float(nSubpixels * nSubpixels)
        mask[grayInds[0, iInterior], grayInds[1, iInterior]] = pixelValue

    return mask


def find_optimum_1d(xVec, arrayToFit):
    """
    Fits a parabola to a 1D array and returns the location of the minimum or maximum.

    Args:
        xVec (array_like):  
            1D array of coordinate values corresponding to `arrayToFit`.  
        arrayToFit (array_like):  
            1D array of values to fit.  

    Returns:
        xOpt (float):  
            Best-fit value for the optimum (i.e., minimum or maximum) of the parabola.  
    """

    if len(xVec) != len(arrayToFit):
        raise ValueError("Both input arrays must have same length.")

    # Normalize arrayToFit to avoid spurious numerical precision issues.
    arrayToFit = arrayToFit / np.max(arrayToFit)

    orderOfFit = 2
    p0, p1, _ = np.polyfit(xVec, arrayToFit, orderOfFit)

    # Avoid divide by zero error
    if np.abs(p0) < np.finfo(p0).eps:
        raise ValueError(
            "Quadratic fit failed because data being "
            + "fitted has no quadratic component."
        )

    # Find x value at the optimum
    xOpt = -p1 / (2 * p0)

    return xOpt


def find_optimum_2d(xVec, yVec, arrayToFit, mask):
    """
    Fits a paraboloid to a 2D array and returns the location of the minimum or maximum.

    Args:
        xVec (array_like):  
            1D array of coordinate values along axis 1 of `arrayToFit`.  
        yVec (array_like):  
            1D array of coordinate values along axis 0 of `arrayToFit`.  
        arrayToFit (array_like):  
            2D array of values to fit.  
        mask (array_like):  
            2D boolean mask indicating which pixels to use. Must have the same shape as `arrayToFit`.  

    Returns:
        tuple:  
            - `xBest` (float): Best-fit value along axis 1.  
            - `yBest` (float): Best-fit value along axis 0.  

    Notes:
        - Modified from code at  
          [MATLAB Central](https://au.mathworks.com/matlabcentral/answers/5482-fit-a-3d-curve).  
        - Based on equations from  
          [Math StackExchange](https://math.stackexchange.com/questions/2010758/how-do-i-fit-a-paraboloid-surface-to-nine-points-and-find-the-minimum).  
    """

    if arrayToFit.shape != mask.shape:
        raise ValueError("arrayToFit and mask must have same shape")
    nx = len(xVec)
    ny = len(yVec)
    if arrayToFit.shape[0] != ny:
        raise ValueError("yVec and axis 0 of arrayToFit must have same length")
    if arrayToFit.shape[1] != nx:
        raise ValueError("xVec and axis 1 of arrayToFit must have same length")

    # Normalize arrayToFit to avoid spurious numerical precision issues.
    arrayToFit = arrayToFit / np.max(arrayToFit)

    maskBool = np.asarray(mask).astype(bool)
    nPix = np.sum(mask.astype(int))

    #  Set the basis functions
    [X, Y] = np.meshgrid(xVec, yVec)
    f0 = X**2
    f1 = X * Y
    f2 = Y**2
    f3 = X
    f4 = Y
    f5 = np.ones((ny, nx))

    # Write as matrix equation and solve for coefficients
    A = np.concatenate(
        (
            f0[maskBool].reshape((nPix, 1)),
            f1[maskBool].reshape((nPix, 1)),
            f2[maskBool].reshape((nPix, 1)),
            f3[maskBool].reshape((nPix, 1)),
            f4[maskBool].reshape((nPix, 1)),
            f5[maskBool].reshape((nPix, 1)),
        ),
        axis=1,
    )
    y = arrayToFit[maskBool].flatten()
    temp = np.linalg.lstsq(A, y, rcond=None)
    coeffs = temp[0]

    # Take partial derivatives w.r.t. x and y, set to zero, and solve for
    # x and y.
    # G = 0 = a*X**2 + b*X*Y + c*Y**2 + d*X + e*Y + f
    # @G/@X = 2*a*X + b*Y + d = 0 --> Solve for X. Use eq. for Y below.
    # @G/@Y = b*X + 2*c*Y + e = 0 --> Solve for Y in terms of X.
    a, b, c, d, e = coeffs[0:5]

    # Avoid divide by zero errors
    x_denominator = b * b - 4.0 * a * c
    y_denominator = 2 * c
    # The factors 4 and 2 come from the values in the denomimators
    if (
        np.abs(x_denominator) < 4 * np.finfo(x_denominator).resolution
        or np.abs(y_denominator) < 2 * np.finfo(y_denominator).resolution
    ):
        raise ValueError(
            "Quadratic fit failed because data being "
            + "fitted has no quadratic component."
        )

    xBest = (2.0 * c * d - e * b) / (b * b - 4.0 * a * c)
    yBest = -(b * xBest + e) / (2 * c)

    return xBest, yBest


def _roi_mask_for_spots(optim_params, fixed_params):
    """
    Computes the stellar offset and spot separation simultaneously.

    Args:
        optim_params (list):  
            List of parameters to be optimized by `scipy.optimize.minimize()`.  
            These three variables must be provided in this exact order:  
            `[xOffsetGuess, yOffsetGuess, spotSepGuessPix]`          
            - `xOffsetGuess` (float): Initial guess for the star offset from the array center pixel in the x direction. Units: pixels.  
            - `yOffsetGuess` (float): Initial guess for the star offset from the array center pixel in the y direction. Units: pixels.  
            - `spotSepGuessPix` (float): Initial guess for the spot separation from the star center. Units: pixels.  
        fixed_params (list):  
            List of fixed parameters used in computing the cost function.  
            These four variables must be provided in this exact order:  
            `[spotArray, probeRotVecDeg, roiRadiusPix, nSubpixels]`         
            - `spotArray` (numpy.ndarray):  
              2D array of DM-generated satellite spots.  
              Computed externally from probed images as `(Iplus + Iminus)/2 - Iunprobed`.  
            - `probeRotVecDeg` (array_like):  
              1D array of angles (in degrees) counterclockwise from the x-axis  
              to rotate the regions of interest when summing satellite spots.  
              A pair of satellite spots is represented by a single value.  
              - Example for a single pair along the x-axis: `[0]` (not `[0, 180]`).  
              - Example for a plus-shaped layout of spots: `[0, 90]`.  
            - `roiRadiusPix` (float):  
              Radius of each region of interest used when summing the intensity  
              of a satellite spot. Units: pixels.  
            - `nSubpixels` (int):  
              Number of subpixels used to refine edge values of the region-of-interest mask.  
              The value of the edge pixels in the ROI is the mean of all subpixel values.  

    Returns:
        tuple:  
            - `cost` (float):  
              The summed total intensity passing through the region-of-interest mask,  
              multiplied by `-1`. Used as the cost function to be minimized by `scipy.optimize.minimize()`.  
            - `roi_mask` (numpy.ndarray):  
              2D array with values between 0 and 1 (inclusive), indicating how much  
              to weight the intensity in each pixel in the summed cost.  
    """

    # Unpack and check the input list of optimization variables
    try:
        xOffsetGuess, yOffsetGuess, spotSepGuessPix = optim_params
    except:
        raise ValueError("optim_params must be an iterable of length 3")

    # Unpack and check the values in the list of fixed inputs.
    try:
        spotArray, probeRotVecDeg, roiRadiusPix, nSubpixels = fixed_params
    except:
        raise ValueError("fixed_params must be an iterable of length 4")

    # Generate mask to isolate the satellite spots in spotArray
    ny, nx = spotArray.shape
    roi_mask = np.zeros((ny, nx))
    for iRot, rotDeg in enumerate(probeRotVecDeg):
        rotRad = np.radians(rotDeg)
        xProbePlusCoord = np.array(
            [
                np.sin(rotRad) * spotSepGuessPix + yOffsetGuess,
                np.cos(rotRad) * spotSepGuessPix + xOffsetGuess,
            ]
        )
        rotRad += np.pi
        xProbeMinusCoord = np.array(
            [
                np.sin(rotRad) * spotSepGuessPix + yOffsetGuess,
                np.cos(rotRad) * spotSepGuessPix + xOffsetGuess,
            ]
        )

        roi_mask += circle(
            nx,
            ny,
            roiRadiusPix,
            xProbePlusCoord[1],
            xProbePlusCoord[0],
            nSubpixels=nSubpixels,
        )
        roi_mask += circle(
            nx,
            ny,
            roiRadiusPix,
            xProbeMinusCoord[1],
            xProbeMinusCoord[0],
            nSubpixels=nSubpixels,
        )

    # Cost is negative because scipy minimizes the cost function
    # but we want to maximize the sum.
    cost = -np.sum(roi_mask * spotArray)

    return cost, roi_mask


def _cost_func_spots(optim_params, fixed_params):
    """
    Return only the cost value for scipy.optimize.minimize().
        
    This is a thin wrapper for occastro.roi_mask_for_spots() because
    scipy.optimize.minimize() requires the given function to return only
    the cost function value but we want some other diagnostic outputs from
    occastro.roi_mask_for_spots() as well.

    Args:
        optim_params (tuple): Tuple of optimization parameters.
        fixed_params (dict): Dictionary of fixed parameters.

    Returns:
        float: The cost function value.
    """

    cost, _ = _roi_mask_for_spots(optim_params, fixed_params)

    return cost


def calc_star_location_and_spot_separation(
    spotArray, xOffsetGuess, yOffsetGuess, tuningParamDict
):
    """
    Calculates the center of the occulted star using satellite spots.

    A single processed image of satellite spots is used. Multiple software masks 
    are generated and applied to the measured spots to determine the stellar location. 
    The best estimate of the star's location is the one that maximizes the total summed 
    energy, as determined by a 2D quadratic fit.

    All filenames may be absolute or relative paths. If relative, they will be 
    relative to the current working directory, not to any particular location in Calibration.

    Args:
        spotArray (numpy.ndarray): 
            2D array of the DM-generated satellite spots. Calculated outside 
            this function from probed images as `(Iplus + Iminus)/2 - Iunprobed`.
        xOffsetGuess (float): 
            Starting guess for the number of pixels in the x direction that the 
            star is offset from the center pixel of `spotArray`. The convention 
            for the center pixel follows that of FFTs.
        yOffsetGuess (float): 
            Starting guess for the number of pixels in the y direction that the 
            star is offset from the center pixel of `spotArray`. The convention 
            for the center pixel follows that of FFTs.
        tuningParamDict (dict): 
            Dictionary containing tuning parameter values.
            
            - **spotSepPix (float)**:  
              Expected separation of the satellite spots from the star. Used as 
              the separation for the center of the region of interest. Units of 
              pixels. Compute beforehand as separation in lambda/D multiplied by 
              pixels per lambda/D.
            - **roiRadiusPix (float)**:  
              Radius of each region of interest used when summing the intensity 
              of a satellite spot. Units of pixels.
            - **probeRotVecDeg (array_like)**:  
              1D array of angles (in degrees) counterclockwise from the x-axis 
              to rotate the regions of interest when summing the satellite spots.  
              A pair of satellite spots is given by just one value.  
              - Example for a single pair along the x-axis: `[0]` (not `[0, 180]`).  
              - Example for a plus-shaped layout of spots: `[0, 90]`.
            - **nSubpixels (int)**:  
              Number of subpixels across used to make edge values of the region-of-interest 
              mask. The value of the edge pixels in the ROI is the mean of all subpixel values.
            - **nSteps (int)**:  
              Number of points used along each direction for the grid search. 
              Odd numbers are preferable for symmetry when the array is truly centered.
            - **stepSize (float)**:  
              Step size used in the grid search. Units of pixels.
            - **nIter (int)**:  
              Number of iterations in the loop that refines the stellar center location.

    Returns:
        tuple:  
            - **xOffsetEst (float)**: Estimated lateral offset of the stellar center 
              from the center pixel of `spotArray` in the x direction.  
            - **yOffsetEst (float)**: Estimated lateral offset of the stellar center 
              from the center pixel of `spotArray` in the y direction.  
              
              The convention for the center pixel follows that of FFTs.
            - **roi_mask (numpy.ndarray)**:  
              2D float array of the best-fit region-of-interest mask used to fit 
              the translation and scaling of the ROI regions to `spotArray`.
    """

    # tuningParamDict = loadyaml(fnTuning)
    # spotSepGuessPix = tuningParamDict["spotSepGuessPix"]
    spotSepGuessPix = tuningParamDict["spotSepPix"]
    roiRadiusPix = tuningParamDict["roiRadiusPix"]
    probeRotVecDeg = tuningParamDict["probeRotVecDeg"]
    nSubpixels = tuningParamDict["nSubpixels"]
    offset_tol_pix = tuningParamDict["offset_tol_pix"]
    sep_tol_pix = tuningParamDict["sep_tol_pix"]
    opt_method = tuningParamDict["opt_method"]

    # check.real_positive_scalar(spotSepGuessPix, 'spotSepGuessPix', TypeError)
    # check.real_positive_scalar(roiRadiusPix, 'roiRadiusPix', TypeError)
    # check.oneD_array(probeRotVecDeg, 'probeRotVecDeg', TypeError)
    # check.positive_scalar_integer(nSubpixels, 'nSubpixels', TypeError)

    # Define optimization input lists
    bounds = [
        [xOffsetGuess - offset_tol_pix, xOffsetGuess + offset_tol_pix],
        [yOffsetGuess - offset_tol_pix, yOffsetGuess + offset_tol_pix],
        [spotSepGuessPix - sep_tol_pix, spotSepGuessPix + sep_tol_pix],
    ]
    initial_guess = [xOffsetGuess, yOffsetGuess, spotSepGuessPix]
    fixed_params = [spotArray, probeRotVecDeg, roiRadiusPix, nSubpixels]

    # Run the optimization
    result = minimize(
        _cost_func_spots,
        initial_guess,
        args=fixed_params,
        bounds=bounds,
        method=opt_method,
    )
    if result.success:
        fitted_params = result.x
    else:
        raise ValueError(result.message)

    param_dict = {
        "xOffset": fitted_params[0],
        "yOffset": fitted_params[1],
        "spotSepPix": fitted_params[2],
    }

    _, roi_mask = _roi_mask_for_spots(fitted_params, fixed_params)

    return param_dict, roi_mask


def calc_star_location_from_spots(spotArray, xOffsetGuess, yOffsetGuess, tuningParamDict):
    """
    Calculates the center of the occulted star using satellite spots.

    A single processed image of satellite spots is used. Multiple software masks 
    are generated and applied to the measured spots to determine the stellar location. 
    The best estimate of the star's location is the one that maximizes the total summed 
    energy, as determined by a 2D quadratic fit.

    All filenames may be absolute or relative paths. If relative, they will be 
    relative to the current working directory, not to any particular location in Calibration.

    Args:
        spotArray (numpy.ndarray): 
            2D array of the DM-generated satellite spots. Calculated outside 
            this function from probed images as `(Iplus + Iminus)/2 - Iunprobed`.
        xOffsetGuess (float): 
            Starting guess for the number of pixels in the x direction that the 
            star is offset from the center pixel of `spotArray`. The convention 
            for the center pixel follows that of FFTs.
        yOffsetGuess (float): 
            Starting guess for the number of pixels in the y direction that the 
            star is offset from the center pixel of `spotArray`. The convention 
            for the center pixel follows that of FFTs.
        tuningParamDict (dict): 
            Dictionary containing tuning parameter values.
            
            - **spotSepPix (float)**:  
              Expected separation of the satellite spots from the star. Used as 
              the separation for the center of the region of interest. Units of 
              pixels. Compute beforehand as separation in lambda/D multiplied by 
              pixels per lambda/D.
            - **roiRadiusPix (float)**:  
              Radius of each region of interest used when summing the intensity 
              of a satellite spot. Units of pixels.
            - **probeRotVecDeg (array_like)**:  
              1D array of angles (in degrees) counterclockwise from the x-axis 
              to rotate the regions of interest when summing the satellite spots.  
              A pair of satellite spots is given by just one value.  
              - Example for a single pair along the x-axis: `[0]` (not `[0, 180]`).  
              - Example for a plus-shaped layout of spots: `[0, 90]`.
            - **nSubpixels (int)**:  
              Number of subpixels across used to make edge values of the region-of-interest 
              mask. The value of the edge pixels in the ROI is the mean of all subpixel values.
            - **nSteps (int)**:  
              Number of points used along each direction for the grid search. 
              Odd numbers are preferable for symmetry when the array is truly centered.
            - **stepSize (float)**:  
              Step size used in the grid search. Units of pixels.
            - **nIter (int)**:  
              Number of iterations in the loop that refines the stellar center location.

    Returns:
        tuple:  
            - **xOffsetEst (float)**: Estimated lateral offset of the stellar center 
              from the center pixel of `spotArray` in the x direction.  
            - **yOffsetEst (float)**: Estimated lateral offset of the stellar center 
              from the center pixel of `spotArray` in the y direction.  
              
            The convention for the center pixel follows that of FFTs.
    """
    
    # check.twoD_array(spotArray, 'spotArray', TypeError)
    # check.real_scalar(xOffsetGuess, 'xOffsetGuess', TypeError)
    # check.real_scalar(yOffsetGuess, 'yOffsetGuess', TypeError)

    # tuningParamDict = loadyaml(fnYAML)
    spotSepPix = tuningParamDict["spotSepPix"]
    roiRadiusPix = tuningParamDict["roiRadiusPix"]
    probeRotVecDeg = tuningParamDict["probeRotVecDeg"]
    nSubpixels = tuningParamDict["nSubpixels"]
    nSteps = tuningParamDict["nSteps"]
    stepSize = tuningParamDict["stepSize"]
    nIter = tuningParamDict["nIter"]

    # check.real_positive_scalar(spotSepPix, 'spotSepPix', TypeError)
    # check.real_positive_scalar(roiRadiusPix, 'roiRadiusPix', TypeError)
    # check.oneD_array(probeRotVecDeg, 'probeRotVecDeg', TypeError)
    # check.positive_scalar_integer(nSubpixels, 'nSubpixels', TypeError)
    # check.positive_scalar_integer(nSteps, 'nSteps', TypeError)
    # check.real_positive_scalar(stepSize, 'stepSize', TypeError)
    # check.positive_scalar_integer(nIter, 'nIter', TypeError)

    ny, nx = spotArray.shape
    costFuncMat = np.zeros((nSteps, nSteps))
    xOffsetEst = 0
    yOffsetEst = 0

    for iter_ in range(nIter):
        xOffsetVec = (
            np.arange(nSteps) * stepSize - (nSteps - 1) / 2 * stepSize + xOffsetEst
        )
        yOffsetVec = (
            np.arange(nSteps) * stepSize - (nSteps - 1) / 2 * stepSize + yOffsetEst
        )

        for iy, yOffset in enumerate(yOffsetVec):
            for ix, xOffset in enumerate(xOffsetVec):
                # Generate mask of all ROI regions
                roi_mask = np.zeros((ny, nx))
                for iRot, rotDeg in enumerate(probeRotVecDeg):
                    rotRad = np.radians(rotDeg)
                    xProbePlusCoord = np.array(
                        [
                            np.sin(rotRad) * spotSepPix + yOffsetGuess + yOffset,
                            np.cos(rotRad) * spotSepPix + xOffsetGuess + xOffset,
                        ]
                    )
                    rotRad += np.pi
                    xProbeMinusCoord = np.array(
                        [
                            np.sin(rotRad) * spotSepPix + yOffsetGuess + yOffset,
                            np.cos(rotRad) * spotSepPix + xOffsetGuess + xOffset,
                        ]
                    )

                    roi_mask += circle(
                        nx,
                        ny,
                        roiRadiusPix,
                        xProbePlusCoord[1],
                        xProbePlusCoord[0],
                        nSubpixels=nSubpixels,
                    )
                    roi_mask += circle(
                        nx,
                        ny,
                        roiRadiusPix,
                        xProbeMinusCoord[1],
                        xProbeMinusCoord[0],
                        nSubpixels=nSubpixels,
                    )

                costFuncMat[iy, ix] = np.sum(roi_mask * spotArray)

        xOffsetEst, yOffsetEst = find_optimum_2d(
            xOffsetVec, yOffsetVec, costFuncMat, np.ones_like(costFuncMat)
        )

    # estimates are w.r.t. initial guess. Return, "from the center pixel"
    return xOffsetEst + xOffsetGuess, yOffsetEst + yOffsetGuess


def calc_spot_separation(spotArray, xOffset, yOffset, tuningParamDict):
    """
    Calculates the radial separation in pixels of the satellite spots.

    A single processed image of satellite spots is used. Several software masks 
    are generated and applied to the measured spots to determine the radial 
    separation of the spots from the given star center. The best estimate of 
    the radial spot separation is the one that maximizes the total summed 
    energy in the software mask, as determined by a 1D quadratic fit.

    All filenames may be absolute or relative paths. If relative, they will be 
    relative to the current working directory, not to any particular location 
    in Calibration.

    Args:
        spotArray (numpy.ndarray): 
            2D array of the DM-generated satellite spots. Calculated outside 
            this function from probed images as `(Iplus + Iminus)/2 - Iunprobed`.
        xOffset (float): 
            Previously estimated stellar center offset from the array's center 
            pixel. Units of pixels. The convention for the center pixel follows 
            that of FFTs.
        yOffset (float): 
            Previously estimated stellar center offset from the array's center 
            pixel. Units of pixels. The convention for the center pixel follows 
            that of FFTs.
        tuningParamDict (dict): 
            Dictionary containing tuning parameter values.
            
            - **spotSepPix (float)**:  
              Expected (model-based) separation of the satellite spots from the 
              star. Used as the starting point for the separation for the 
              center of the region of interest. Units of pixels. Compute 
              beforehand as separation in lambda/D multiplied by pixels per 
              lambda/D (e.g., `6.5 * (51.46 * 0.575 / 13)`).
            - **roiRadiusPix (float)**:  
              Radius of each region of interest used when summing the intensity 
              of a satellite spot. Units of pixels.
            - **probeRotVecDeg (array_like)**:  
              1D array of angles (in degrees) counterclockwise from the x-axis 
              to rotate the regions of interest when summing the satellite spots. 
              A pair of satellite spots is given by just one value.  
              - Example for a single pair along the x-axis: `[0]` (not `[0, 180]`).  
              - Example for a plus-shaped layout of spots: `[0, 90]`.
            - **nSubpixels (int)**:  
              Number of subpixels across used to make edge values of the region-of-interest 
              mask. The value of the edge pixels in the ROI is the mean of all the subpixel values.
            - **nSteps (int)**:  
              Number of points used along each direction for the grid search. 
              Odd numbers are preferable for symmetry when the array is truly centered.
            - **stepSize (float)**:  
              Step size used in the grid search. Units of pixels.
            - **nIter (int)**:  
              Number of iterations in the loop that refines the radial separation of the satellite spots.

    Returns:
        float: 
            Estimated radial separation of the satellite spots from the stellar center. 
            Units of pixels.
    """
    # check.twoD_array(spotArray, 'spotArray', TypeError)
    # check.real_scalar(xOffset, 'xOffset', TypeError)
    # check.real_scalar(yOffset, 'yOffset', TypeError)

    # tuningParamDict = loadyaml(fnYAML)
    # spotSepGuessPix = tuningParamDict["spotSepGuessPix"]
    spotSepGuessPix = tuningParamDict["spotSepPix"]
    roiRadiusPix = tuningParamDict["roiRadiusPix"]
    probeRotVecDeg = tuningParamDict["probeRotVecDeg"]
    nSubpixels = tuningParamDict["nSubpixels"]
    nSteps = tuningParamDict["nSteps"]
    stepSize = tuningParamDict["stepSize"]
    nIter = tuningParamDict["nIter"]

    # check.real_positive_scalar(spotSepGuessPix, 'spotSepGuessPix', TypeError)
    # check.real_positive_scalar(roiRadiusPix, 'roiRadiusPix', TypeError)
    # check.oneD_array(probeRotVecDeg, 'probeRotVecDeg', TypeError)
    # check.positive_scalar_integer(nSubpixels, 'nSubpixels', TypeError)
    # check.positive_scalar_integer(nSteps, 'nSteps', TypeError)
    # check.real_positive_scalar(stepSize, 'stepSize', TypeError)
    # check.positive_scalar_integer(nIter, 'nIter', TypeError)

    ny, nx = spotArray.shape
    costFuncVec = np.zeros((nSteps,))
    spotSepEst = spotSepGuessPix  # initialize

    for iter_ in range(nIter):
        spotSepVec = (
            np.arange(nSteps) * stepSize - (nSteps - 1) / 2 * stepSize + spotSepEst
        )

        for iSep, spotSep in enumerate(spotSepVec):
            # Generate mask of all ROI regions
            roi_mask = np.zeros((ny, nx))
            for iRot, rotDeg in enumerate(probeRotVecDeg):
                rotRad = np.radians(rotDeg)
                xProbePlusCoord = np.array(
                    [
                        np.sin(rotRad) * spotSep + yOffset,
                        np.cos(rotRad) * spotSep + xOffset,
                    ]
                )
                rotRad += np.pi
                xProbeMinusCoord = np.array(
                    [
                        np.sin(rotRad) * spotSep + yOffset,
                        np.cos(rotRad) * spotSep + xOffset,
                    ]
                )

                roi_mask += circle(
                    nx,
                    ny,
                    roiRadiusPix,
                    xProbePlusCoord[1],
                    xProbePlusCoord[0],
                    nSubpixels=nSubpixels,
                )
                roi_mask += circle(
                    nx,
                    ny,
                    roiRadiusPix,
                    xProbeMinusCoord[1],
                    xProbeMinusCoord[0],
                    nSubpixels=nSubpixels,
                )

            costFuncVec[iSep] = np.sum(roi_mask * spotArray)

        # At the first iteration, uses the maximum instead of a
        # quadratic fit in order to get a larger capture range.
        if iter_ == 0 and nIter > 1:
            bestInd = np.argmax(costFuncVec)
            spotSepEst = spotSepVec[bestInd]
        else:
            spotSepEst = find_optimum_1d(spotSepVec, costFuncVec)

    return spotSepEst


def star_center_from_satellite_spots(
    img_ref,
    img_sat_spot,
    star_coordinate_guess,
    thetaOffsetGuess,
    satellite_spot_parameters,
):
    """
    Estimates the star center and spot locations from satellite spot images and science data.

    Args:
        img_ref (numpy.ndarray):
            2D image representing a clean occulted focal-plane image with a base DM setting.
        img_sat_spot (numpy.ndarray):
            2D image representing a clean occulted focal-plane image with a relative satellite-spot DM setting added.
        star_coordinate_guess (tuple of float):
            Starting guess for the absolute (x, y) coordinate of the star (in pixels).
            The offset calculation is referenced to the image center, which is assumed
            to be (image_size // 2, image_size // 2).
        thetaOffsetGuess (float):
            Theta rotation (in degrees) of spot locations on the camera, which might be different
            from expected due to clocking error between the DM and the camera.
        satellite_spot_parameters (dict, optional):
            Dictionary containing tuning parameters for spot separation and offset estimation. The dictionary
            should have the following structure:

            offset : dict
                Parameters for estimating the offset of the star center:

                spotSepPix : float
                    Expected (model-based) separation of the satellite spots from the star.
                    Units: pixels.
                roiRadiusPix : float
                    Radius of the region of interest around each satellite spot.
                    Units: pixels.
                probeRotVecDeg : array_like
                    Angles (degrees CCW from x-axis) specifying the position of satellite spot pairs.
                nSubpixels : int
                    Number of subpixels across for calculating region-of-interest mask edges.
                nSteps : int
                    Number of points in grid search along each direction.
                stepSize : float
                    Step size for the grid search.
                    Units: pixels.
                nIter : int
                    Number of iterations refining the radial separation.

            separation : dict
                Parameters for estimating the separation of satellite spots from the star:

                spotSepPix : float
                    Expected separation between star and satellite spots.
                    Units: pixels.
                roiRadiusPix : float
                    Radius of the region of interest around each satellite spot.
                    Units: pixels.
                probeRotVecDeg : array_like
                    Angles (degrees CCW from x-axis) specifying the position of satellite spot pairs.
                nSubpixels : int
                    Number of subpixels across for calculating region-of-interest mask edges.
                nSteps : int
                    Number of points in grid search along each direction.
                stepSize : float
                    Step size for the grid search.
                    Units: pixels.
                nIter : int
                    Number of iterations refining the radial separation.

    Returns:
        numpy.ndarray:
            Estimated absolute coordinates [x, y] of the star center in the spots image.
        numpy.ndarray:
            Calculated locations of the satellite spots.
    """

    # check inputs
    img_shp = img_ref.shape
    if img_sat_spot.shape != img_shp:
        raise TypeError("Satellite spot image not the same shape as the science image.")

    # Unpack the star guess and calculate offsets relative to image center
    xGuess, yGuess = star_coordinate_guess
    img_center_x = img_sat_spot.shape[1] // 2
    img_center_y = img_sat_spot.shape[0] // 2
    xOffsetGuess = xGuess - img_center_x
    yOffsetGuess = yGuess - img_center_y

    # Subtract reference image from satellite spot image to highlight satellite spots
    img_spots = img_sat_spot - img_ref

    # Grab the relevant tuning parameters
    tuningParamDict = satellite_spot_parameters

    # estimate star location:
    # calc_star_location_from_spots should return (xOffsetEst, yOffsetEst)
    xOffsetEst, yOffsetEst = calc_star_location_from_spots(
        img_spots, xOffsetGuess, yOffsetGuess, tuningParamDict['offset']
    )

    # estimate spot separation (from the star, i.e., radius)
    spotRadiusEst = calc_spot_separation(
        img_spots, xOffsetEst, yOffsetEst, tuningParamDict['separation']
    )

    # get spot rotation vector (angles in degrees)
    probeRotVecDeg = tuningParamDict['separation']["probeRotVecDeg"]

    # calculate locations of spot pairs
    list_spots_xy = []
    for theta_deg in probeRotVecDeg:
        theta = (theta_deg + thetaOffsetGuess) * np.pi / 180.0
        # One spot
        list_spots_xy.append([
            spotRadiusEst * np.cos(theta) + xOffsetEst,
            spotRadiusEst * np.sin(theta) + yOffsetEst,
        ])
        # Opposite spot (theta + pi)
        list_spots_xy.append([
            spotRadiusEst * np.cos(theta + np.pi) + xOffsetEst,
            spotRadiusEst * np.sin(theta + np.pi) + yOffsetEst,
        ])

    # Convert estimated offsets back to absolute coordinates
    star_xy = np.array([
        xOffsetEst + img_center_x,
        yOffsetEst + img_center_y
    ], dtype='float')

    list_spots_xy = np.array(list_spots_xy, dtype='float')

    return star_xy, list_spots_xy
