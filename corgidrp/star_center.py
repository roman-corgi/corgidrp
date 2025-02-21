import numpy as np
from scipy.optimize import minimize

satellite_spot_parameters = {
    "NFOV": {
        "offset": {
            "spotSepPix": 14.79,
            "roiRadiusPix": 4.5,
            "probeRotVecDeg": [0, 90],
            "nSubpixels": 100,
            "nSteps": 7,
            "stepSize": 1,
            "nIter": 6,
        },
        "separation": {
            "spotSepPix": 14.79,
            "roiRadiusPix": 1.5,
            "probeRotVecDeg": [0, 90],
            "nSubpixels": 100,
            "nSteps": 21,
            "stepSize": 0.25,
            "nIter": 5,
        }
    }
}

def circle(nx, ny, roiRadiusPix, xShear, yShear, nSubpixels=100):
    """
    Generate a circular aperture with an antialiased edge at specified offsets.

    Used as a software window for isolating a region of interest. Grayscale
    edges are used because the detector sampling may be low enough that
    fractional values along the edges are important.

    Parameters
    ----------
    nx, ny : array_like
        Dimensions of the 2-D array to create.
    roiRadiusPix : float
        Radius of the circle in pixels.
    xShear, yShear : float
        Lateral offsets in pixels of the circle's center from the array's
        center pixel.
    nSubpixels : int, optional
        Each edge pixel of the circle is subdivided into a square subarray
        nSubpixels across. The subarray is given binary values and then
        averaged to give the edge pixel a value between 0 and 1, inclusive.
        The default value is 100. Must be a positive scalar integer.

    Returns
    -------
    mask : numpy ndarray
        2-D array containing the circle
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
    Fit a parabola to a 1-D array and return the location of the min or max.

    Parameters
    ----------
    xVec : array_like
        1-D array of coordinate values for the values in arrayToFit.
    arrayToFit : array_like
        1-D array of values to fit.

    Returns
    -------
    xOpt : float
        Best fit value for the optimum (i.e., min or max) of the parabola.
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
    Fit a paraboloid to a 2-D array and return the location of the min or max.

    Parameters
    ----------
    xVec : array_like
        1-D array of coordinate values along axis 1 of arrayToFit.
    yVec : array_like
        1-D array of coordinate values along axis 0 of arrayToFit.
    arrayToFit : array_like
        2-D array of values to fit.
    mask : array_like
        2-D boolean mask of which pixels to use. Same shape as arrayToFit.

    Returns
    -------
    xBest : float
        Best fit value along axis 1.
    yBest : float
        Best fit value along axis 0.

    Notes
    -----
    Modified from code at
    https://au.mathworks.com/matlabcentral/answers/5482-fit-a-3d-curve
    and equations at
    https://math.stackexchange.com/questions/2010758/how-do-i-fit-a-paraboloid-surface-to-nine-points-and-find-the-minimum
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
    Compute the stellar offset and spot separation simultaneously.

    Parameters
    ----------
    optim_params : list
        List of the parameters to be optimized by scipy.optimize.minimize().
        They must be these 3 variables in this exact order:
            [xOffsetGuess, yOffsetGuess, spotSepGuessPix]
            xOffsetGuess, yOffsetGuess are the starting guesses for the star
            offsets from the array center pixel. Units of pixels
            spotSepGuessPix is the starting guess for the spot separation from
            the star center in units of pixels.
    fixed_params : list
        List of the fixed parameters used when computing the cost function.
        They must be these 4 variables in this exact order:
            [spotArray, probeRotVecDeg, roiRadiusPix, nSubpixels]
            spotArray is the 2-D array of the DM-generated satellite spots.
            Calculated outside this function from probed images as
            (Iplus + Iminus)/2 - Iunprobed.
            probeRotVecDeg is the 1-D array of how many degrees
            counterclockwise from the x-axis to rotate the regions of interest
            used when summing the satellite spots. Note that a pair of
            satellite spots is given by just one value. For example, for a
            single pair of satellite spots along the x-axis use [0, ] and not
            [0, 180]. And for a plus-shaped layout of spots, use [0, 90].
            roiRadiusPix is the radius of each region of interest used when
            summing the intensity of a satellite spot. Units of pixels.
            nSubpixels is the number of subpixels across used to make edge
            values of the region-of-interest mask. The value of the edge
            pixels in the ROI is the mean of all the subpixel values.

    Returns
    -------
    cost : float
        The summed total intensity getting through the region-of-interest mask
        multiplied by -1. This is used as the cost to be minimized by
        scipy.optimize.minimize()..
    roi_mask : numpy ndarray
        2-D array with values between 0 and 1, inclusive, telling how much
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
    """
    cost, _ = _roi_mask_for_spots(optim_params, fixed_params)

    return cost


def calc_star_location_and_spot_separation(
    spotArray, xOffsetGuess, yOffsetGuess, tuningParamDict
):
    """
    Calculate the center of the occulted star using satellite spots.

    Just one processed image of satellite spots is used. A multitude of
    software masks are generated and applied to the measured spots in order to
    determine the stellar location. The best estimate of the star location is
    the one that maximizes the total summed energy, as found by a 2-D quadratic
    fit.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Parameters
    ----------
    spotArray : numpy ndarray
        2-D array of the DM-generated satellite spots. Calculated outside
        this function from probed images as (Iplus + Iminus)/2 - Iunprobed.
    xOffsetGuess, yOffsetGuess : float
        Starting guess for the number of pixels in x and y that the star is
        offset from the center pixel of the array spotArray. The convention
        for the center pixel follows that of FFTs.
    tuningParamDict : dict
            Dictionary containing the tuning parameter values.
            The dictionary should contain the following keys:
            - spotSepPix : float
                Expected separation of the satellite spots from the star. Used as the
                separation for the center of the region of interest. Units of pixels.
                Compute beforehand as sep in lambda/D and multiply by pix per lambda/D.
            - roiRadiusPix : float
                Radius of each region of interest used when summing the intensity of a
                satellite spot. Units of pixels.
            - probeRotVecDeg : array_like
                1-D array of how many degrees counterclockwise from the x-axis to
                rotate the regions of interest used when summing the satellite spots.
                Note that a pair of satellite spots is given by just one value. For
                example, for a single pair of satellite spots along the x-axis use [0,]
                and [0, 180]. And for a plus-shaped layout of spots, use [0, 90].
            - nSubpixels : int
                Number of subpixels across used to make edge values of the region-of-
                interest mask. The value of the edge pixels in the ROI is the mean of
                all the subpixel values.
            - nSteps : int
                Number of points used along each direction for the grid search.
                Odd numbers are better to provide symmetry of values when the array is
                truly centered.
            - stepSize : float
                The step size used in the grid search. Units of pixels.
            - nIter : int
                Number of iterations in the loop that hones in on the stellar center
                location.

    Returns
    -------
    xOffsetEst, yOffsetEst : float
        Estimated lateral offsets of the stellar center from the center pixel
        of the array spotArray. The convention for the center pixel follows
        that of FFTs.
    roi_mask : numpy ndarray
        2-D float array of the best-fit region-of-interest mask used to fit
        the translation and scaling of the ROI regions to spotArray.
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
    Calculate the center of the occulted star using satellite spots.

    Just one processed image of satellite spots is used. A multitude of
    software masks are generated and applied to the measured spots in order to
    determine the stellar location. The best estimate of the star location is
    the one that maximizes the total summed energy, as found by a 2-D quadratic
    fit.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Parameters
    ----------
    spotArray : numpy ndarray
        2-D array of the DM-generated satellite spots. Calculated outside
        this function from probed images as (Iplus + Iminus)/2 - Iunprobed.
    xOffsetGuess, yOffsetGuess : float
        Starting guess for the number of pixels in x and y that the star is
        offset from the center pixel of the array spotArray. The convention
        for the center pixel follows that of FFTs.
    tuningParamDict : dict
        Dictionary containing the tuning parameter values.
        The dictionary should contain the following keys:
        - spotSepPix : float
            Expected separation of the satellite spots from the star. Used as the
            separation for the center of the region of interest. Units of pixels.
            Compute beforehand as sep in lambda/D and multiply by pix per lambda/D.
        - roiRadiusPix : float
            Radius of each region of interest used when summing the intensity of a
            satellite spot. Units of pixels.
        - probeRotVecDeg : array_like
            1-D array of how many degrees counterclockwise from the x-axis to
            rotate the regions of interest used when summing the satellite spots.
            Note that a pair of satellite spots is given by just one value. For
            example, for a single pair of satellite spots along the x-axis use [0,]
            and [0, 180]. And for a plus-shaped layout of spots, use [0, 90].
        - nSubpixels : int
            Number of subpixels across used to make edge values of the region-of-
            interest mask. The value of the edge pixels in the ROI is the mean of
            all the subpixel values.
        - nSteps : int
            Number of points used along each direction for the grid search.
            Odd numbers are better to provide symmetry of values when the array is
            truly centered.
        - stepSize : float
            The step size used in the grid search. Units of pixels.
        - nIter : int
            Number of iterations in the loop that hones in on the stellar center
            location.

    Returns
    -------
    xOffsetEst, yOffsetEst : float
        Estimated lateral offsets of the stellar center from the center pixel
        of the array spotArray. The convention for the center pixel follows
        that of FFTs.
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
    Calculate the radial separation in pixels of the satellite spots.

    Just one processed image of satellite spots is used. Several software
    masks are generated and applied to the measured spots in order to
    determine the radial separation of the spots from the given star center.
    The best estimate of the radial spot separation is the one that maximizes
    the total summed energy in the software mask, as found by a 1-D quadratic
    fit.

    All filenames may be absolute or relative paths.  If relative, they will be
    relative to the current working directory, not to any particular location
    in Calibration.

    Parameters
    ----------
    spotArray : numpy ndarray
        2-D array of the DM-generated satellite spots. Calculated outside
        this function from probed images as (Iplus + Iminus)/2 - Iunprobed.
    xOffset, yOffset : float
        Previously estimated stellar center offset from the array's center
        pixel. Units of pixels. The convention for the center pixel follows
        that of FFTs.
    tuningParamDict : dict
        Dictionary containing the tuning parameter values.
        - spotSepPix : float
            Expected (i.e., model-based) separation of the satellite spots from the
            star. Used as the starting point for the separation for the center of
            the region of interest. Units of pixels. Compute beforehand as
            separation in lambda/D multiplied by pixels per lambda/D.
            6.5*(51.46*0.575/13)
        - roiRadiusPix : float
            Radius of each region of interest used when summing the intensity of a
            satellite spot. Units of pixels.
        - probeRotVecDeg : array_like
            1-D array of how many degrees counterclockwise from the x-axis to
            rotate the regions of interest used when summing the satellite spots.
            Note that a pair of satellite spots is given by just one value. For
            example, for a single pair of satellite spots along the x-axis use
            [0, ] and not [0, 180]. And for a plus-shaped layout of spots,
            use [0, 90].
        - nSubpixels : int
            Number of subpixels across used to make edge values of the region-of-
            interest mask. The value of the edge pixels in the ROI is the mean of
            all the subpixel values.
        - nSteps : int
            Number of points used along each direction for the grid search.
            Odd numbers are better to provide symmetry of values when the array is
            truly centered.
        - stepSize : float
            The step size used in the grid search. Units of pixels.
        - nIter : int
            Number of iterations in the loop that hones in on the radial separation
            of the satellite spots.

    Returns
    -------
    spotSepEst : float
        Estimated radial separation of the satellite spots from the stellar
        center. Units of pixels.
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
    img_plus,
    img_minus,
    xOffsetGuess,
    yOffsetGuess,
    thetaOffsetGuess,
    satellite_spot_parameters,
    observing_mode='NFOV',
):
    """
    Estimate the star center and spot locations from satellite spot images and science data.

    Parameters
    ----------
    img_ref : numpy ndarray
        2-D image representing a clean occulted focal-plane image with a base DM setting.
    img_plus : numpy ndarray
        2-D image representing a clean occulted focal-plane image with a relative satellite-spot DM setting added.
    img_minus : numpy ndarray
        2-D image representing a clean occulted focal-plane image with the same relative DM setting satellite-spot subtracted.
    xOffsetGuess, yOffsetGuess : float
        Starting guess for the number of pixels in x and y that the star is offset from the center pixel of the spots image.
    thetaOffsetGuess : float (degrees)
        Theta rotation of spot locations on the camera, which might be different from expected due to clocking error between the DM and the camera.
    satellite_spot_parameters : dict
        Dictionary containing tuning parameters for spot separation and offset estimation.
    mode : str, optional
        Mode for selecting the satellite spot parameters from the `satellite_spot_parameters` dictionary.

    Returns
    -------
    star_xy : numpy ndarray
        Estimated lateral offsets of the stellar center from the center pixel of the spots image.
    list_spots_xy : numpy ndarray
        List of spot locations.

    Notes
    -----
    Offset Tuning parameters in the satellite_spot_parameters dictionary are explained below:

    spotSepPix : float
        Expected (i.e., model-based) separation of the satellite spots from the
        star. Used as the starting point for the separation for the center of
        the region of interest. Units of pixels. Compute beforehand as
        separation in lambda/D multiplied by pixels per lambda/D.
        6.5*(51.46*0.575/13)
    roiRadiusPix : float
        Radius of each region of interest used when summing the intensity of a
        satellite spot. Units of pixels.
    probeRotVecDeg : array_like
        1-D array of how many degrees counterclockwise from the x-axis to
        rotate the regions of interest used when summing the satellite spots.
        Note that a pair of satellite spots is given by just one value. For
        example, for a single pair of satellite spots along the x-axis use
        [0, ] and not [0, 180]. And for a plus-shaped layout of spots,
        use [0, 90].
    nSubpixels : int
        Number of subpixels across used to make edge values of the region-of-
        interest mask. The value of the edge pixels in the ROI is the mean of
        all the subpixel values.
    nSteps : int
        Number of points used along each direction for the grid search.
        Odd numbers are better to provide symmetry of values when the array is
        truly centered.
    stepSize : float
        The step size used in the grid search. Units of pixels.
    nIter : int
        Number of iterations in the loop that hones in on the radial separation
        of the satellite spots.

    Separation Tuning parameters in the satellite_spot_parameters dictionary are explained
    below:

    spotSepGuessPix : float
        Expected (i.e., model-based) separation of the satellite spots from the
        star. Used as the starting point for the separation for the center of
        the region of interest. Units of pixels. Compute beforehand as
        separation in lambda/D multiplied by pixels per lambda/D.
        6.5*(51.46*0.575/13)
    roiRadiusPix : float
        Radius of each region of interest used when summing the intensity of a
        satellite spot. Units of pixels.
    probeRotVecDeg : array_like
        1-D array of how many degrees counterclockwise from the x-axis to
        rotate the regions of interest used when summing the satellite spots.
        Note that a pair of satellite spots is given by just one value. For
        example, for a single pair of satellite spots along the x-axis use
        [0, ] and not [0, 180]. And for a plus-shaped layout of spots,
        use [0, 90].
    nSubpixels : int
        Number of subpixels across used to make edge values of the region-of-
        interest mask. The value of the edge pixels in the ROI is the mean of
        all the subpixel values.
    nSteps : int
        Number of points used along each direction for the grid search.
        Odd numbers are better to provide symmetry of values when the array is
        truly centered.
    stepSize : float
        The step size used in the grid search. Units of pixels.
    nIter : int
        Number of iterations in the loop that hones in on the radial separation
        of the satellite spots.

    """

    # check inputs
    img_shp = img_ref.shape
    if img_plus.shape != img_shp:
        raise TypeError("img_plus not same shape as img_ref")
    if img_minus.shape != img_shp:
        raise TypeError("img_minus not same shape as img_ref")

    # combine input images to create image with satellite spots
    img_spots = 0.5 * (img_plus + img_minus) - img_ref

    tuningParamDict = satellite_spot_parameters[observing_mode]
    # estimate star location
    # xOffsetEst, yOffsetEst are relative to image center, fft style
    # was using fn_offset_YAML
    xOffsetEst, yOffsetEst = calc_star_location_from_spots(
        img_spots, xOffsetGuess, yOffsetGuess, tuningParamDict['offset']
    )

    # estimate spot separation (from the star, i.e. radius)
    # was using fn_separation_YAML
    spotRadiusEst = calc_spot_separation(
        img_spots, xOffsetEst, yOffsetEst, tuningParamDict['separation']
    )

    # get spot theta values from input YAML
    # tuningParamDict = loadyaml(fn_separation_YAML)
    probeRotVecDeg = tuningParamDict['separation']["probeRotVecDeg"]
    # check.oneD_array(probeRotVecDeg, 'probeRotVecDeg',
    #                  TypeError) # check is redundant

    # calculate locations of spot pairs
    list_spots_xy = []
    for theta_deg in probeRotVecDeg:
        theta = (theta_deg + thetaOffsetGuess) * np.pi / 180.0
        list_spots_xy.append(
            [
                spotRadiusEst * np.cos(theta) + xOffsetEst,
                spotRadiusEst * np.sin(theta) + yOffsetEst,
            ],
        )
        list_spots_xy.append(
            [
                spotRadiusEst * np.cos(theta + np.pi) + xOffsetEst,
                spotRadiusEst * np.sin(theta + np.pi) + yOffsetEst,
            ],
        )

    star_xy = [xOffsetEst, yOffsetEst]

    return np.array(star_xy, dtype='float'), np.array(list_spots_xy, dtype='float')