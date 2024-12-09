# Functions related with Core Throughout and off-axis PSF

# CTC requirements
"""
1090881 - Given a core throughput dataset consisting of M clean frames 
(nominally 1024x1024) taken at different FSM positions, the CTC GSW shall
estimate the pixel location and core throughput of each PSF.

NOTE: the list of M clean frames may be a subset of the frames collected during
core throughput data collection, to allow for the removal of outliers.

1090882 - Given 1) the location of the center of the FPM coronagraphic mask in
EXCAM pixels during the coronagraphic observing sequence and 2) the FPAM and
FSAM encoder positions during both the coronagraphic and core throughput observing
sequences, the CTC GSW shall compute the center of the FPM coronagraphic mask
during the core throughput observing sequence. 

1090883 - Given 1) an array of PSF pixel locations and 2) the location of the
center of the FPAM coronagraphic mask in EXCAM pixels during core throughput
calibrations, and 3) corresponding core throughputs for each PSF, the CTC GSW
shall compute a 2D floating-point interpolated core throughput map.

1090884 - Given 1) a core throughput dataset consisting of a set of clean frames
(nominally 1024x1024) taken at different FSM positions, and 2) a list of N (x, y)
coordinates, in units of EXCAM pixels, which fall within the area covered by the
core throughput dataset, the CTC GSW shall produce a 1024x1024xN cube of PSF
images best centered at each set of coordinates.

1077737 - Given one or more level L3 EXCAM data frames of target and/or reference
stars at one or more roll angles intended for PSF-subtraction, CTC GSW shall
apply the "L3 → L4" pipeline for "TTR5 Analog PSF-subtracted on-sky data" as
described the table "Imaging Data Processing Pipelines" in D-105748 TDD FDD.

Note: Any data analysis marked above as "Same as L3 → L4 pipeline used for TTR5
Analog PSF-subtracted on-sky data"" will also use this pipeline.
See: https://wiki.jpl.nasa.gov/display/CGIfocus8/13-TDD%3A+Tech+Demo+Data+FDD

"""


def estimate_psf_pix_and_ct(
    dataset_in,
    fsm_pos):
    """
    1090881 - Given a core throughput dataset consisting of M clean frames
    (nominally 1024x1024) taken at different FSM positions, the CTC GSW shall
    estimate the pixel location and core throughput of each PSF.

    NOTE: the list of M clean frames may be a subset of the frames collected during
    core throughput data collection, to allow for the removal of outliers.

    Args:
      dataset_in (corgidrp.Dataset): A core throughput dataset consisting of
        M clean frames (nominally 1024x1024) taken at different FSM positions.
        Units: photoelectrons / second / pixel.

        The first PSF must be the unocculted PSF.

      fsm_pos (array): Array with FSM positions. Units: TBD
        

    Returns:
      psf_pix (array): Array with PSF's pixel positions. Units: EXCAM pixels
        referred to the (0,0) pixel.

      psf_ct (array): Array with PSF's core throughput values. Units:
        dimensionless (Values must be within 0 and 1).
    """
    dataset = dataset_in.copy()

    # check that the first psf is unocculted (by checking max value across all psfs)
 

    return psf_pix, psf_ct

