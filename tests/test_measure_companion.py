import pytest
import os
import numpy as np
import corgidrp
from corgidrp.mocks import create_default_headers
from corgidrp.mocks import create_flux_image
from corgidrp.data import Image, Dataset, FluxcalFactor
import corgidrp.fluxcal as fluxcal
import corgidrp.l4_to_tda as l4_to_tda
from astropy.modeling.models import BlackBody
import astropy.units as u

