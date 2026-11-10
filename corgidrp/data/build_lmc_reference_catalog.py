"""Build the LMC astrometric reference catalogue lmc_astrometric_reference.ecsv.

Stand-alone developer script, not part of the pipeline: run it when the catalogue has to
change (e.g. for a new Gaia release). It is kept next to the catalogue it produces so the two
are versioned together; corgidrp/data is not an importable package (corgidrp.data is the
data.py module), so this file is never imported by the pipeline.

The catalogue merges Jay Anderson's HST catalogue of the JWST calibration field in the LMC
(JWST-STScI-007716) with the Gaia DR3 stars HST did not measure (mostly stars that saturated
in the deep HST exposures), at the common epoch J2015.5, and records origin, selection and
self-checks in the ECSV metadata.

    python corgidrp/data/build_lmc_reference_catalog.py --gaia-file /some/scratch/gaia.ecsv

Needs network access and astroquery (not a corgidrp dependency) unless both input files are
given with --anderson-file and --gaia-file. A Gaia query result is saved to --gaia-file on
first use and reused afterwards, so a rebuild does not depend on the Gaia archive.
"""
import argparse
import datetime
import hashlib
import os
import tempfile
import urllib.request
import warnings

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import Table, vstack
from astropy.time import Time
from erfa import ErfaWarning

import corgidrp

DEFAULT_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lmc_astrometric_reference.ecsv")

ANDERSON_URL = "https://www.stsci.edu/~jayander/JWST_CALFIELD2020/master_output.xymvrdeepmeehuGAIA_DR3"
ANDERSON_REFERENCE = ("Anderson, J., Fall, M., and the Astrometry Working Group 2021, JWST-STScI-007716, "
                           "The JWST Calibration Field: Absolute Astrometry and Proper Motions with GAIA "
                           "and a Second HST Epoch")
GAIA_REFERENCE = "Gaia Collaboration, Vallenari, A., et al. 2023, A&A, 674, A1 (Gaia DR3)"
GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap"

FIELD_CENTER = (80.5, -69.5)   # deg, centre of Anderson's reference frame
GAIA_CONE_RADIUS = 0.1         # deg, covers the whole HST footprint
CATALOG_EPOCH = 2015.5         # Julian year, TCB
GAIA_EPOCH = 2016.0            # Julian year, TCB
HST_MAG_LIMIT = 22.5           # F606W VEGAMAG, same cut as the former JWST_CALFIELD2020.csv
GAIA_MAG_LIMIT = 19.5          # Gaia G; fainter missing stars are mostly blends or poorly measured
DUPLICATE_RADIUS = 0.1         # arcsec
WIDE_DUPLICATE_RADIUS = 0.3    # arcsec, for Gaia stars without proper motion (see near_hst_star)
DUPLICATE_MAG_DIFF = 0.5       # mag, G against F606W
FOOTPRINT_RADIUS = 5.0         # arcsec
FOOTPRINT_MIN_STARS = 10
RUWE_LIMIT = 1.4
MAX_FRAME_OFFSET_MAS = 0.2
SYNC_ROW_LIMIT = 2000          # Gaia archive cap on synchronous query results

FLAG_PM_ASSUMED = 1
FLAG_HIGH_RUWE = 2
FLAG_GAIA_DUPLICATE = 4

GAIA_COLUMNS = ["source_id", "ra", "dec", "ra_error", "dec_error", "pmra", "pmra_error",
                     "pmdec", "pmdec_error", "parallax", "parallax_error", "phot_g_mean_mag",
                     "ruwe", "astrometric_params_solved", "duplicated_source"]

COLUMNS = {   # name: (unit, description)
    "GAIA_ID": (None, "Gaia DR3 source_id, 0 if the star has no Gaia counterpart"),
    "RA": ("deg", "Right ascension, ICRS, epoch J2015.5"),
    "DEC": ("deg", "Declination, ICRS, epoch J2015.5"),
    "RA_ERR": ("mas", "Uncertainty of RA on the sky (times cos dec) at J2015.5"),
    "DEC_ERR": ("mas", "Uncertainty of DEC at J2015.5"),
    "VMAG": ("mag", "Approximately V-band magnitude used for brightness ranking: HST F606W "
                      "(VEGAMAG) or Gaia G, as given in MAG_BAND"),
    "MAG_BAND": (None, "Band of VMAG: F606W (HST ACS/WFC, VEGAMAG) or G (Gaia DR3)"),
    "GMAG": ("mag", "Gaia DR3 G magnitude, NaN without Gaia counterpart"),
    "PMRA": ("mas / yr", "Proper motion in RA, mu_alpha* = mu_alpha cos(dec)"),
    "PMDEC": ("mas / yr", "Proper motion in Dec"),
    "PMRA_ERR": ("mas / yr", "Uncertainty of PMRA"),
    "PMDEC_ERR": ("mas / yr", "Uncertainty of PMDEC"),
    "PARALLAX": ("mas", "Gaia DR3 parallax, NaN if not available"),
    "PARALLAX_ERR": ("mas", "Uncertainty of PARALLAX"),
    "RUWE": (None, "Gaia DR3 renormalised unit weight error, NaN if not available"),
    "POS_SOURCE": (None, "Source of RA/DEC: HST or GAIA"),
    "PM_SOURCE": (None, "Source of the proper motion: HST, GAIA, or BULK (LMC bulk motion assumed)"),
    "FLAG": (None, "Bitmask: 1 proper motion assumed, 2 RUWE > 1.4, 4 Gaia duplicated_source"),
}


def build_catalog(output_path=DEFAULT_OUTPUT, anderson_file=None, gaia_file=None):
    """
    Build the LMC astrometric reference catalogue used by boresight_calibration.

    Merges Anderson's HST catalogue of the JWST calibration field in the LMC (JWST-STScI-007716)
    with the Gaia DR3 stars HST did not measure (mostly stars that saturated in the deep HST
    exposures), at the common epoch J2015.5, and records origin and selection in the ECSV
    metadata. The catalogue is checked after writing and only then moved to output_path, so a
    failed build never replaces an existing catalogue.

    Args:
        output_path (str): where to write the ECSV catalogue
        anderson_file (str): local copy of Anderson's master_output.xymvrdeepmeehuGAIA_DR3;
            downloaded if None
        gaia_file (str): ECSV with a saved Gaia query; queried and written if it does not exist,
            queried and not saved if None

    Returns:
        astropy.table.Table: the catalogue as written

    Raises:
        RuntimeError: if a Gaia query fails or the catalogue fails its self-checks
    """

    with tempfile.TemporaryDirectory() as scratch:
        if anderson_file is None:
            anderson_file = os.path.join(scratch, os.path.basename(ANDERSON_URL))
            urllib.request.urlretrieve(ANDERSON_URL, anderson_file)
        with open(anderson_file, "rb") as handle:
            anderson_sha256 = hashlib.sha256(handle.read()).hexdigest()
        hst_all = read_anderson(anderson_file)

    bulk = bulk_motion(hst_all[np.asarray(hst_all["VMAG"], float) < HST_MAG_LIMIT])

    if gaia_file is not None and os.path.exists(gaia_file):
        gaia = Table.read(gaia_file, format="ascii.ecsv")
    else:
        gaia = query_gaia()
        if gaia_file is not None:
            gaia.write(gaia_file, format="ascii.ecsv")

    gaia_coords = gaia_at_epoch(gaia, bulk)
    catalog = assemble(hst_all, gaia, gaia_coords, bulk)
    offset = frame_offset(catalog, gaia, gaia_coords)
    catalog.sort(["VMAG", "GAIA_ID", "RA"], kind="stable")  # ties broken, so row order is reproducible
    for name, (unit, description) in COLUMNS.items():
        catalog[name].unit = unit
        catalog[name].description = description
    catalog.meta = metadata(catalog, anderson_sha256, gaia.meta, bulk, offset, datetime.date.today().isoformat())

    linked_ids = np.asarray(hst_all["GAIA_ID"], np.int64)
    temporary = output_path + ".tmp"
    catalog.write(temporary, format="ascii.ecsv", overwrite=True)
    try:
        self_checks(Table.read(temporary, format="ascii.ecsv"), linked_ids[linked_ids != 0],
                    gaia["source_id"], offset)
    except RuntimeError:
        os.remove(temporary)
        raise
    os.replace(temporary, output_path)
    return catalog


def read_anderson(path):
    """
    Read Anderson's master_output.xymvrdeepmeehuGAIA_DR3 isolated-star catalogue.

    Columns used (1-based, as in the file header): 4 F606W VEGAMAG, 5/6 RA/Dec at 2015.5 [deg],
    7/8 position errors [mas], 9/10 proper motions [mas/yr], 11/12 their errors,
    34 Gaia DR3 source_id (0 if none).

    Args:
        path (str): path to the master file

    Returns:
        astropy.table.Table: columns MAG, RA, DEC, RA_ERR, DEC_ERR, PMRA, PMDEC, PMRA_ERR,
            PMDEC_ERR, GAIA_ID
    """

    values = np.loadtxt(path, comments="#", usecols=(3, 4, 5, 6, 7, 8, 9, 10, 11))
    table = Table(values, names=["VMAG", "RA", "DEC", "RA_ERR", "DEC_ERR", "PMRA", "PMDEC",
                                 "PMRA_ERR", "PMDEC_ERR"])
    table["GAIA_ID"] = np.loadtxt(path, comments="#", usecols=(33,), dtype=np.int64)
    return table


def bulk_motion(hst):
    """
    LMC bulk proper motion (medians) and internal dispersion (mean robust std of both axes).

    Args:
        hst (astropy.table.Table): HST rows with PMRA, PMDEC [mas/yr]

    Returns:
        dict: pmra, pmdec, dispersion [mas/yr]
    """
    pmra = np.asarray(hst["PMRA"], float)
    pmdec = np.asarray(hst["PMDEC"], float)

    def robust_std(values):
        return 1.4826 * np.median(np.abs(values - np.median(values)))

    return {"pmra": float(np.median(pmra)), "pmdec": float(np.median(pmdec)),
            "dispersion": float(0.5 * (robust_std(pmra) + robust_std(pmdec)))}


def cone_tiles(center, radius, n_per_side):
    """
    Half-open RA/Dec boxes [ra_min, ra_max) x [dec_min, dec_max) that cover a cone.

    A star on a shared edge belongs to exactly one box; the outer edges are padded slightly.

    Args:
        center (tuple): (ra, dec) of the cone centre [deg]
        radius (float): cone radius [deg]
        n_per_side (int): number of boxes along RA and along Dec

    Returns:
        list of tuple: (ra_min, ra_max, dec_min, dec_max) per box [deg]
    """
    pad = 1e-6
    dec_half = radius + pad
    # widest RA extent is at the declination closest to the pole
    ra_half = radius / np.cos(np.radians(min(abs(center[1]) + radius, 89.9))) + pad
    ra_edges = np.linspace(center[0] - ra_half, center[0] + ra_half, n_per_side + 1)
    dec_edges = np.linspace(center[1] - dec_half, center[1] + dec_half, n_per_side + 1)
    return [(ra_edges[i], ra_edges[i + 1], dec_edges[j], dec_edges[j + 1])
            for i in range(n_per_side) for j in range(n_per_side)]


def query_gaia(attempts=3):
    """
    Query Gaia DR3 for all sources in the cone around the field.

    The archive caps synchronous queries at SYNC_ROW_LIMIT rows without warning, and its
    asynchronous queue can stall for a long time. So the cone is split into tiles small enough
    for synchronous queries; a tile at the cap, or a total that differs from COUNT(*) of the
    cone, raises.

    Args:
        attempts (int): number of tries per query (the archive is sometimes unstable)

    Returns:
        astropy.table.Table: GAIA_COLUMNS with NaN for missing values; meta holds the ADQL
            text of the whole cone and the query date
    """
    from astroquery.gaia import Gaia   # only needed to rebuild the catalogue

    def run(adql):
        error = None
        for _ in range(attempts):
            try:
                return Gaia.launch_job(adql).get_results()
            except Exception as failure:   # network or archive failure: retry
                error = failure
        raise RuntimeError("Gaia query failed {0} times: {1}\n{2}".format(attempts, error, adql))

    center, radius = FIELD_CENTER, GAIA_CONE_RADIUS
    cone = "1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {0}, {1}, {2}))".format(center[0], center[1], radius)
    adql = "SELECT " + ", ".join(GAIA_COLUMNS) + " FROM gaiadr3.gaia_source WHERE " + cone
    expected = int(run("SELECT COUNT(*) AS n FROM gaiadr3.gaia_source WHERE " + cone)["n"][0])

    # aim for about a third of the row cap per tile (stars are not spread evenly)
    n_per_side = int(np.ceil(np.sqrt(3.0 * expected / SYNC_ROW_LIMIT))) + 1
    pieces = []
    for ra_min, ra_max, dec_min, dec_max in cone_tiles(center, radius, n_per_side):
        piece = run(adql + " AND ra >= {0!r} AND ra < {1!r} AND dec >= {2!r} AND dec < {3!r}".format(
            float(ra_min), float(ra_max), float(dec_min), float(dec_max)))
        if len(piece) >= SYNC_ROW_LIMIT:
            raise RuntimeError("Gaia tile hit the {0}-row synchronous limit".format(SYNC_ROW_LIMIT))
        pieces.append(piece)
    result = vstack(pieces)
    if len(result) != expected:
        raise RuntimeError("Gaia query returned {0} rows, COUNT(*) says {1}".format(len(result), expected))

    gaia = Table(meta={"adql": adql, "queried": datetime.date.today().isoformat()})
    for name in GAIA_COLUMNS:
        if name in ("source_id", "astrometric_params_solved"):
            gaia[name] = np.asarray(result[name]).astype(np.int64)
        elif name == "duplicated_source":
            gaia[name] = np.ma.filled(np.ma.asarray(result[name]), False).astype(bool)
        else:
            gaia[name] = np.ma.filled(np.ma.asarray(result[name], dtype=float), np.nan)
    return gaia


def gaia_at_epoch(gaia, bulk):
    """
    Move Gaia DR3 positions from J2016.0 to the catalogue epoch J2015.5.

    Stars without a Gaia proper motion (two-parameter solutions) are moved with the LMC bulk
    motion, the same motion the catalogue assigns them.

    Args:
        gaia (astropy.table.Table): Gaia table (see query_gaia)
        bulk (dict): output of bulk_motion

    Returns:
        astropy.coordinates.SkyCoord: ICRS positions at J2015.5, in the order of gaia
    """

    pmra, pmdec, known = gaia_pm(gaia)
    coords = SkyCoord(ra=np.asarray(gaia["ra"], float) * u.deg, dec=np.asarray(gaia["dec"], float) * u.deg,
                      pm_ra_cosdec=np.where(known, pmra, bulk["pmra"]) * u.mas / u.yr,
                      pm_dec=np.where(known, pmdec, bulk["pmdec"]) * u.mas / u.yr,
                      obstime=Time(GAIA_EPOCH, format="jyear", scale="tcb"), frame="icrs")
    with warnings.catch_warnings():
        # without distances ERFA assumes very distant stars and says so; harmless here
        warnings.simplefilter("ignore", ErfaWarning)
        moved = coords.apply_space_motion(new_obstime=Time(CATALOG_EPOCH, format="jyear", scale="tcb"))
    return SkyCoord(ra=moved.ra, dec=moved.dec, frame="icrs")


def gaia_pm(gaia):
    """
    Gaia proper motions and which stars have one.

    Args:
        gaia (astropy.table.Table): Gaia table

    Returns:
        tuple: (pmra, pmdec, known) numpy arrays; known is False for two-parameter solutions
    """
    pmra = np.asarray(gaia["pmra"], float)
    pmdec = np.asarray(gaia["pmdec"], float)
    return pmra, pmdec, np.isfinite(pmra) & np.isfinite(pmdec)


def quality_flags(ruwe, duplicated):
    """
    FLAG bits from Gaia quality indicators.

    Args:
        ruwe (array-like): RUWE values, NaN where unknown
        duplicated (array-like): Gaia duplicated_source flags

    Returns:
        numpy.ndarray: int16 bitmask
    """
    with np.errstate(invalid="ignore"):
        high_ruwe = np.asarray(ruwe, float) > RUWE_LIMIT
    return (high_ruwe * FLAG_HIGH_RUWE + np.asarray(duplicated, bool) * FLAG_GAIA_DUPLICATE).astype(np.int16)


def hst_rows(hst, gaia):
    """
    Catalogue rows for HST stars, with Gaia information added where Anderson linked a Gaia ID.

    Args:
        hst (astropy.table.Table): kept HST rows (read_anderson format)
        gaia (astropy.table.Table): Gaia table

    Returns:
        astropy.table.Table: rows with the catalogue columns
    """

    n = len(hst)
    rows = Table()
    rows["GAIA_ID"] = np.asarray(hst["GAIA_ID"], np.int64)
    for name in ("RA", "DEC", "RA_ERR", "DEC_ERR", "VMAG"):
        rows[name] = np.asarray(hst[name], float)
    rows["MAG_BAND"] = np.full(n, "F606W")
    rows["GMAG"] = np.full(n, np.nan)
    for name in ("PMRA", "PMDEC", "PMRA_ERR", "PMDEC_ERR"):
        rows[name] = np.asarray(hst[name], float)
    for name in ("PARALLAX", "PARALLAX_ERR", "RUWE"):
        rows[name] = np.full(n, np.nan)
    rows["POS_SOURCE"] = np.full(n, "HST", dtype="<U4")
    rows["PM_SOURCE"] = np.full(n, "HST", dtype="<U4")
    rows["FLAG"] = np.zeros(n, np.int16)

    gaia_index = {int(source_id): i for i, source_id in enumerate(gaia["source_id"])}
    for i in np.flatnonzero(rows["GAIA_ID"] != 0):
        j = gaia_index.get(int(rows["GAIA_ID"][i]))
        if j is None:
            continue   # reported by self_checks as a truncated query
        star = gaia[j]
        rows["GMAG"][i] = star["phot_g_mean_mag"]
        rows["PARALLAX"][i] = star["parallax"]
        rows["PARALLAX_ERR"][i] = star["parallax_error"]
        rows["RUWE"][i] = star["ruwe"]
        rows["FLAG"][i] |= quality_flags([star["ruwe"]], [star["duplicated_source"]])[0]
        # bright stars saturate in the deep HST frames; Gaia measures their motion far better
        gaia_pm_error = np.hypot(star["pmra_error"], star["pmdec_error"])
        if np.isfinite(gaia_pm_error) and gaia_pm_error < np.hypot(rows["PMRA_ERR"][i], rows["PMDEC_ERR"][i]):
            for name in ("PMRA", "PMDEC", "PMRA_ERR", "PMDEC_ERR"):
                rows[name][i] = star[name.lower().replace("_err", "_error")]
            rows["PM_SOURCE"][i] = "GAIA"
    return rows


def gaia_rows(gaia, coords, bulk):
    """
    Catalogue rows for Gaia-only stars.

    Position errors are moved from J2016.0 to J2015.5 by adding half a year of proper-motion
    error in quadrature.

    Args:
        gaia (astropy.table.Table): the Gaia stars to add
        coords (astropy.coordinates.SkyCoord): their positions at J2015.5
        bulk (dict): output of bulk_motion, used where Gaia has no proper motion

    Returns:
        astropy.table.Table: rows with the catalogue columns
    """

    n = len(gaia)
    pmra, pmdec, known = gaia_pm(gaia)
    pmra_error = np.where(known, np.asarray(gaia["pmra_error"], float), bulk["dispersion"])
    pmdec_error = np.where(known, np.asarray(gaia["pmdec_error"], float), bulk["dispersion"])
    years = GAIA_EPOCH - CATALOG_EPOCH
    gmag = np.asarray(gaia["phot_g_mean_mag"], float)

    rows = Table()
    rows["GAIA_ID"] = np.asarray(gaia["source_id"], np.int64)
    rows["RA"] = coords.ra.deg
    rows["DEC"] = coords.dec.deg
    rows["RA_ERR"] = np.hypot(np.asarray(gaia["ra_error"], float), years * pmra_error)
    rows["DEC_ERR"] = np.hypot(np.asarray(gaia["dec_error"], float), years * pmdec_error)
    rows["VMAG"] = gmag
    rows["MAG_BAND"] = np.full(n, "G", dtype="<U5")
    rows["GMAG"] = gmag
    rows["PMRA"] = np.where(known, pmra, bulk["pmra"])
    rows["PMDEC"] = np.where(known, pmdec, bulk["pmdec"])
    rows["PMRA_ERR"] = pmra_error
    rows["PMDEC_ERR"] = pmdec_error
    rows["PARALLAX"] = np.asarray(gaia["parallax"], float)
    rows["PARALLAX_ERR"] = np.asarray(gaia["parallax_error"], float)
    rows["RUWE"] = np.asarray(gaia["ruwe"], float)
    rows["POS_SOURCE"] = np.full(n, "GAIA", dtype="<U4")
    rows["PM_SOURCE"] = np.where(known, "GAIA", "BULK").astype("<U4")
    rows["FLAG"] = (np.where(known, 0, FLAG_PM_ASSUMED)
                    | quality_flags(gaia["ruwe"], gaia["duplicated_source"])).astype(np.int16)
    return rows


def assemble(hst_all, gaia, gaia_coords, bulk):
    """
    Merge HST and Gaia DR3 into the reference catalogue (unsorted, no metadata).

    Args:
        hst_all (astropy.table.Table): all rows of Anderson's master file
        gaia (astropy.table.Table): Gaia table
        gaia_coords (astropy.coordinates.SkyCoord): Gaia positions at J2015.5
        bulk (dict): output of bulk_motion

    Returns:
        astropy.table.Table: the catalogue columns, in the order of COLUMNS
    """

    hst = hst_all[np.asarray(hst_all["VMAG"], float) < HST_MAG_LIMIT]
    hst_all_coords = SkyCoord(np.asarray(hst_all["RA"], float), np.asarray(hst_all["DEC"], float), unit="deg")
    hst_coords = SkyCoord(np.asarray(hst["RA"], float), np.asarray(hst["DEC"], float), unit="deg")

    # inside the HST footprint: enough Anderson stars around the Gaia star
    gaia_index, _, _, _ = hst_all_coords.search_around_sky(gaia_coords, FOOTPRINT_RADIUS * u.arcsec)
    in_footprint = np.bincount(gaia_index, minlength=len(gaia)) >= FOOTPRINT_MIN_STARS

    # already in the catalogue: linked by Anderson's Gaia cross-match, or on top of an HST star
    linked_ids = np.asarray(hst_all["GAIA_ID"], np.int64)
    represented = np.isin(np.asarray(gaia["source_id"], np.int64), linked_ids[linked_ids != 0])
    represented |= near_hst_star(gaia, gaia_coords, hst, hst_coords)

    with np.errstate(invalid="ignore"):
        bright_enough = np.asarray(gaia["phot_g_mean_mag"], float) < GAIA_MAG_LIMIT
    add = in_footprint & ~represented & bright_enough

    catalog = vstack([hst_rows(hst, gaia), gaia_rows(gaia[add], gaia_coords[add], bulk)])
    return catalog[list(COLUMNS)]


def near_hst_star(gaia, gaia_coords, hst, hst_coords):
    """
    Which Gaia stars are the same star as a kept HST star, judged by position and brightness.

    Any Gaia star within DUPLICATE_RADIUS of an HST star is a duplicate. A star without a
    Gaia proper motion is also one if an HST star within WIDE_DUPLICATE_RADIUS differs from
    it by less than DUPLICATE_MAG_DIFF in magnitude (G against F606W). In the LMC field such
    pairs are far more common within 0.3 arcsec than chance alignments, and only for those stars.

    Args:
        gaia (astropy.table.Table): Gaia table
        gaia_coords (astropy.coordinates.SkyCoord): Gaia positions at J2015.5
        hst (astropy.table.Table): kept HST rows
        hst_coords (astropy.coordinates.SkyCoord): their positions

    Returns:
        numpy.ndarray: bool, one entry per Gaia star
    """
    near = np.zeros(len(gaia), bool)
    gaia_index, hst_index, separation, _ = hst_coords.search_around_sky(gaia_coords, WIDE_DUPLICATE_RADIUS * u.arcsec)
    near[gaia_index[separation.arcsec < DUPLICATE_RADIUS]] = True

    # two-parameter positions can be off by tens of mas: a same-brightness HST star nearby is the same star
    _, _, known = gaia_pm(gaia)
    same_brightness = np.abs(np.asarray(gaia["phot_g_mean_mag"], float)[gaia_index]
                             - np.asarray(hst["VMAG"], float)[hst_index]) < DUPLICATE_MAG_DIFF
    near[gaia_index[same_brightness & ~known[gaia_index]]] = True
    return near


def frame_offset(catalog, gaia, gaia_coords):
    """
    Median HST minus Gaia position offset for well-measured stars in both catalogues.

    Args:
        catalog (astropy.table.Table): output of assemble
        gaia (astropy.table.Table): Gaia table
        gaia_coords (astropy.coordinates.SkyCoord): Gaia positions at J2015.5

    Returns:
        tuple: (median RA offset on the sky, median Dec offset) [mas]; NaN if no star qualifies
    """
    gaia_index = {int(source_id): i for i, source_id in enumerate(gaia["source_id"])}
    hst_index, gaia_index_list = [], []
    for i in np.flatnonzero((catalog["POS_SOURCE"] == "HST") & (catalog["GAIA_ID"] != 0)):
        j = gaia_index.get(int(catalog["GAIA_ID"][i]))
        if j is not None and 14 < gaia["phot_g_mean_mag"][j] < 19 and gaia["ra_error"][j] < 0.5 \
                and np.isfinite(gaia["pmra"][j]):
            hst_index.append(i)
            gaia_index_list.append(j)
    if not hst_index:
        return float("nan"), float("nan")
    ra_hst = np.asarray(catalog["RA"][hst_index], float)
    dec_hst = np.asarray(catalog["DEC"][hst_index], float)
    dra = (ra_hst - gaia_coords.ra.deg[gaia_index_list]) * np.cos(np.radians(dec_hst)) * 3.6e6
    ddec = (dec_hst - gaia_coords.dec.deg[gaia_index_list]) * 3.6e6
    return float(np.median(dra)), float(np.median(ddec))


def self_checks(catalog, linked_ids, queried_ids, offset_mas):
    """
    Check the catalogue and raise if anything is wrong.

    Args:
        catalog (astropy.table.Table): the catalogue, ideally as read back from disk
        linked_ids (array-like): Gaia IDs Anderson linked to HST stars (non-zero)
        queried_ids (array-like): source_id of every star the Gaia query returned
        offset_mas (tuple): (RA, Dec) median HST minus Gaia offset [mas]

    Raises:
        RuntimeError: listing every failed check
    """
    failures = []

    ids = np.asarray(catalog["GAIA_ID"], np.int64)
    ids = ids[ids != 0]
    if len(ids) != len(np.unique(ids)):
        failures.append("duplicate Gaia IDs")

    coords = SkyCoord(np.asarray(catalog["RA"], float), np.asarray(catalog["DEC"], float), unit="deg")
    first, second, _, _ = coords.search_around_sky(coords, DUPLICATE_RADIUS * u.arcsec)
    close_pairs = int(np.sum(first < second))
    if close_pairs:
        failures.append("{0} pairs of rows closer than {1} arcsec".format(close_pairs, DUPLICATE_RADIUS))

    not_queried = sorted(set(np.asarray(linked_ids, np.int64).tolist())
                         - set(np.asarray(queried_ids, np.int64).tolist()))
    if not_queried:
        failures.append("Gaia query looks truncated: {0} linked IDs not returned, e.g. {1}".format(
            len(not_queried), not_queried[:3]))

    if not all(np.isfinite(offset_mas)) or max(abs(offset_mas[0]), abs(offset_mas[1])) >= MAX_FRAME_OFFSET_MAS:
        failures.append("HST minus Gaia frame offset {0} mas exceeds {1} mas".format(
            offset_mas, MAX_FRAME_OFFSET_MAS))

    if failures:
        raise RuntimeError("Catalogue self-checks failed:\n  " + "\n  ".join(failures))


def metadata(catalog, anderson_sha256, gaia_meta, bulk, offset_mas, today):
    """
    ECSV metadata describing origin, selection and checks.

    Args:
        catalog (astropy.table.Table): the finished catalogue (used for counts)
        anderson_sha256 (str): checksum of the Anderson file used
        gaia_meta (dict): meta of the Gaia table (adql, queried)
        bulk (dict): output of bulk_motion
        offset_mas (tuple): output of frame_offset
        today (str): ISO build date

    Returns:
        dict: metadata, plain Python types only
    """
    def count(column, value):
        return int(np.sum(catalog[column] == value))

    return {
        "title": "LMC astrometric reference field for Roman CGI astrometric calibration",
        "description": ("HST ACS/WFC F606W catalogue of the JWST/Roman calibration field in the LMC "
                        "(Anderson et al. 2021), supplemented with Gaia DR3 stars that HST did not "
                        "measure. All positions at one epoch; propagate with PMRA/PMDEC to the "
                        "observation time. VMAG is approximately V band: F606W for HST stars, "
                        "G for Gaia-only stars (MAG_BAND)."),
        "epoch": CATALOG_EPOCH, "epoch_format": "jyear", "epoch_scale": "tcb", "frame": "ICRS",
        "pmra_convention": "PMRA is mu_alpha* = mu_alpha cos(dec)",
        "sources": [
            {"name": "HST JWST calibration field catalogue (isolated stars)",
             "reference": ANDERSON_REFERENCE, "url": ANDERSON_URL, "sha256": anderson_sha256,
             "notes": ("HST epochs 2006.39 and 2017.38 tied to the Gaia DR2 frame; positions "
                       "interpolated to 2015.5; VMAG is calibrated F606W VEGAMAG")},
            {"name": "Gaia DR3", "reference": GAIA_REFERENCE, "service": GAIA_TAP_URL,
             "adql": gaia_meta["adql"], "queried": gaia_meta["queried"],
             "notes": ("run as half-open RA/Dec tiles of synchronous queries restricted to this cone; "
                       "row total checked against COUNT(*) of the cone; positions moved from J2016.0 to "
                       "J2015.5 with their proper motions (LMC bulk motion where Gaia has none)")},
        ],
        "selection": {
            "hst": "F606W < {0} (VEGAMAG), as in the former JWST_CALFIELD2020.csv".format(HST_MAG_LIMIT),
            "gaia_only": ("inside the HST footprint (>= {0} HST stars within {1} arcsec), no Anderson "
                          "Gaia ID link, no kept HST star within {2} arcsec (for stars without Gaia proper "
                          "motion also none within {3} arcsec and {4} mag), G < {5}").format(
                              FOOTPRINT_MIN_STARS, FOOTPRINT_RADIUS, DUPLICATE_RADIUS,
                              WIDE_DUPLICATE_RADIUS, DUPLICATE_MAG_DIFF, GAIA_MAG_LIMIT),
        },
        "proper_motions": {
            "hst": "HST two-epoch proper motions on the Gaia frame (Anderson et al. 2021)",
            "gaia": "Gaia DR3, for Gaia-only stars and for HST stars whose Gaia error is smaller",
            "bulk": {"pmra": round(bulk["pmra"], 4), "pmdec": round(bulk["pmdec"], 4),
                     "dispersion": round(bulk["dispersion"], 4), "unit": "mas/yr",
                     "notes": "median and robust spread of the HST proper motions; used where Gaia has none"},
        },
        "flags": {"bit_1": "proper motion assumed (LMC bulk motion)", "bit_2": "Gaia RUWE > 1.4",
                  "bit_4": "Gaia duplicated_source"},
        "checks": {"hst_minus_gaia_median_offset_mas": [round(offset_mas[0], 4), round(offset_mas[1], 4)]},
        "build": {"script": "corgidrp/data/build_lmc_reference_catalog.py",
                  "corgidrp_version": corgidrp.__version__, "date": today},
        "counts": {"total": len(catalog), "pos_hst": count("POS_SOURCE", "HST"),
                   "pos_gaia": count("POS_SOURCE", "GAIA"), "pm_hst": count("PM_SOURCE", "HST"),
                   "pm_gaia": count("PM_SOURCE", "GAIA"), "pm_bulk": count("PM_SOURCE", "BULK")},
    }


def main(argv=None):
    """Command line entry point.

    Args:
        argv (list of str): arguments (default: sys.argv)
    """
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="catalogue to write [%(default)s]")
    parser.add_argument("--anderson-file", default=None, help="local copy of the Anderson master file (else downloaded)")
    parser.add_argument("--gaia-file", default=None,
                        help="ECSV with a saved Gaia query; written on first use, reused afterwards")
    args = parser.parse_args(argv)
    catalog = build_catalog(args.output, anderson_file=args.anderson_file, gaia_file=args.gaia_file)
    print("Wrote {0}: {1}".format(args.output, catalog.meta["counts"]))


if __name__ == "__main__":
    main()
