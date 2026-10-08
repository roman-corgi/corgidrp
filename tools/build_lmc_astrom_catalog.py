"""Build the LMC astrometric reference catalogue used for Roman CGI astrometric calibration.

Merges Jay Anderson's HST-based catalogue of the JWST calibration field in the LMC
(JWST-STScI-007716) with the Gaia DR3 stars HST could not measure (mostly stars that
saturated in the deep HST exposures), at the common epoch J2015.5, runs self-checks, and
writes corgidrp/data/lmc_astrometric_reference.ecsv.

Run from the repository root:

    git show 6e346b02:corgidrp/data/JWST_CALFIELD2020.csv > /some/scratch/legacy.csv
    python tools/build_lmc_astrom_catalog.py --legacy-csv /some/scratch/legacy.csv \
        --gaia-file /some/scratch/gaia_query.ecsv

Needs network access and astroquery. astroquery is a dependency of this script only, not
of corgidrp. With --gaia-file, the Gaia query result is saved on the first run and reused
afterwards, so a rebuild does not depend on the Gaia archive being available.
"""
import argparse
import datetime
import hashlib
import os
import subprocess
import tempfile
import urllib.request
import warnings

import numpy as np
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.io import ascii
from astropy.table import Table, vstack
from astropy.time import Time
from erfa import ErfaWarning

ANDERSON_URL = "https://www.stsci.edu/~jayander/JWST_CALFIELD2020/master_output.xymvrdeepmeehuGAIA_DR3"
ANDERSON_REFERENCE = ("Anderson, J., Fall, M., and the Astrometry Working Group 2021, JWST-STScI-007716, "
                      "The JWST Calibration Field: Absolute Astrometry and Proper Motions with GAIA "
                      "and a Second HST Epoch")
GAIA_REFERENCE = "Gaia Collaboration, Vallenari, A., et al. 2023, A&A, 674, A1 (Gaia DR3)"
GAIA_TAP_URL = "https://gea.esac.esa.int/tap-server/tap"

FIELD_CENTER = (80.5, -69.5)   # deg, centre of Anderson's reference frame
GAIA_CONE_RADIUS = 0.1         # deg, covers the whole HST footprint
CATALOG_EPOCH = Time(2015.5, format="jyear", scale="tcb")
GAIA_EPOCH = Time(2016.0, format="jyear", scale="tcb")

HST_MAG_LIMIT = 22.5           # F606W VEGAMAG, same cut as the legacy JWST_CALFIELD2020.csv
GAIA_MAG_LIMIT = 19.5          # Gaia G; fainter missing stars are mostly blends or poorly measured
DUPLICATE_RADIUS = 0.1 * u.arcsec
FOOTPRINT_RADIUS = 5 * u.arcsec
FOOTPRINT_MIN_STARS = 10
RUWE_LIMIT = 1.4
MAX_FRAME_OFFSET_MAS = 0.2
SYNC_ROW_LIMIT = 2000           # Gaia archive cap on synchronous query results

FLAG_PM_ASSUMED = 1
FLAG_HIGH_RUWE = 2
FLAG_GAIA_DUPLICATE = 4

# stars added by hand in PR #1057 (issue #1049); the G = 19.80 one is below GAIA_MAG_LIMIT
PR1057_KEPT_IDS = [4657987350779055232, 4657987346473687296, 4657987346506841344,
                   4657987346439013120, 4657987350829299328, 4657987385189011456]
PR1057_EXCLUDED_IDS = [4657987350779038592]

GAIA_COLUMNS = ["source_id", "ra", "dec", "ra_error", "dec_error", "pmra", "pmra_error",
                "pmdec", "pmdec_error", "parallax", "parallax_error", "phot_g_mean_mag",
                "ruwe", "astrometric_params_solved", "duplicated_source"]

CATALOG_COLUMNS = ["GAIA_ID", "RA", "DEC", "RA_ERR", "DEC_ERR", "MAG", "MAG_BAND", "GMAG",
                   "PMRA", "PMDEC", "PMRA_ERR", "PMDEC_ERR", "PARALLAX", "PARALLAX_ERR", "RUWE",
                   "POS_SOURCE", "PM_SOURCE", "FLAG"]

COLUMN_UNITS = {"RA": u.deg, "DEC": u.deg, "RA_ERR": u.mas, "DEC_ERR": u.mas, "MAG": u.mag,
                "GMAG": u.mag, "PMRA": u.mas / u.yr, "PMDEC": u.mas / u.yr,
                "PMRA_ERR": u.mas / u.yr, "PMDEC_ERR": u.mas / u.yr, "PARALLAX": u.mas,
                "PARALLAX_ERR": u.mas}

# enough decimals that every legacy value reads back bit-identical
COLUMN_FORMATS = {"RA": ".10f", "DEC": ".9f", "RA_ERR": ".4f", "DEC_ERR": ".4f", "MAG": ".4f",
                  "GMAG": ".4f", "PMRA": ".4f", "PMDEC": ".4f", "PMRA_ERR": ".4f",
                  "PMDEC_ERR": ".4f", "PARALLAX": ".4f", "PARALLAX_ERR": ".4f", "RUWE": ".3f"}

COLUMN_DESCRIPTIONS = {
    "GAIA_ID": "Gaia DR3 source_id, 0 if the star has no Gaia counterpart",
    "RA": "Right ascension, ICRS, epoch J2015.5",
    "DEC": "Declination, ICRS, epoch J2015.5",
    "RA_ERR": "Uncertainty of RA on the sky (times cos dec) at J2015.5",
    "DEC_ERR": "Uncertainty of DEC at J2015.5",
    "MAG": "Magnitude used for brightness ranking; band given in MAG_BAND",
    "MAG_BAND": "F606W (HST ACS/WFC, VEGAMAG) or G (Gaia DR3)",
    "GMAG": "Gaia DR3 G magnitude, NaN without Gaia counterpart",
    "PMRA": "Proper motion in RA, mu_alpha* = mu_alpha cos(dec)",
    "PMDEC": "Proper motion in Dec",
    "PMRA_ERR": "Uncertainty of PMRA",
    "PMDEC_ERR": "Uncertainty of PMDEC",
    "PARALLAX": "Gaia DR3 parallax, NaN if not available",
    "PARALLAX_ERR": "Uncertainty of PARALLAX",
    "RUWE": "Gaia DR3 renormalised unit weight error, NaN if not available",
    "POS_SOURCE": "Source of RA/DEC: HST or GAIA",
    "PM_SOURCE": "Source of the proper motion: HST, GAIA, or BULK (LMC bulk motion assumed)",
    "FLAG": "Bitmask: 1 proper motion assumed, 2 RUWE > 1.4, 4 Gaia duplicated_source",
}


def parse_anderson_master(path):
    """Read Anderson's master_output.xymvrdeepmeehuGAIA_DR3 isolated-star catalogue.

    Columns used (1-based, as in the file header): 4 F606W VEGAMAG, 5/6 RA/Dec at 2015.5 [deg],
    7/8 position errors [mas], 9/10 proper motions [mas/yr], 11/12 their errors,
    34 Gaia DR3 source_id (0 if none).

    Args:
        path (str): path to the downloaded master file.

    Returns:
        astropy.table.Table: one row per star, columns MAG, RA, DEC, RA_ERR, DEC_ERR, PMRA,
            PMDEC, PMRA_ERR, PMDEC_ERR, GAIA_ID.
    """
    values = np.loadtxt(path, comments="#", usecols=(3, 4, 5, 6, 7, 8, 9, 10, 11))
    gaia_ids = np.loadtxt(path, comments="#", usecols=(33,), dtype=np.int64)
    names = ["MAG", "RA", "DEC", "RA_ERR", "DEC_ERR", "PMRA", "PMDEC", "PMRA_ERR", "PMDEC_ERR"]
    table = Table(values, names=names)
    table["GAIA_ID"] = gaia_ids
    return table


def measure_bulk_motion(hst):
    """Measure the LMC bulk proper motion and internal dispersion from HST stars.

    Args:
        hst (astropy.table.Table): HST rows with PMRA, PMDEC [mas/yr].

    Returns:
        dict: pmra, pmdec (medians) and dispersion (mean of the robust standard deviations of
            both axes), all floats in mas/yr.
    """
    pmra = np.asarray(hst["PMRA"], float)
    pmdec = np.asarray(hst["PMDEC"], float)

    def robust_std(values):
        return 1.4826 * np.median(np.abs(values - np.median(values)))

    return {"pmra": float(np.median(pmra)), "pmdec": float(np.median(pmdec)),
            "dispersion": float(0.5 * (robust_std(pmra) + robust_std(pmdec)))}


def cone_tiles(center, radius, n_per_side):
    """Half-open RA/Dec boxes that together cover a cone's bounding box.

    Each box is [ra_min, ra_max) x [dec_min, dec_max), so a star on a shared edge belongs to
    exactly one box. The outer edges are padded slightly so the whole cone is inside.

    Args:
        center (tuple): (ra, dec) of the cone centre [deg].
        radius (float): cone radius [deg].
        n_per_side (int): number of boxes along RA and along Dec.

    Returns:
        list of tuple: (ra_min, ra_max, dec_min, dec_max) per box [deg].
    """
    pad = 1e-6
    dec_half = radius + pad
    # widest RA extent is at the declination closest to the pole
    ra_half = radius / np.cos(np.radians(min(abs(center[1]) + radius, 89.9))) + pad
    ra_edges = np.linspace(center[0] - ra_half, center[0] + ra_half, n_per_side + 1)
    dec_edges = np.linspace(center[1] - dec_half, center[1] + dec_half, n_per_side + 1)
    return [(ra_edges[i], ra_edges[i + 1], dec_edges[j], dec_edges[j + 1])
            for i in range(n_per_side) for j in range(n_per_side)]


def _run_sync_query(adql, attempts):
    """Run a synchronous Gaia archive query, retrying on failure.

    Args:
        adql (str): query text.
        attempts (int): number of tries.

    Returns:
        astropy.table.Table: the query result.
    """
    from astroquery.gaia import Gaia  # build-time only dependency

    last_error = None
    for _ in range(attempts):
        try:
            return Gaia.launch_job(adql).get_results()
        except Exception as error:   # network or archive failure: retry
            last_error = error
    raise RuntimeError("Gaia query failed {0} times: {1}\n{2}".format(attempts, last_error, adql))


def query_gaia(center=FIELD_CENTER, radius=GAIA_CONE_RADIUS, attempts=3):
    """Query Gaia DR3 for all sources in a cone around the field.

    The archive caps synchronous queries at SYNC_ROW_LIMIT rows without warning, and its
    asynchronous queue can stall for a long time. So the cone is split into tiles small enough
    for synchronous queries; a tile at the cap, or a total that differs from a COUNT(*) of the
    cone, aborts the build.

    Args:
        center (tuple): (ra, dec) of the cone centre [deg].
        radius (float): cone radius [deg].
        attempts (int): number of tries per query (the archive is sometimes unstable).

    Returns:
        tuple: (astropy.table.Table in the Gaia convention with NaN for missing values,
            str ADQL query text describing the whole cone).
    """
    cone = "1 = CONTAINS(POINT('ICRS', ra, dec), CIRCLE('ICRS', {0}, {1}, {2}))".format(center[0], center[1], radius)
    adql = "SELECT " + ", ".join(GAIA_COLUMNS) + " FROM gaiadr3.gaia_source WHERE " + cone
    expected = int(_run_sync_query("SELECT COUNT(*) AS n FROM gaiadr3.gaia_source WHERE " + cone, attempts)["n"][0])

    # aim for about a third of the row cap per tile (stars are not spread evenly)
    n_per_side = int(np.ceil(np.sqrt(3.0 * expected / SYNC_ROW_LIMIT))) + 1
    pieces = []
    for ra_min, ra_max, dec_min, dec_max in cone_tiles(center, radius, n_per_side):
        tile = adql + (" AND ra >= {0!r} AND ra < {1!r} AND dec >= {2!r} AND dec < {3!r}"
                       .format(float(ra_min), float(ra_max), float(dec_min), float(dec_max)))
        piece = _run_sync_query(tile, attempts)
        if len(piece) >= SYNC_ROW_LIMIT:
            raise RuntimeError("Gaia tile hit the {0}-row synchronous limit; use more tiles".format(SYNC_ROW_LIMIT))
        pieces.append(piece)
    result = vstack(pieces)
    if len(result) != expected:
        raise RuntimeError("Gaia query returned {0} rows, COUNT(*) says {1}".format(len(result), expected))

    gaia = Table()
    for name in GAIA_COLUMNS:
        column = result[name]
        if name in ("source_id", "astrometric_params_solved"):
            gaia[name] = np.asarray(column).astype(np.int64)
        elif name == "duplicated_source":
            gaia[name] = np.ma.filled(np.ma.asarray(column), False).astype(bool)
        else:
            gaia[name] = np.ma.filled(np.ma.asarray(column, dtype=float), np.nan)
    return gaia, adql


def gaia_positions_at_catalog_epoch(gaia, bulk):
    """Move Gaia DR3 positions from J2016.0 to the catalogue epoch J2015.5.

    Stars without a Gaia proper motion (two-parameter solutions) are moved with the LMC bulk
    motion, the same motion the catalogue assigns them.

    Args:
        gaia (astropy.table.Table): Gaia table (see query_gaia).
        bulk (dict): output of measure_bulk_motion.

    Returns:
        astropy.coordinates.SkyCoord: ICRS positions at J2015.5, in the order of gaia.
    """
    pmra = np.asarray(gaia["pmra"], float)
    pmdec = np.asarray(gaia["pmdec"], float)
    known = np.isfinite(pmra) & np.isfinite(pmdec)
    coords = SkyCoord(ra=np.asarray(gaia["ra"], float) * u.deg,
                      dec=np.asarray(gaia["dec"], float) * u.deg,
                      pm_ra_cosdec=np.where(known, pmra, bulk["pmra"]) * u.mas / u.yr,
                      pm_dec=np.where(known, pmdec, bulk["pmdec"]) * u.mas / u.yr,
                      obstime=GAIA_EPOCH, frame="icrs")
    with warnings.catch_warnings():
        # without distances ERFA assumes very distant stars and says so; harmless here
        warnings.simplefilter("ignore", ErfaWarning)
        moved = coords.apply_space_motion(new_obstime=CATALOG_EPOCH)
    return SkyCoord(ra=moved.ra, dec=moved.dec, frame="icrs")


def _gaia_quality_flags(ruwe, duplicated):
    """FLAG bits from Gaia quality indicators.

    Args:
        ruwe (numpy.ndarray): RUWE values, NaN where unknown.
        duplicated (numpy.ndarray): Gaia duplicated_source flags.

    Returns:
        numpy.ndarray: int16 bitmask.
    """
    with np.errstate(invalid="ignore"):
        high_ruwe = np.asarray(ruwe, float) > RUWE_LIMIT
    return (high_ruwe * FLAG_HIGH_RUWE + np.asarray(duplicated, bool) * FLAG_GAIA_DUPLICATE).astype(np.int16)


def _hst_rows(hst, gaia):
    """Catalogue rows for HST stars, with Gaia information added where linked by ID.

    Args:
        hst (astropy.table.Table): kept HST rows (parse_anderson_master format).
        gaia (astropy.table.Table): Gaia table.

    Returns:
        astropy.table.Table: rows with CATALOG_COLUMNS.
    """
    n = len(hst)
    rows = Table()
    rows["GAIA_ID"] = np.asarray(hst["GAIA_ID"], np.int64)
    for name in ("RA", "DEC", "RA_ERR", "DEC_ERR", "MAG"):
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
            continue   # reported by run_self_checks as a truncated query
        star = gaia[j]
        rows["GMAG"][i] = star["phot_g_mean_mag"]
        rows["PARALLAX"][i] = star["parallax"]
        rows["PARALLAX_ERR"][i] = star["parallax_error"]
        rows["RUWE"][i] = star["ruwe"]
        rows["FLAG"][i] |= _gaia_quality_flags([star["ruwe"]], [star["duplicated_source"]])[0]
        gaia_pm_error = np.hypot(star["pmra_error"], star["pmdec_error"])
        hst_pm_error = np.hypot(rows["PMRA_ERR"][i], rows["PMDEC_ERR"][i])
        # bright stars saturate in the deep HST frames; Gaia measures them far better
        if np.isfinite(gaia_pm_error) and gaia_pm_error < hst_pm_error:
            rows["PMRA"][i] = star["pmra"]
            rows["PMDEC"][i] = star["pmdec"]
            rows["PMRA_ERR"][i] = star["pmra_error"]
            rows["PMDEC_ERR"][i] = star["pmdec_error"]
            rows["PM_SOURCE"][i] = "GAIA"
    return rows


def _gaia_rows(gaia, coords, bulk):
    """Catalogue rows for Gaia-only stars.

    Args:
        gaia (astropy.table.Table): the Gaia stars to add.
        coords (astropy.coordinates.SkyCoord): their positions at J2015.5.
        bulk (dict): output of measure_bulk_motion, used where Gaia has no proper motion.

    Returns:
        astropy.table.Table: rows with CATALOG_COLUMNS.
    """
    n = len(gaia)
    pmra = np.asarray(gaia["pmra"], float)
    pmdec = np.asarray(gaia["pmdec"], float)
    known = np.isfinite(pmra) & np.isfinite(pmdec)
    gmag = np.asarray(gaia["phot_g_mean_mag"], float)

    rows = Table()
    rows["GAIA_ID"] = np.asarray(gaia["source_id"], np.int64)
    rows["RA"] = coords.ra.deg
    rows["DEC"] = coords.dec.deg
    rows["RA_ERR"] = np.asarray(gaia["ra_error"], float)
    rows["DEC_ERR"] = np.asarray(gaia["dec_error"], float)
    rows["MAG"] = gmag
    rows["MAG_BAND"] = np.full(n, "G", dtype="<U5")
    rows["GMAG"] = gmag
    rows["PMRA"] = np.where(known, pmra, bulk["pmra"])
    rows["PMDEC"] = np.where(known, pmdec, bulk["pmdec"])
    rows["PMRA_ERR"] = np.where(known, np.asarray(gaia["pmra_error"], float), bulk["dispersion"])
    rows["PMDEC_ERR"] = np.where(known, np.asarray(gaia["pmdec_error"], float), bulk["dispersion"])
    rows["PARALLAX"] = np.asarray(gaia["parallax"], float)
    rows["PARALLAX_ERR"] = np.asarray(gaia["parallax_error"], float)
    rows["RUWE"] = np.asarray(gaia["ruwe"], float)
    rows["POS_SOURCE"] = np.full(n, "GAIA", dtype="<U4")
    rows["PM_SOURCE"] = np.where(known, "GAIA", "BULK").astype("<U4")
    rows["FLAG"] = (np.where(known, 0, FLAG_PM_ASSUMED)
                    | _gaia_quality_flags(gaia["ruwe"], gaia["duplicated_source"])).astype(np.int16)
    return rows


def assemble_catalog(hst_all, gaia, gaia_coords, bulk):
    """Merge HST and Gaia DR3 into the reference catalogue (unsorted, no metadata).

    Args:
        hst_all (astropy.table.Table): all rows of Anderson's master file.
        gaia (astropy.table.Table): Gaia table (see query_gaia).
        gaia_coords (astropy.coordinates.SkyCoord): Gaia positions at J2015.5.
        bulk (dict): output of measure_bulk_motion.

    Returns:
        astropy.table.Table: catalogue with CATALOG_COLUMNS.
    """
    hst = hst_all[np.asarray(hst_all["MAG"], float) < HST_MAG_LIMIT]
    hst_all_coords = SkyCoord(np.asarray(hst_all["RA"], float), np.asarray(hst_all["DEC"], float), unit="deg")
    hst_coords = SkyCoord(np.asarray(hst["RA"], float), np.asarray(hst["DEC"], float), unit="deg")

    # inside the HST footprint: enough Anderson stars around the Gaia star
    gaia_index, _, _, _ = hst_all_coords.search_around_sky(gaia_coords, FOOTPRINT_RADIUS)
    in_footprint = np.bincount(gaia_index, minlength=len(gaia)) >= FOOTPRINT_MIN_STARS

    # already in the catalogue: linked by Anderson's Gaia cross-match, or on top of an HST star
    linked_ids = np.asarray(hst_all["GAIA_ID"], np.int64)
    linked_ids = linked_ids[linked_ids != 0]
    _, separation, _ = gaia_coords.match_to_catalog_sky(hst_coords)
    represented = np.isin(np.asarray(gaia["source_id"], np.int64), linked_ids) | (separation < DUPLICATE_RADIUS)

    with np.errstate(invalid="ignore"):
        bright_enough = np.asarray(gaia["phot_g_mean_mag"], float) < GAIA_MAG_LIMIT
    add = in_footprint & ~represented & bright_enough

    catalog = vstack([_hst_rows(hst, gaia), _gaia_rows(gaia[add], gaia_coords[add], bulk)])
    return catalog[CATALOG_COLUMNS]


def frame_offset_mas(catalog, gaia, gaia_coords):
    """Median HST minus Gaia position offset for well-measured stars in both catalogues.

    Args:
        catalog (astropy.table.Table): output of assemble_catalog.
        gaia (astropy.table.Table): Gaia table.
        gaia_coords (astropy.coordinates.SkyCoord): Gaia positions at J2015.5.

    Returns:
        tuple: (median RA offset on the sky, median Dec offset), floats in mas; NaN if no
            star qualifies.
    """
    gaia_index = {int(source_id): i for i, source_id in enumerate(gaia["source_id"])}
    hst_rows, gaia_rows = [], []
    for i in np.flatnonzero((catalog["POS_SOURCE"] == "HST") & (catalog["GAIA_ID"] != 0)):
        j = gaia_index.get(int(catalog["GAIA_ID"][i]))
        if j is None:
            continue
        gmag = gaia["phot_g_mean_mag"][j]
        if 14 < gmag < 19 and gaia["ra_error"][j] < 0.5 and np.isfinite(gaia["pmra"][j]):
            hst_rows.append(i)
            gaia_rows.append(j)
    if not hst_rows:
        return float("nan"), float("nan")
    ra_h = np.asarray(catalog["RA"][hst_rows], float)
    dec_h = np.asarray(catalog["DEC"][hst_rows], float)
    dra = (ra_h - gaia_coords.ra.deg[gaia_rows]) * np.cos(np.radians(dec_h)) * 3.6e6
    ddec = (dec_h - gaia_coords.dec.deg[gaia_rows]) * 3.6e6
    return float(np.median(dra)), float(np.median(ddec))


def run_self_checks(catalog, legacy, linked_ids, queried_ids, offset_mas, required_ids=(), excluded_ids=()):
    """Check the catalogue and raise if anything is wrong.

    Args:
        catalog (astropy.table.Table): the catalogue, ideally as read back from disk.
        legacy (astropy.table.Table): the legacy JWST_CALFIELD2020.csv (VMAG, RA, DEC).
        linked_ids (array-like): Gaia IDs Anderson linked to HST stars (non-zero).
        queried_ids (array-like): source_id of every star the Gaia query returned.
        offset_mas (tuple): (RA, Dec) median HST minus Gaia offset [mas].
        required_ids (iterable): Gaia IDs that must be in the catalogue.
        excluded_ids (iterable): Gaia IDs that must not be in the catalogue.

    Raises:
        RuntimeError: listing every failed check.
    """
    failures = []

    hst = catalog[catalog["POS_SOURCE"] == "HST"]
    new_rows = set(zip(np.asarray(hst["RA"], float).tolist(), np.asarray(hst["DEC"], float).tolist(),
                       np.asarray(hst["MAG"], float).tolist()))
    legacy_rows = list(zip(np.asarray(legacy["RA"], float).tolist(), np.asarray(legacy["DEC"], float).tolist(),
                           np.asarray(legacy["VMAG"], float).tolist()))
    missing = [row for row in legacy_rows if row not in new_rows]
    if missing:
        failures.append("{0} legacy rows not reproduced exactly, e.g. {1}".format(len(missing), missing[:3]))

    present = set(np.asarray(catalog["GAIA_ID"], np.int64).tolist())
    absent_required = [i for i in required_ids if i not in present]
    if absent_required:
        failures.append("required Gaia IDs missing: {0}".format(absent_required))
    present_excluded = [i for i in excluded_ids if i in present]
    if present_excluded:
        failures.append("excluded Gaia IDs present: {0}".format(present_excluded))

    ids = np.asarray(catalog["GAIA_ID"], np.int64)
    ids = ids[ids != 0]
    if len(ids) != len(np.unique(ids)):
        failures.append("duplicate Gaia IDs")

    coords = SkyCoord(np.asarray(catalog["RA"], float), np.asarray(catalog["DEC"], float), unit="deg")
    first, second, _, _ = coords.search_around_sky(coords, DUPLICATE_RADIUS)
    close_pairs = int(np.sum(first < second))
    if close_pairs:
        failures.append("{0} pairs of rows closer than {1}".format(close_pairs, DUPLICATE_RADIUS))

    not_queried = sorted(set(np.asarray(linked_ids, np.int64).tolist())
                         - set(np.asarray(queried_ids, np.int64).tolist()))
    if not_queried:
        failures.append("Gaia query looks truncated: {0} linked IDs not returned, e.g. {1}".format(
            len(not_queried), not_queried[:3]))

    if not all(np.isfinite(offset_mas)) or max(abs(offset_mas[0]), abs(offset_mas[1])) >= MAX_FRAME_OFFSET_MAS:
        failures.append("HST minus Gaia frame offset {0} mas exceeds {1} mas".format(offset_mas, MAX_FRAME_OFFSET_MAS))

    if failures:
        raise RuntimeError("Catalogue self-checks failed:\n  " + "\n  ".join(failures))


def apply_units_and_formats(catalog):
    """Attach units, output formats and descriptions to the catalogue columns (in place).

    Args:
        catalog (astropy.table.Table): catalogue with CATALOG_COLUMNS.
    """
    for name, unit in COLUMN_UNITS.items():
        catalog[name].unit = unit
    for name, fmt in COLUMN_FORMATS.items():
        catalog[name].info.format = fmt
    for name, text in COLUMN_DESCRIPTIONS.items():
        catalog[name].description = text


def _sha256(path):
    """SHA-256 of a file.

    Args:
        path (str): file path.

    Returns:
        str: hex digest.
    """
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit():
    """Commit of the repository HEAD, or 'unknown' outside git.

    Returns:
        str: commit hash.
    """
    result = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True)
    return result.stdout.strip() or "unknown"


def build_metadata(catalog, anderson_sha256, anderson_retrieved, adql, gaia_queried, bulk, offset_mas):
    """ECSV metadata describing origin, selection and checks.

    Args:
        catalog (astropy.table.Table): the finished catalogue (used for counts).
        anderson_sha256 (str): checksum of the Anderson file used.
        anderson_retrieved (str): ISO date the Anderson file was downloaded.
        adql (str): Gaia query text.
        gaia_queried (str): ISO date of the Gaia query.
        bulk (dict): output of measure_bulk_motion.
        offset_mas (tuple): output of frame_offset_mas.

    Returns:
        dict: metadata, YAML-serialisable (plain Python types only).
    """
    def count(column, value):
        return int(np.sum(catalog[column] == value))

    return {
        "title": "LMC astrometric reference field for Roman CGI astrometric calibration",
        "description": ("HST ACS/WFC F606W catalogue of the JWST/Roman calibration field in the LMC "
                        "(Anderson et al. 2021), supplemented with Gaia DR3 stars that HST did not "
                        "measure. All positions at one epoch; propagate with PMRA/PMDEC to the "
                        "observation time."),
        "epoch": 2015.5, "epoch_format": "jyear", "epoch_scale": "tcb", "frame": "ICRS",
        "pmra_convention": "PMRA is mu_alpha* = mu_alpha cos(dec)",
        "sources": [
            {"name": "HST JWST calibration field catalogue (isolated stars)", "reference": ANDERSON_REFERENCE,
             "url": ANDERSON_URL, "sha256": anderson_sha256, "retrieved": anderson_retrieved,
             "notes": ("HST epochs 2006.39 and 2017.38 tied to the Gaia DR2 frame; positions "
                       "interpolated to 2015.5; MAG is calibrated F606W VEGAMAG")},
            {"name": "Gaia DR3", "reference": GAIA_REFERENCE, "service": GAIA_TAP_URL,
             "adql": adql, "queried": gaia_queried,
             "notes": ("run as half-open RA/Dec tiles of synchronous queries restricted to this cone; "
                       "row total checked against COUNT(*) of the cone; positions moved from J2016.0 to "
                       "J2015.5 with their proper motions (LMC bulk motion where Gaia has none)")},
        ],
        "selection": {
            "hst": "F606W < {0} (VEGAMAG), as in the legacy JWST_CALFIELD2020.csv".format(HST_MAG_LIMIT),
            "gaia_only": ("inside the HST footprint (>= {0} HST stars within {1}), no Anderson Gaia ID "
                          "link, no kept HST star within {2}, G < {3}").format(
                              FOOTPRINT_MIN_STARS, FOOTPRINT_RADIUS, DUPLICATE_RADIUS, GAIA_MAG_LIMIT),
        },
        "proper_motions": {
            "hst": "HST two-epoch proper motions on the Gaia frame (Anderson et al. 2021)",
            "gaia": "Gaia DR3, for Gaia-only stars and for HST stars whose Gaia error is smaller",
            "bulk": {"pmra": round(bulk["pmra"], 4), "pmdec": round(bulk["pmdec"], 4),
                     "dispersion": round(bulk["dispersion"], 4),
                     "unit": "mas/yr",
                     "notes": "median and robust spread of the HST proper motions; used where Gaia has none"},
        },
        "flags": {"bit_1": "proper motion assumed (LMC bulk motion)", "bit_2": "Gaia RUWE > 1.4",
                  "bit_4": "Gaia duplicated_source"},
        "checks": {"hst_minus_gaia_median_offset_mas": [round(offset_mas[0], 4), round(offset_mas[1], 4)],
                   "legacy_rows_reproduced_exactly": True,
                   "pr1057_ids_present": [int(i) for i in PR1057_KEPT_IDS],
                   "pr1057_ids_excluded_below_mag_limit": [int(i) for i in PR1057_EXCLUDED_IDS]},
        "build": {"script": "tools/build_lmc_astrom_catalog.py",
                  "script_sha256": _sha256(os.path.abspath(__file__)),
                  "git_commit": _git_commit(),
                  "date": datetime.date.today().isoformat()},
        "counts": {"total": len(catalog), "pos_hst": count("POS_SOURCE", "HST"),
                   "pos_gaia": count("POS_SOURCE", "GAIA"), "pm_hst": count("PM_SOURCE", "HST"),
                   "pm_gaia": count("PM_SOURCE", "GAIA"), "pm_bulk": count("PM_SOURCE", "BULK")},
    }


def main(argv=None):
    """Build, check and write the catalogue.

    Args:
        argv (list of str): command line arguments (default: sys.argv).
    """
    parser = argparse.ArgumentParser(description="Build the LMC astrometric reference catalogue for CGI.")
    parser.add_argument("--output", default=os.path.join("corgidrp", "data", "lmc_astrometric_reference.ecsv"))
    parser.add_argument("--legacy-csv", required=True,
                        help=("legacy catalogue every HST row must reproduce exactly; recover it with "
                              "`git show 6e346b02:corgidrp/data/JWST_CALFIELD2020.csv > legacy.csv`"))
    parser.add_argument("--anderson-file", default=None, help="local copy of the Anderson master file")
    parser.add_argument("--gaia-file", default=None,
                        help="ECSV with a saved Gaia query; written on first use, reused afterwards")
    args = parser.parse_args(argv)
    today = datetime.date.today().isoformat()

    with tempfile.TemporaryDirectory() as scratch:
        anderson_path = args.anderson_file
        if anderson_path is None:
            anderson_path = os.path.join(scratch, "master_output.xymvrdeepmeehuGAIA_DR3")
            urllib.request.urlretrieve(ANDERSON_URL, anderson_path)
        anderson_sha256 = _sha256(anderson_path)
        hst_all = parse_anderson_master(anderson_path)

    hst = hst_all[np.asarray(hst_all["MAG"], float) < HST_MAG_LIMIT]
    bulk = measure_bulk_motion(hst)

    if args.gaia_file and os.path.exists(args.gaia_file):
        gaia = Table.read(args.gaia_file, format="ascii.ecsv")
        adql, gaia_queried = gaia.meta["adql"], gaia.meta["queried"]
    else:
        gaia, adql = query_gaia()
        gaia_queried = today
        if args.gaia_file:
            gaia.meta = {"adql": adql, "queried": gaia_queried}
            gaia.write(args.gaia_file, format="ascii.ecsv", overwrite=True)

    gaia_coords = gaia_positions_at_catalog_epoch(gaia, bulk)
    catalog = assemble_catalog(hst_all, gaia, gaia_coords, bulk)
    offset = frame_offset_mas(catalog, gaia, gaia_coords)
    catalog.sort("MAG")
    apply_units_and_formats(catalog)
    catalog.meta = build_metadata(catalog, anderson_sha256, today, adql, gaia_queried, bulk, offset)
    catalog.write(args.output, format="ascii.ecsv", overwrite=True)

    linked_ids = np.asarray(hst_all["GAIA_ID"], np.int64)
    readback = Table.read(args.output, format="ascii.ecsv")
    try:
        run_self_checks(readback, ascii.read(args.legacy_csv), linked_ids[linked_ids != 0], gaia["source_id"],
                        offset, required_ids=PR1057_KEPT_IDS, excluded_ids=PR1057_EXCLUDED_IDS)
    except RuntimeError:
        os.remove(args.output)   # never leave an unchecked catalogue behind
        raise

    print("Wrote {0}: {1}".format(args.output, catalog.meta["counts"]))
    print("Bulk motion {0}; HST - Gaia offset {1} mas".format(bulk, offset))


if __name__ == "__main__":
    main()
