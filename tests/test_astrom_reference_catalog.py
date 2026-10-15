"""Checks of the LMC astrometric reference catalogue shipped in corgidrp/data."""
import numpy as np
import astropy.units as u
import corgidrp.astrom as astrom

# stars added by hand in PR #1057; its 4657987346473687296 (a two-parameter solution) is the
# same star as an HST source 0.13 arcsec away and is covered by that HST row
PR1057_KEPT_IDS = [4657987350779055232, 4657987346506841344, 4657987346439013120,
                   4657987350829299328, 4657987385189011456]
REQUIRED_COLUMNS = {"GAIA_ID": None, "RA": u.deg, "DEC": u.deg, "RA_ERR": u.mas, "DEC_ERR": u.mas,
                    "MAG": u.mag, "MAG_BAND": None, "GMAG": u.mag, "PMRA": u.mas / u.yr,
                    "PMDEC": u.mas / u.yr, "PMRA_ERR": u.mas / u.yr, "PMDEC_ERR": u.mas / u.yr,
                    "PARALLAX": u.mas, "PARALLAX_ERR": u.mas, "RUWE": None, "POS_SOURCE": None,
                    "PM_SOURCE": None, "FLAG": None}


def _catalog():
    """Load the packaged catalogue.

    Returns:
        astropy.table.Table: the catalogue
    """
    return astrom.read_field_catalog(astrom.default_field_path)


def test_catalog_has_required_columns_and_units():
    """Every documented column is present with its unit."""
    catalog = _catalog()
    for name, unit in REQUIRED_COLUMNS.items():
        assert name in catalog.colnames, name
        if unit is not None:
            assert catalog[name].unit == unit, name


def test_catalog_metadata_records_epoch_and_provenance():
    """Epoch, time scale, sources and selection are recorded in the file."""
    meta = _catalog().meta
    assert meta["epoch"] == 2015.5
    assert meta["epoch_format"] == "jyear"
    assert meta["epoch_scale"] == "tcb"
    assert meta["frame"] == "ICRS"
    assert len(meta["sources"]) == 2
    assert "url" in meta["sources"][0] and "adql" in meta["sources"][1]
    assert "hst" in meta["selection"] and "gaia_only" in meta["selection"]


def test_catalog_values_are_complete_and_consistent():
    """Positions, magnitudes and proper motions are finite; labels use the documented values."""
    catalog = _catalog()
    for name in ("RA", "DEC", "MAG", "PMRA", "PMDEC", "PMRA_ERR", "PMDEC_ERR"):
        assert np.all(np.isfinite(np.asarray(catalog[name], float))), name
    assert set(catalog["POS_SOURCE"]) <= {"HST", "GAIA"}
    assert set(catalog["PM_SOURCE"]) <= {"HST", "GAIA", "BULK"}
    assert set(catalog["MAG_BAND"]) <= {"F606W", "G"}
    hst = catalog["POS_SOURCE"] == "HST"
    assert np.all(np.asarray(catalog["MAG"][hst], float) < 22.5)
    assert np.all(catalog["MAG_BAND"][hst] == "F606W")


def test_gaia_ids_unique_and_pr1057_stars_present():
    """No Gaia star appears twice, and the stars added by hand in PR #1057 are covered."""
    ids = np.asarray(_catalog()["GAIA_ID"], np.int64)
    linked = ids[ids != 0]
    assert len(linked) == len(np.unique(linked))
    for source_id in PR1057_KEPT_IDS:
        assert source_id in linked


def test_assumed_proper_motions_are_flagged():
    """Stars given the LMC bulk motion carry FLAG bit 1, and only those."""
    catalog = _catalog()
    bulk = catalog["PM_SOURCE"] == "BULK"
    assert np.all((np.asarray(catalog["FLAG"][bulk]) & 1) == 1)
    assert np.all((np.asarray(catalog["FLAG"][~bulk]) & 1) == 0)
