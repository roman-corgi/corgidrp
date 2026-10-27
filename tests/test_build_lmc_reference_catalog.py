"""Tests for the LMC astrometric reference catalogue: its build script, on small synthetic
catalogues (no network), and the catalogue shipped in corgidrp/data."""
import importlib.util
import os

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

import corgidrp.astrom as astrom

# corgidrp/data is not an importable package, so load the build script from its file
_script = os.path.join(os.path.dirname(astrom.__file__), "data", "build_lmc_reference_catalog.py")
_spec = importlib.util.spec_from_file_location("build_lmc_reference_catalog", _script)
build = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(build)

CENTER = (80.5, -69.5)
BULK = {"pmra": 1.875, "pmdec": 0.376, "dispersion": 0.14}


def _offset(dx_arcsec, dy_arcsec):
    """Return (ra, dec) [deg] offset from CENTER by on-sky (dx east, dy north) arcsec."""
    ra = CENTER[0] + dx_arcsec / 3600. / np.cos(np.radians(CENTER[1]))
    dec = CENTER[1] + dy_arcsec / 3600.
    return ra, dec


def make_hst(extra=()):
    """A 7x7 grid of HST stars, 1 arcsec apart, around CENTER, plus optional extra rows.

    Args:
        extra (iterable of dict): extra rows; keys override the defaults below.

    Returns:
        astropy.table.Table: Anderson-style table as returned by build.read_anderson.
    """
    rows = []
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            ra, dec = _offset(dx, dy)
            rows.append(dict(VMAG=20.0, RA=ra, DEC=dec, RA_ERR=0.2, DEC_ERR=0.2, PMRA=1.875,
                             PMDEC=0.376, PMRA_ERR=0.02, PMDEC_ERR=0.02, GAIA_ID=0))
    for row in extra:
        base = dict(rows[0])
        base.update(row)
        rows.append(base)
    return Table(rows=rows, names=list(rows[0].keys()))


def make_gaia(rows):
    """Gaia-style table from a list of dicts with at least source_id, ra, dec, phot_g_mean_mag.

    Args:
        rows (list of dict): one dict per star; missing quantities get sensible defaults.

    Returns:
        astropy.table.Table: table in the Gaia convention used by the builder.
    """
    defaults = dict(ra_error=0.05, dec_error=0.05, pmra=1.9, pmra_error=0.03, pmdec=0.3,
                    pmdec_error=0.03, parallax=0.0, parallax_error=0.03, ruwe=1.0,
                    astrometric_params_solved=31, duplicated_source=False)
    full = []
    for row in rows:
        r = dict(defaults)
        r.update(row)
        full.append(r)
    names = build.GAIA_COLUMNS
    table = Table(rows=[[r[n] for n in names] for r in full], names=names)
    table["source_id"] = table["source_id"].astype(np.int64)
    return table


def two_parameter(**row):
    """Gaia row without proper motion or parallax (two-parameter solution).

    Args:
        row: source_id, ra, dec, phot_g_mean_mag and any other values to set.

    Returns:
        dict: the row with proper motion and parallax set to NaN.
    """
    row.update(pmra=np.nan, pmdec=np.nan, pmra_error=np.nan, pmdec_error=np.nan,
               parallax=np.nan, parallax_error=np.nan, astrometric_params_solved=3)
    return row


def run_assemble(hst, gaia):
    coords = build.gaia_at_epoch(gaia, BULK)
    return build.assemble(hst, gaia, coords, BULK)


def gaia_only(catalog):
    return catalog[catalog["POS_SOURCE"] == "GAIA"]


def test_isolated_bright_gaia_star_in_footprint_is_added():
    ra, dec = _offset(0.5, 0.5)
    catalog = run_assemble(make_hst(), make_gaia([dict(source_id=11, ra=ra, dec=dec, phot_g_mean_mag=15.0)]))
    added = gaia_only(catalog)
    assert list(added["GAIA_ID"]) == [11]
    assert added["MAG_BAND"][0] == "G"
    assert added["PM_SOURCE"][0] == "GAIA"
    assert added["FLAG"][0] == 0


def test_gaia_star_linked_by_id_is_not_added():
    ra, dec = _offset(0.5, 0.5)
    hst = make_hst(extra=[dict(RA=ra, DEC=dec, GAIA_ID=12, VMAG=18.0)])
    catalog = run_assemble(hst, make_gaia([dict(source_id=12, ra=ra + 1e-5, dec=dec, phot_g_mean_mag=18.0)]))
    assert len(gaia_only(catalog)) == 0


def test_gaia_star_within_duplicate_radius_is_not_added():
    ra, dec = _offset(0.05, 0.0)   # 0.05 arcsec from the grid star at CENTER
    catalog = run_assemble(make_hst(), make_gaia([dict(source_id=13, ra=ra, dec=dec, phot_g_mean_mag=18.0)]))
    assert len(gaia_only(catalog)) == 0


def _near_pair(gaia_row, hst_mag):
    """An HST star at (0.5, 0.5) arcsec and a Gaia star 0.2 arcsec east of it."""
    ra_h, dec_h = _offset(0.5, 0.5)
    ra_g, dec_g = _offset(0.7, 0.5)
    gaia_row.update(source_id=24, ra=ra_g, dec=dec_g)
    return make_hst(extra=[dict(RA=ra_h, DEC=dec_h, VMAG=hst_mag)]), make_gaia([gaia_row])


def test_two_parameter_star_near_hst_star_of_same_brightness_is_not_added():
    # two-parameter positions can be off by tens of mas, and the pair is one star
    hst, gaia = _near_pair(two_parameter(phot_g_mean_mag=19.0), hst_mag=19.2)
    assert len(gaia_only(run_assemble(hst, gaia))) == 0


def test_two_parameter_star_near_much_fainter_hst_star_is_added():
    hst, gaia = _near_pair(two_parameter(phot_g_mean_mag=19.0), hst_mag=21.0)
    assert list(gaia_only(run_assemble(hst, gaia))["GAIA_ID"]) == [24]


def test_five_parameter_star_near_hst_star_of_same_brightness_is_added():
    # a precise Gaia position 0.2 arcsec away is a different star
    hst, gaia = _near_pair(dict(phot_g_mean_mag=19.0), hst_mag=19.2)
    assert list(gaia_only(run_assemble(hst, gaia))["GAIA_ID"]) == [24]


def test_gaia_star_fainter_than_limit_is_not_added():
    ra, dec = _offset(0.5, 0.5)
    catalog = run_assemble(make_hst(), make_gaia([dict(source_id=14, ra=ra, dec=dec, phot_g_mean_mag=19.6)]))
    assert len(gaia_only(catalog)) == 0


def test_gaia_star_outside_footprint_is_not_added():
    ra, dec = _offset(60.0, 0.0)
    catalog = run_assemble(make_hst(), make_gaia([dict(source_id=15, ra=ra, dec=dec, phot_g_mean_mag=15.0)]))
    assert len(gaia_only(catalog)) == 0


def test_hst_rows_fainter_than_limit_are_dropped():
    ra, dec = _offset(0.5, 0.5)
    hst = make_hst(extra=[dict(RA=ra, DEC=dec, VMAG=22.6)])
    catalog = run_assemble(hst, make_gaia([dict(source_id=16, ra=0.0, dec=0.0, phot_g_mean_mag=15.0)]))
    assert np.all(catalog["VMAG"][catalog["POS_SOURCE"] == "HST"] < 22.5)
    assert np.sum(catalog["POS_SOURCE"] == "HST") == 49


def test_two_parameter_star_gets_bulk_motion_and_flag():
    ra, dec = _offset(0.5, 0.5)
    gaia = make_gaia([two_parameter(source_id=17, ra=ra, dec=dec, phot_g_mean_mag=19.0)])
    added = gaia_only(run_assemble(make_hst(), gaia))
    assert added["PM_SOURCE"][0] == "BULK"
    assert added["FLAG"][0] & build.FLAG_PM_ASSUMED
    assert added["PMRA"][0] == pytest.approx(BULK["pmra"])
    assert added["PMDEC"][0] == pytest.approx(BULK["pmdec"])
    assert added["PMRA_ERR"][0] == pytest.approx(BULK["dispersion"])


def test_two_parameter_star_is_moved_to_catalog_epoch_with_bulk_motion():
    gaia = make_gaia([two_parameter(source_id=18, ra=CENTER[0], dec=CENTER[1], phot_g_mean_mag=19.0)])
    coords = build.gaia_at_epoch(gaia, BULK)
    ddec_mas = (coords.dec.deg[0] - CENTER[1]) * 3.6e6
    # J2016.0 -> J2015.5 is -0.5 yr
    assert ddec_mas == pytest.approx(-0.5 * BULK["pmdec"], abs=1e-3)


def test_gaia_only_position_errors_include_proper_motion_over_half_a_year():
    ra, dec = _offset(0.5, 0.5)
    gaia = make_gaia([dict(source_id=23, ra=ra, dec=dec, phot_g_mean_mag=15.0, ra_error=0.03,
                           dec_error=0.04, pmra_error=0.08, pmdec_error=0.06)])
    added = gaia_only(run_assemble(make_hst(), gaia))
    assert added["RA_ERR"][0] == pytest.approx(np.hypot(0.03, 0.5 * 0.08))
    assert added["DEC_ERR"][0] == pytest.approx(np.hypot(0.04, 0.5 * 0.06))


def test_bright_hst_star_takes_better_gaia_proper_motion():
    ra, dec = _offset(0.5, 0.5)
    hst = make_hst(extra=[dict(RA=ra, DEC=dec, GAIA_ID=19, VMAG=15.0, PMRA=3.0, PMDEC=2.0,
                               PMRA_ERR=0.8, PMDEC_ERR=0.8)])
    gaia = make_gaia([dict(source_id=19, ra=ra, dec=dec, phot_g_mean_mag=15.0, pmra=1.80,
                           pmdec=0.35, pmra_error=0.02, pmdec_error=0.02)])
    catalog = run_assemble(hst, gaia)
    row = catalog[catalog["GAIA_ID"] == 19]
    assert len(row) == 1 and row["POS_SOURCE"][0] == "HST"
    assert row["PM_SOURCE"][0] == "GAIA"
    assert row["PMRA"][0] == pytest.approx(1.80)
    assert row["GMAG"][0] == pytest.approx(15.0)


def test_faint_hst_star_keeps_its_more_precise_hst_proper_motion():
    ra, dec = _offset(0.5, 0.5)
    hst = make_hst(extra=[dict(RA=ra, DEC=dec, GAIA_ID=20, VMAG=19.5)])
    gaia = make_gaia([dict(source_id=20, ra=ra, dec=dec, phot_g_mean_mag=19.5, pmra=2.5,
                           pmra_error=0.4, pmdec_error=0.4)])
    row = run_assemble(hst, gaia)
    row = row[row["GAIA_ID"] == 20]
    assert row["PM_SOURCE"][0] == "HST"
    assert row["PMRA"][0] == pytest.approx(1.875)


def test_quality_flags_from_gaia():
    ra1, dec1 = _offset(0.5, 0.5)
    ra2, dec2 = _offset(1.5, 0.5)
    gaia = make_gaia([dict(source_id=21, ra=ra1, dec=dec1, phot_g_mean_mag=15.0, ruwe=2.0),
                      dict(source_id=22, ra=ra2, dec=dec2, phot_g_mean_mag=15.0, duplicated_source=True)])
    added = gaia_only(run_assemble(make_hst(), gaia))
    flags = dict(zip(added["GAIA_ID"], added["FLAG"]))
    assert flags[21] == build.FLAG_HIGH_RUWE
    assert flags[22] == build.FLAG_GAIA_DUPLICATE


def test_measure_bulk_motion():
    hst = make_hst()
    hst["PMRA"][0] = 50.0   # one outlier must not move the median
    bulk = build.bulk_motion(hst)
    assert bulk["pmra"] == pytest.approx(1.875)
    assert bulk["pmdec"] == pytest.approx(0.376)
    assert bulk["dispersion"] >= 0


def _consistent_inputs():
    ra, dec = _offset(0.5, 0.5)
    gaia = make_gaia([dict(source_id=31, ra=ra, dec=dec, phot_g_mean_mag=15.0)])
    return run_assemble(make_hst(), gaia), gaia


def test_self_checks_pass_on_consistent_input():
    catalog, gaia = _consistent_inputs()
    build.self_checks(catalog, linked_ids=[], queried_ids=gaia["source_id"], offset_mas=(0.01, -0.02))


def test_self_checks_fail_when_linked_gaia_id_missing_from_query():
    catalog, gaia = _consistent_inputs()
    with pytest.raises(RuntimeError, match="truncated"):
        build.self_checks(catalog, linked_ids=[12345], queried_ids=gaia["source_id"], offset_mas=(0.0, 0.0))


def test_self_checks_fail_on_large_frame_offset():
    catalog, gaia = _consistent_inputs()
    with pytest.raises(RuntimeError, match="frame offset"):
        build.self_checks(catalog, [], gaia["source_id"], (0.5, 0.0))


def test_self_checks_fail_on_rows_on_top_of_each_other():
    catalog, gaia = _consistent_inputs()
    catalog.add_row(catalog[0])
    with pytest.raises(RuntimeError, match="pairs of rows"):
        build.self_checks(catalog, [], gaia["source_id"], (0.0, 0.0))


def _tiles_containing(tiles, ra, dec):
    """Indices of the half-open tiles that contain a point.

    Args:
        tiles (list of tuple): (ra_min, ra_max, dec_min, dec_max) boxes [deg].
        ra (float): right ascension [deg].
        dec (float): declination [deg].

    Returns:
        list of int: indices of the containing tiles.
    """
    return [i for i, (ra0, ra1, dec0, dec1) in enumerate(tiles) if ra0 <= ra < ra1 and dec0 <= dec < dec1]


def test_cone_tiles_cover_every_point_of_the_cone_exactly_once():
    tiles = build.cone_tiles(CENTER, 0.1, 7)
    assert len(tiles) == 49
    rng = np.random.default_rng(1)
    radius = 0.1 * np.sqrt(rng.random(2000))
    angle = rng.random(2000) * 2 * np.pi
    points = SkyCoord(CENTER[0] * u.deg, CENTER[1] * u.deg).directional_offset_by(angle * u.rad, radius * u.deg)
    edge = SkyCoord(CENTER[0] * u.deg, CENTER[1] * u.deg).directional_offset_by(
        np.linspace(0, 2 * np.pi, 360) * u.rad, 0.1 * u.deg)
    for coord in list(points) + list(edge):
        assert len(_tiles_containing(tiles, coord.ra.deg, coord.dec.deg)) == 1


def test_cone_tiles_assign_points_on_internal_edges_to_one_tile():
    tiles = build.cone_tiles(CENTER, 0.1, 7)
    ra_edge, dec_edge = tiles[0][1], tiles[0][3]   # shared upper edges of the first tile
    assert len(_tiles_containing(tiles, ra_edge, dec_edge)) == 1


def _write_inputs(tmp_path, hst, gaia):
    """Write synthetic Anderson master and saved Gaia query files.

    Args:
        tmp_path (pathlib.Path): directory to write into.
        hst (astropy.table.Table): make_hst output.
        gaia (astropy.table.Table): make_gaia output.

    Returns:
        tuple: (Anderson file path, Gaia file path) as str.
    """
    anderson = tmp_path / "master_output"
    lines = ["# synthetic Anderson master file"]
    for row in hst:
        # columns 1-3 and 13-33 are not read by the builder
        values = ["0"] * 3 + ["{0:.10f}".format(row[name]) for name in
                              ("VMAG", "RA", "DEC", "RA_ERR", "DEC_ERR", "PMRA", "PMDEC", "PMRA_ERR", "PMDEC_ERR")]
        values += ["0"] * 21 + [str(int(row["GAIA_ID"]))]
        lines.append(" ".join(values))
    anderson.write_text("\n".join(lines) + "\n")
    gaia_file = tmp_path / "gaia.ecsv"
    gaia.meta = {"adql": "SELECT synthetic", "queried": "2026-10-15"}
    gaia.write(str(gaia_file), format="ascii.ecsv")
    return str(anderson), str(gaia_file)


def _builder_inputs(tmp_path, linked_id=41):
    """Synthetic inputs: the HST grid with one star linked to Gaia, plus one Gaia-only star."""
    ra1, dec1 = _offset(0.5, 0.5)
    ra2, dec2 = _offset(1.5, 0.5)
    hst = make_hst(extra=[dict(RA=ra1, DEC=dec1, VMAG=16.0, GAIA_ID=linked_id)])
    # zero proper motion, so the linked star sits exactly on its HST position at J2015.5
    gaia = make_gaia([dict(source_id=41, ra=ra1, dec=dec1, phot_g_mean_mag=16.0, pmra=0.0, pmdec=0.0),
                      dict(source_id=42, ra=ra2, dec=dec2, phot_g_mean_mag=15.0)])
    return _write_inputs(tmp_path, hst, gaia)


def test_builder_writes_catalogue_with_provenance(tmp_path):
    anderson, gaia_file = _builder_inputs(tmp_path)
    output = str(tmp_path / "catalog.ecsv")
    build.build_catalog(output, anderson_file=anderson, gaia_file=gaia_file)
    catalog = astrom.read_field_catalog(output)
    assert len(catalog) == 51
    assert list(catalog["GAIA_ID"][catalog["POS_SOURCE"] == "GAIA"]) == [42]
    assert np.all(np.diff(np.asarray(catalog["VMAG"], float)) >= 0)
    assert catalog["RA"].unit == u.deg and catalog["PMRA"].unit == u.mas / u.yr
    assert catalog.meta["epoch"] == 2015.5
    assert catalog.meta["sources"][1]["adql"] == "SELECT synthetic"
    assert len(catalog.meta["sources"][0]["sha256"]) == 64


def test_failed_build_leaves_existing_catalogue_untouched(tmp_path):
    anderson, gaia_file = _builder_inputs(tmp_path, linked_id=43)   # 43 is not in the Gaia query
    output = tmp_path / "catalog.ecsv"
    output.write_text("previous catalogue")
    with pytest.raises(RuntimeError, match="truncated"):
        build.build_catalog(str(output), anderson_file=anderson, gaia_file=gaia_file)
    assert output.read_text() == "previous catalogue"
    assert [p.name for p in tmp_path.iterdir() if p.name.startswith("catalog")] == ["catalog.ecsv"]


# ---- the catalogue shipped in corgidrp/data

# stars added by hand in PR #1057; its 4657987346473687296 (a two-parameter solution) is the
# unresolved blend of two HST stars 0.45 arcsec apart, both of which are in the catalogue
PR1057_KEPT_IDS = [4657987350779055232, 4657987346506841344, 4657987346439013120,
                   4657987350829299328, 4657987385189011456]
REQUIRED_COLUMNS = {"GAIA_ID": None, "RA": u.deg, "DEC": u.deg, "RA_ERR": u.mas, "DEC_ERR": u.mas,
                    "VMAG": u.mag, "MAG_BAND": None, "GMAG": u.mag, "PMRA": u.mas / u.yr,
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
    for name in ("RA", "DEC", "VMAG", "PMRA", "PMDEC", "PMRA_ERR", "PMDEC_ERR"):
        assert np.all(np.isfinite(np.asarray(catalog[name], float))), name
    assert set(catalog["POS_SOURCE"]) <= {"HST", "GAIA"}
    assert set(catalog["PM_SOURCE"]) <= {"HST", "GAIA", "BULK"}
    assert set(catalog["MAG_BAND"]) <= {"F606W", "G"}
    hst = catalog["POS_SOURCE"] == "HST"
    assert np.all(np.asarray(catalog["VMAG"][hst], float) < 22.5)
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
