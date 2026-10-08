"""Tests for tools/build_lmc_astrom_catalog.py on small synthetic catalogues (no network)."""
import importlib.util
import os

import numpy as np
import pytest
from astropy.coordinates import SkyCoord
from astropy.table import Table
import astropy.units as u

_script = os.path.join(os.path.dirname(__file__), "..", "tools", "build_lmc_astrom_catalog.py")
_spec = importlib.util.spec_from_file_location("build_lmc_astrom_catalog", _script)
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
        astropy.table.Table: Anderson-style table as returned by parse_anderson_master.
    """
    rows = []
    for dx in range(-3, 4):
        for dy in range(-3, 4):
            ra, dec = _offset(dx, dy)
            rows.append(dict(MAG=20.0, RA=ra, DEC=dec, RA_ERR=0.2, DEC_ERR=0.2, PMRA=1.875,
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
        astropy.table.Table: table in the Gaia convention used by the build script.
    """
    defaults = dict(ra_error=0.05, dec_error=0.05, pmra=1.9, pmra_error=0.03, pmdec=0.3,
                    pmdec_error=0.03, parallax=0.0, parallax_error=0.03, ruwe=1.0,
                    astrometric_params_solved=31, duplicated_source=False)
    full = []
    for row in rows:
        r = dict(defaults)
        r.update(row)
        full.append(r)
    names = ["source_id", "ra", "dec", "ra_error", "dec_error", "pmra", "pmra_error", "pmdec",
             "pmdec_error", "parallax", "parallax_error", "phot_g_mean_mag", "ruwe",
             "astrometric_params_solved", "duplicated_source"]
    table = Table(rows=[[r[n] for n in names] for r in full], names=names)
    table["source_id"] = table["source_id"].astype(np.int64)
    return table


def run_assemble(hst, gaia):
    coords = build.gaia_positions_at_catalog_epoch(gaia, BULK)
    return build.assemble_catalog(hst, gaia, coords, BULK)


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
    hst = make_hst(extra=[dict(RA=ra, DEC=dec, GAIA_ID=12, MAG=18.0)])
    catalog = run_assemble(hst, make_gaia([dict(source_id=12, ra=ra + 1e-5, dec=dec, phot_g_mean_mag=18.0)]))
    assert len(gaia_only(catalog)) == 0


def test_gaia_star_within_duplicate_radius_is_not_added():
    ra, dec = _offset(0.05, 0.0)   # 0.05 arcsec from the grid star at CENTER
    catalog = run_assemble(make_hst(), make_gaia([dict(source_id=13, ra=ra, dec=dec, phot_g_mean_mag=18.0)]))
    assert len(gaia_only(catalog)) == 0


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
    hst = make_hst(extra=[dict(RA=ra, DEC=dec, MAG=22.6)])
    catalog = run_assemble(hst, make_gaia([dict(source_id=16, ra=0.0, dec=0.0, phot_g_mean_mag=15.0)]))
    assert np.all(catalog["MAG"][catalog["POS_SOURCE"] == "HST"] < 22.5)
    assert np.sum(catalog["POS_SOURCE"] == "HST") == 49


def test_two_parameter_star_gets_bulk_motion_and_flag():
    ra, dec = _offset(0.5, 0.5)
    gaia = make_gaia([dict(source_id=17, ra=ra, dec=dec, phot_g_mean_mag=19.0, pmra=np.nan,
                           pmdec=np.nan, pmra_error=np.nan, pmdec_error=np.nan,
                           parallax=np.nan, parallax_error=np.nan, astrometric_params_solved=3)])
    added = gaia_only(run_assemble(make_hst(), gaia))
    assert added["PM_SOURCE"][0] == "BULK"
    assert added["FLAG"][0] & build.FLAG_PM_ASSUMED
    assert added["PMRA"][0] == pytest.approx(BULK["pmra"])
    assert added["PMDEC"][0] == pytest.approx(BULK["pmdec"])
    assert added["PMRA_ERR"][0] == pytest.approx(BULK["dispersion"])


def test_two_parameter_star_is_moved_to_catalog_epoch_with_bulk_motion():
    gaia = make_gaia([dict(source_id=18, ra=CENTER[0], dec=CENTER[1], phot_g_mean_mag=19.0,
                           pmra=np.nan, pmdec=np.nan)])
    coords = build.gaia_positions_at_catalog_epoch(gaia, BULK)
    ddec_mas = (coords.dec.deg[0] - CENTER[1]) * 3.6e6
    # J2016.0 -> J2015.5 is -0.5 yr
    assert ddec_mas == pytest.approx(-0.5 * BULK["pmdec"], abs=1e-3)


def test_bright_hst_star_takes_better_gaia_proper_motion():
    ra, dec = _offset(0.5, 0.5)
    hst = make_hst(extra=[dict(RA=ra, DEC=dec, GAIA_ID=19, MAG=15.0, PMRA=3.0, PMDEC=2.0,
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
    hst = make_hst(extra=[dict(RA=ra, DEC=dec, GAIA_ID=20, MAG=19.5)])
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
    bulk = build.measure_bulk_motion(hst)
    assert bulk["pmra"] == pytest.approx(1.875)
    assert bulk["pmdec"] == pytest.approx(0.376)
    assert bulk["dispersion"] >= 0


def _consistent_inputs():
    hst = make_hst()
    ra, dec = _offset(0.5, 0.5)
    gaia = make_gaia([dict(source_id=31, ra=ra, dec=dec, phot_g_mean_mag=15.0)])
    catalog = run_assemble(hst, gaia)
    legacy = Table({"VMAG": hst["MAG"], "RA": hst["RA"], "DEC": hst["DEC"]})
    return catalog, legacy, gaia


def test_self_checks_pass_on_consistent_input():
    catalog, legacy, gaia = _consistent_inputs()
    build.run_self_checks(catalog, legacy, linked_ids=[], queried_ids=gaia["source_id"],
                          offset_mas=(0.01, -0.02), required_ids=[31], excluded_ids=[99])


def test_self_checks_fail_on_changed_legacy_row():
    catalog, legacy, gaia = _consistent_inputs()
    legacy["RA"][3] += 1e-10
    with pytest.raises(RuntimeError, match="legacy"):
        build.run_self_checks(catalog, legacy, [], gaia["source_id"], (0.0, 0.0))


def test_self_checks_fail_when_linked_gaia_id_missing_from_query():
    catalog, legacy, gaia = _consistent_inputs()
    with pytest.raises(RuntimeError, match="truncated"):
        build.run_self_checks(catalog, legacy, linked_ids=[12345], queried_ids=gaia["source_id"],
                              offset_mas=(0.0, 0.0))


def test_self_checks_fail_on_required_or_excluded_ids():
    catalog, legacy, gaia = _consistent_inputs()
    with pytest.raises(RuntimeError, match="required"):
        build.run_self_checks(catalog, legacy, [], gaia["source_id"], (0.0, 0.0), required_ids=[777])
    with pytest.raises(RuntimeError, match="excluded"):
        build.run_self_checks(catalog, legacy, [], gaia["source_id"], (0.0, 0.0), excluded_ids=[31])


def test_self_checks_fail_on_large_frame_offset():
    catalog, legacy, gaia = _consistent_inputs()
    with pytest.raises(RuntimeError, match="frame offset"):
        build.run_self_checks(catalog, legacy, [], gaia["source_id"], (0.5, 0.0))


def test_written_catalog_reads_back_bit_identical(tmp_path):
    catalog, legacy, gaia = _consistent_inputs()
    # the real legacy file has 10 (RA), 9 (DEC) and up to 4 (VMAG) decimals
    legacy["RA"] = np.round(np.asarray(legacy["RA"], float), 10)
    legacy["DEC"] = np.round(np.asarray(legacy["DEC"], float), 9)
    legacy["VMAG"] = np.round(np.asarray(legacy["VMAG"], float), 4)
    hst_rows = catalog["POS_SOURCE"] == "HST"
    catalog["RA"][hst_rows] = legacy["RA"]
    catalog["DEC"][hst_rows] = legacy["DEC"]
    catalog["MAG"][hst_rows] = legacy["VMAG"]
    build.apply_units_and_formats(catalog)
    path = str(tmp_path / "cat.ecsv")
    catalog.write(path, format="ascii.ecsv")
    readback = Table.read(path, format="ascii.ecsv")
    build.run_self_checks(readback, legacy, [], gaia["source_id"], (0.0, 0.0))


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
