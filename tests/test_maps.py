import numpy as np
import pytest

from guidedvina.maps import (
    ACCEPTOR_TYPES,
    MapError,
    apply_bias_to_maps,
    find_maps,
    gaussian_well,
    read_map,
    write_map,
)


def _write_map(path, nel=8, spacing=1.0, center=(0.0, 0.0, 0.0), fill=0.0):
    n = nel + 1
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("GRID_PARAMETER_FILE NULL\nGRID_DATA_FILE NULL\nMACROMOLECULE r.pdbqt\n")
        fh.write(f"SPACING {spacing}\nNELEMENTS {nel} {nel} {nel}\n")
        fh.write(f"CENTER {center[0]:.3f} {center[1]:.3f} {center[2]:.3f}\n")
        for _ in range(n ** 3):
            fh.write(f"{fill:.4f}\n")
    return path


def test_roundtrip_preserves_values(tmp_path):
    p = _write_map(tmp_path / "r.C_H.map", fill=0.25)
    m = read_map(p)
    assert m.nelements == (8, 8, 8)
    assert m.values.shape == (9, 9, 9)
    assert np.allclose(m.values, 0.25)
    out = write_map(m, tmp_path / "copy.C_H.map")
    assert np.allclose(read_map(out).values, m.values)


def test_truncated_map_is_rejected(tmp_path):
    p = tmp_path / "bad.C_H.map"
    p.write_text(
        "SPACING 1.0\nNELEMENTS 4 4 4\nCENTER 0 0 0\n" + "0.0\n" * 10,
        encoding="utf-8",
    )
    with pytest.raises(MapError):
        read_map(p)


def test_origin_and_grid_coordinates(tmp_path):
    m = read_map(_write_map(tmp_path / "r.C_H.map", nel=8, spacing=0.5, center=(1.0, 2.0, 3.0)))
    assert np.allclose(m.origin, [1.0 - 2.0, 2.0 - 2.0, 3.0 - 2.0])
    xs, ys, zs = m.grid_coordinates()
    assert len(xs) == 9 and xs[0] == pytest.approx(-1.0) and xs[-1] == pytest.approx(3.0)


def test_well_is_deepest_at_the_anchor(tmp_path):
    m = read_map(_write_map(tmp_path / "r.C_H.map", nel=10, spacing=1.0))
    anchor = (2.0, 0.0, 0.0)
    values, changed = gaussian_well(m, anchor, depth=2.0, radius=4.0)
    assert changed > 0
    xs, ys, zs = m.grid_coordinates()
    ix = int(np.argmin(np.abs(xs - 2.0)))
    iy = int(np.argmin(np.abs(ys)))
    iz = int(np.argmin(np.abs(zs)))
    assert values[iz, iy, ix] == pytest.approx(-2.0, abs=1e-6)
    # Far from the anchor the map is untouched.
    assert values[0, 0, 0] == pytest.approx(0.0)
    assert values.min() >= -2.0 - 1e-9


def test_well_does_not_carve_into_the_receptor(tmp_path):
    """Voxels already strongly repulsive must stay repulsive."""
    m = read_map(_write_map(tmp_path / "r.C_H.map", nel=6, spacing=1.0, fill=50.0))
    values, changed = gaussian_well(m, (0.0, 0.0, 0.0), depth=2.0, radius=4.0)
    assert changed == 0
    assert np.allclose(values, 50.0)


def test_typed_bias_touches_only_matching_maps(tmp_path):
    for t in ("C_H", "O_A", "N_D"):
        _write_map(tmp_path / f"base.{t}.map", nel=8, spacing=1.0)
    app = apply_bias_to_maps(
        tmp_path / "base", tmp_path / "biased",
        anchor=(0.0, 0.0, 0.0), depth=1.5, radius=3.0, target="acc",
    )
    assert app.maps_biased == ["O_A"]
    assert len(app.maps_written) == 3          # every map is copied through
    assert set(ACCEPTOR_TYPES) >= set(app.maps_biased)
    assert read_map(tmp_path / "biased.C_H.map").values.min() == pytest.approx(0.0)
    assert read_map(tmp_path / "biased.O_A.map").values.min() == pytest.approx(-1.5)


def test_untargeted_map_still_written(tmp_path):
    for t in ("C_H", "O_A"):
        _write_map(tmp_path / f"base.{t}.map")
    apply_bias_to_maps(tmp_path / "base", tmp_path / "out",
                       anchor=(0, 0, 0), depth=1.0, radius=3.0, target="acc")
    assert len(find_maps(tmp_path / "out")) == 2


def test_missing_maps_is_an_error(tmp_path):
    with pytest.raises(MapError, match="No maps found"):
        apply_bias_to_maps(tmp_path / "nothing", tmp_path / "out",
                           anchor=(0, 0, 0), depth=1.0, radius=3.0)


def test_anchor_outside_the_grid_is_an_error(tmp_path):
    """Silently docking unbiased would be worse than failing."""
    _write_map(tmp_path / "base.C_H.map", nel=8, spacing=1.0)
    with pytest.raises(MapError, match="no voxel was changed|matched no map"):
        apply_bias_to_maps(tmp_path / "base", tmp_path / "out",
                           anchor=(500.0, 500.0, 500.0), depth=2.0, radius=3.0)


def test_unknown_target_is_rejected(tmp_path):
    _write_map(tmp_path / "base.C_H.map")
    with pytest.raises(MapError, match="Unknown bias target"):
        apply_bias_to_maps(tmp_path / "base", tmp_path / "out",
                           anchor=(0, 0, 0), depth=1.0, radius=3.0, target="aromatic")
