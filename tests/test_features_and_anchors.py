import pytest

from guidedvina.anchors import (
    AnchorError,
    LeakageError,
    anchor_from_reference_ligand,
    anchor_from_residues,
    parse_residue_specs,
)
from guidedvina.features import MissingDescriptorError, ligand_features, pocket_features


def test_residue_spec_forms():
    specs = parse_residue_specs(["A:12", "60", "B:70-72"])
    assert [str(s) for s in specs] == ["A:12", "60", "B:70", "B:71", "B:72"]


def test_bad_residue_spec_rejected():
    with pytest.raises(AnchorError):
        parse_residue_specs(["GLY12"])


def test_rotatable_bonds_never_guessed_silently(tmp_path):
    p = tmp_path / "lig.pdb"
    p.write_text(
        "HETATM    1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00           C\n"
        "HETATM    2  C2  LIG A   1       1.500   0.000   0.000  1.00  0.00           C\n",
        encoding="utf-8",
    )
    try:
        import rdkit  # noqa: F401
        pytest.skip("RDKit present: counts are measured, not estimated")
    except ImportError:
        pass
    with pytest.raises(MissingDescriptorError):
        ligand_features(p)
    feats = ligand_features(p, allow_estimate=True)
    assert feats.rotatable_bonds_source == "estimated"
    assert feats.is_estimated


def test_anchor_uses_sidechains_of_named_residues(examples_dir):
    anchor = anchor_from_residues(
        examples_dir / "sample_receptor.pdbqt",
        parse_residue_specs(["A:12", "A:60", "A:61"]),
        center=(0, 0, 0), size=(20, 20, 20),
    )
    assert anchor.source == "residues"
    assert anchor.chains_matched == ["A"]
    assert anchor.atom_count == 12
    assert not anchor.warnings


def test_anchor_warns_when_residue_matches_several_chains(examples_dir):
    anchor = anchor_from_residues(
        examples_dir / "sample_receptor_dimer.pdbqt",
        parse_residue_specs(["12", "60"]),
        center=(0, 0, 0), size=(60, 60, 60),
    )
    assert anchor.chains_matched == ["A", "B"]
    assert any("chains" in w for w in anchor.warnings)


def test_anchor_rejects_residues_outside_box(examples_dir):
    with pytest.raises(AnchorError):
        anchor_from_residues(
            examples_dir / "sample_receptor.pdbqt",
            parse_residue_specs(["A:95"]),
            center=(100, 100, 100), size=(10, 10, 10),
        )


def test_missing_residue_is_an_error(examples_dir):
    with pytest.raises(AnchorError):
        anchor_from_residues(
            examples_dir / "sample_receptor.pdbqt",
            parse_residue_specs(["A:999"]),
            center=(0, 0, 0), size=(20, 20, 20),
        )


def test_reference_ligand_anchor_is_refused_by_default(examples_dir):
    with pytest.raises(LeakageError):
        anchor_from_reference_ligand(examples_dir / "sample_ligand_ref.pdbqt")
    anchor = anchor_from_reference_ligand(
        examples_dir / "sample_ligand_ref.pdbqt", allow_leakage=True
    )
    assert "LEAKY" in anchor.source


def test_pocket_features(examples_dir):
    pf = pocket_features(examples_dir / "sample_receptor.pdbqt", (0, 0, 0), (20, 20, 20))
    assert pf.box_volume == 8000.0
    assert pf.heavy_atoms_in_box > 0
    assert 0.0 <= pf.buriedness <= 1.0
