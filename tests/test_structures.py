import numpy as np

from guidedvina.structures import count_pdbqt_torsions, read_first_model, read_models


def test_parses_chain_and_residue(examples_dir):
    model = read_first_model(examples_dir / "sample_receptor.pdbqt")
    assert model.atoms
    assert {a.chain for a in model.atoms} == {"A"}
    assert 12 in {a.res_seq for a in model.atoms}


def test_dimer_has_two_chains(examples_dir):
    model = read_first_model(examples_dir / "sample_receptor_dimer.pdbqt")
    assert {a.chain for a in model.atoms} == {"A", "B"}


def test_torsion_count_read_from_remark(examples_dir):
    assert count_pdbqt_torsions(examples_dir / "sample_ligand.pdbqt") == 4


def test_mercury_is_not_hydrogen(tmp_path):
    p = tmp_path / "hg.pdb"
    p.write_text(
        "ATOM      1  HG  HOH A   1       0.000   0.000   0.000  1.00  0.00          HG\n",
        encoding="utf-8",
    )
    atoms = read_first_model(p).atoms
    assert len(atoms) == 1
    assert atoms[0].is_hydrogen is False
    assert atoms[0].element == "Hg"


def test_autodock_types_map_to_elements(tmp_path):
    p = tmp_path / "t.pdbqt"
    p.write_text(
        "ATOM      1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00     0.000 A \n"
        "ATOM      2  O1  LIG A   1       1.000   0.000   0.000  1.00  0.00     0.000 OA\n"
        "ATOM      3  H1  LIG A   1       2.000   0.000   0.000  1.00  0.00     0.000 HD\n",
        encoding="utf-8",
    )
    model = read_first_model(p)
    assert [a.element for a in model.atoms] == ["C", "O", "H"]
    assert len(model.heavy()) == 2


def test_multi_model_split(tmp_path):
    body = (
        "MODEL 1\nREMARK VINA RESULT:  -7.5 0 0\n"
        "ATOM      1  C1  LIG A   1       0.000   0.000   0.000  1.00  0.00     0.000 C \n"
        "ENDMDL\nMODEL 2\nREMARK VINA RESULT:  -7.1 0 0\n"
        "ATOM      1  C1  LIG A   1       1.000   0.000   0.000  1.00  0.00     0.000 C \n"
        "ENDMDL\n"
    )
    p = tmp_path / "m.pdbqt"
    p.write_text(body, encoding="utf-8")
    models = read_models(p)
    assert len(models) == 2
    assert models[0].score() == -7.5
    assert np.allclose(models[1].coords()[0], [1.0, 0.0, 0.0])
