import pathlib
import subprocess
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import check_vectors as cv

ROOT = pathlib.Path(__file__).resolve().parent.parent
RUST = ROOT / "target" / "release" / ("mnist-snn-rs" + (".exe" if sys.platform == "win32" else ""))

CONFIG = {"inputs": 2, "outputs": 2, "steps": 4, "weight_bits": 2, "frac_bits": 0, "gain_bits": 0,
          "base_bits": 0, "offset": 0, "leak": 1, "rest": 0, "floor": -10, "t_ref": 1, "div": 1}


def tiny(tmp_path, trace="2 0\n0 -10\n0 -10\n0 -9\n"):
    """Two inputs, two outputs, worked by hand: neuron 0 wins and fires at step 2."""
    (tmp_path / "config.txt").write_text("".join(f"{k} {v}\n" for k, v in CONFIG.items()))
    (tmp_path / "weights.txt").write_text("3 0\n1 1\n")
    (tmp_path / "gains.txt").write_text("2\n3\n")
    (tmp_path / "0000.in").write_text("label 1\n1 0\n2 0 1\n4 1\n")
    (tmp_path / "0000.out").write_text("winner 0\n2 0\n")
    (tmp_path / "0000.trace").write_text(trace)
    return tmp_path


def test_hand_worked_vectors_replay(tmp_path):
    assert cv.check(tiny(tmp_path)) == []


def test_a_wrong_potential_is_found(tmp_path):
    problems = cv.check(tiny(tmp_path, trace="2 0\n0 -10\n0 -10\n0 -8\n"))
    assert problems == ["0000: potentials differ from step 4"]


def test_an_empty_directory_is_a_problem(tmp_path):
    (tmp_path / "config.txt").write_text("outputs 1\n")
    (tmp_path / "weights.txt").write_text("1\n")
    (tmp_path / "gains.txt").write_text("1\n")
    assert "no *.in files" in cv.check(tmp_path)[0]


@pytest.mark.skipif(not RUST.exists(), reason="needs a release build of mnist-snn-rs")
def test_vectors_from_the_rust_model_replay(tmp_path):
    out = tmp_path / "v"
    subprocess.run([str(RUST), "--headless", "--train", "200", "--test", "6", "--outputs", "8",
                    "--set", "weight_bits=6", "--vectors", str(out)],
                   check=True, capture_output=True, cwd=ROOT)
    assert cv.check(out) == []
    assert len(list(out.glob("*.trace"))) == 6
