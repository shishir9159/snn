import pathlib
import sys
import zipfile

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import nmnist_small


def test_recordings_land_in_digit_folders_named_by_sample(tmp_path):
    with zipfile.ZipFile(tmp_path / "NMNISTsmall.zip", "w") as z:
        z.writestr("NMNISTsmall/train1K.txt", "#sample #class\n1\t5\n2\t0\n")
        z.writestr("NMNISTsmall/test100.txt", "#sample #class\n60001\t7\n")
        for sample in ("1", "2", "60001"):
            z.writestr(f"NMNISTsmall/{sample}.bs2", bytes([int(sample) % 256] * 5))
    assert nmnist_small.lay_out(tmp_path) == {"Train": 2, "Test": 1}
    assert (tmp_path / "Train" / "5" / "00001.bin").read_bytes() == bytes([1] * 5)
    assert (tmp_path / "Train" / "0" / "00002.bin").exists()
    assert (tmp_path / "Test" / "7" / "60001.bin").read_bytes() == bytes([60001 % 256] * 5)
