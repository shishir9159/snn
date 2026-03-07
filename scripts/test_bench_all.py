import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))
import bench_all as b

PROFILE = """7410 events per image
events                  5343.8 us/img
network                 4916.3 us/img
network + learning      6715.7 us/img
"""


def test_counts_expand_from_their_suffix():
    assert [b.expand(c) for c in ["950", "1.2K", "56K", "7.8M", "2.0G"]] == [
        950.0, 1200.0, 56000.0, 7.8e6, 2e9]


def test_phases_come_out_of_profile_output():
    assert b.parse_phases(PROFILE) == {
        "events": 5343.8, "network": 4916.3, "network + learning": 6715.7}
    assert b.parse_phases("nothing here") == {}


def test_epoch_line_gives_accuracy_speed_activity_and_energy():
    line = ("epoch 1 done: train 74.1%  test 82.8%  (85 img/s)  "
            "7.7K spikes 720K synops 11K MACs / img  699 nJ vs 1.50 µJ dense")
    assert b.parse_epoch("noise\n" + line + "\n") == {
        "train %": 74.1, "test %": 82.8, "img/s": 85.0,
        "spikes/img": 7700.0, "synops/img": 720000.0, "MACs/img": 11000.0,
        "nJ/img": 699.0, "dense nJ": 1500.0}
    plain = "epoch 1 done: train 1.0%  test 2.0%  (9 img/s)  5 spikes 40 synops / img  3.46 nJ"
    got = b.parse_epoch(plain)
    assert (got["MACs/img"], got["nJ/img"], got["dense nJ"]) == (0.0, 3.46, 0.0)


def test_epoch_line_without_counts_and_the_last_epoch_wins():
    old = "epoch 1 done: train 10.0%  test 11.0%  (5 img/s)"
    new = "epoch 2 done: train 20.0%  test 21.0%  (6 img/s)"
    got = b.parse_epoch(old + "\n" + new)
    assert (got["test %"], got["synops/img"]) == (21.0, 0.0)
    assert b.parse_epoch("epoch 1    30/300  running acc  1.0%") is None


def test_best_of_keeps_the_fastest_round_per_side():
    times = {"base": iter([{"a": 10.0}, {"a": 8.0}, {"a": 9.0}]),
             "new": iter([{"a": 7.0}, {"a": 9.0}, {"a": 6.0}])}
    best = b.best_of(3, {k: (lambda it=it: next(it)) for k, it in times.items()})
    assert best == {"base": {"a": 8.0}, "new": {"a": 6.0}}


def test_report_shows_the_change_against_a_base(capsys):
    rows = [("wta", "present", 90.0, 100.0), ("pcl", "network", 50.0, None)]
    quality = [("wta", 1, {"train %": 66.8, "test %": 71.3, "img/s": 12840.0,
                           "spikes/img": 769.0, "synops/img": 31000.0, "MACs/img": 0.0,
                           "nJ/img": 27.9, "dense nJ": 144.0})]
    b.report(rows, quality, 5, 6000, 1000)
    out = capsys.readouterr().out
    assert "-10.0% vs 100.0" in out
    assert "best of 5" in out and "6000 training and 1000 test" in out
    assert "71.3" in out and "31,000" in out and "epochs" in out
    assert "0.19" in out, "energy against the dense ANN"
    pcl = next(line for line in out.splitlines() if "network" in line)
    assert "%" not in pcl


def test_an_unknown_model_name_is_refused():
    with pytest.raises(SystemExit):
        b.main(["--only", "nope"])
