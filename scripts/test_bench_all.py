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


def figures(test, nj):
    return {"train %": 66.8, "test %": test, "img/s": 12840.0, "spikes/img": 769.0,
            "synops/img": 31000.0, "MACs/img": 0.0, "nJ/img": nj, "dense nJ": 144.0}


def test_summary_is_the_mean_and_sample_spread_over_seeds():
    s = b.summary([figures(70.0, 27.0), figures(74.0, 29.0), figures(72.0, 28.0)])
    assert s["test %"] == (72.0, 2.0)
    assert s["nJ/img"][0] == 28.0 and s["train %"][1] == 0.0
    assert b.summary([figures(70.0, 27.0)])["test %"] == (70.0, None)


def test_report_shows_the_change_against_a_base_and_the_seed_spread(capsys):
    rows = [("wta", "present", 90.0, 100.0), ("pcl", "network", 50.0, None)]
    quality = [("wta", 1, [figures(70.0, 27.0), figures(74.0, 29.0)])]
    b.report(rows, quality, 5, 6000, 1000, [1, 2])
    out = capsys.readouterr().out
    assert "-10.0% vs 100.0" in out
    assert "best of 5" in out and "6000 training and 1000 test" in out
    assert "seeds 1-2" in out and "72.0 ± 2.8" in out
    assert "31,000" in out and "epochs" in out
    assert "0.19" in out, "energy against the dense ANN"
    pcl = next(line for line in out.splitlines() if "network" in line)
    assert "%" not in pcl


ROBUST = """epoch 1 done: train 66.8%  test 71.3%  (19137 img/s)
robustness: clean 71.3%
robustness: pixel noise 0.1 71.7%  0.2 72.5%
robustness: event dropout 0.1 70.4%  0.7 67.5%
"""


def test_robustness_lines_become_figures():
    assert b.parse_robustness(ROBUST) == {
        "robust clean": 71.3, "robust pixel noise 0.1": 71.7, "robust pixel noise 0.2": 72.5,
        "robust event dropout 0.1": 70.4, "robust event dropout 0.7": 67.5}
    assert b.parse_robustness("epoch 1 done") == {}


def test_robustness_curves_average_seeds_and_start_clean(tmp_path):
    seed = lambda shift: {"test %": 70.0, **{k: v + shift for k, v in b.parse_robustness(ROBUST).items()}}  # noqa: E731
    rows = [("wta", 1, [seed(0.0), seed(2.0)]), ("pcl", 1, [{"test %": 1.0}])]
    curves = b.robustness_curves(rows)
    assert list(curves) == ["wta"], "a model run without --robustness has no curves"
    noise = curves["wta"]["pixel noise"]
    assert noise[0] == (0.0, 72.3, pytest.approx(1.414, abs=1e-3))
    assert [lv for lv, _, _ in noise] == [0.0, 0.1, 0.2]
    chart = tmp_path / "r.svg"
    b.robustness_svg(curves, chart)
    svg = chart.read_text(encoding="utf-8")
    assert svg.count("<polyline") == 2 and "wta · event dropout 0.7: 68.5% ± 1.4" in svg


def test_an_unknown_model_name_is_refused():
    with pytest.raises(SystemExit):
        b.main(["--only", "nope"])
