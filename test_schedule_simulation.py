"""11_schedule_simulation.py 테스트 (pytest 없이 실행: python test_schedule_simulation.py)"""
import importlib.util

import numpy as np
import pandas as pd

_spec = importlib.util.spec_from_file_location("sim", "11_schedule_simulation.py")
sim = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sim)

CAP = 179.0
A = lambda *v: np.array(v, dtype=float)
B = lambda *v: np.array(v, dtype=bool)


def test_x_zero_changes_nothing():
    p = A(100, 100, 200, 100)
    adj, moves, unmoved = sim.shift_load(p, p, B(1, 1, 1, 1), B(0, 0, 1, 0), 0.0, CAP, "any")
    assert np.allclose(adj, p) and moves == [] and unmoved == 0.0


def test_any_fills_lowest_expected_first_up_to_hourly_limit():
    adj, moves, unmoved = sim.shift_load(A(100, 100, 200, 100), A(100, 170, 200, 150),
                                         B(1, 1, 1, 1), B(0, 0, 1, 0), 0.2, CAP, "any")
    assert np.allclose(adj, [120, 100, 160, 120]) and unmoved == 0.0     # 0번은 100의 20% = 20까지만
    assert moves == [(2, 0, 20.0), (2, 3, 20.0)]


def test_hourly_limit_leaves_rest_unmoved():
    adj, moves, unmoved = sim.shift_load(A(200, 100), A(200, 100), B(1, 1), B(1, 0), 0.2, CAP, "any")
    assert np.allclose(adj, [180, 120]) and abs(unmoved - 20.0) < 1e-9


def test_cap_and_limit_spill_to_next_candidates():
    adj, moves, unmoved = sim.shift_load(A(100, 100, 200, 100), A(100, 170, 200, 150),
                                         B(1, 1, 1, 1), B(0, 0, 1, 0), 0.5, CAP, "any")
    assert np.allclose(adj, [150, 109, 112, 129]) and abs(unmoved - 12.0) < 1e-9   # 한도 50, 상한 29, 상한 9
    assert [(s, d) for s, d, _ in moves] == [(2, 0), (2, 3), (2, 1)]


def test_after_only_later_hours_and_unmoved():
    adj, moves, unmoved = sim.shift_load(A(100, 100, 200, 100), A(100, 170, 200, 150),
                                         B(1, 1, 1, 1), B(0, 0, 1, 0), 0.2, CAP, "after")
    assert np.allclose(adj, [100, 100, 171, 129]) and abs(unmoved - 11.0) < 1e-9


def test_after_alert_at_last_hour_moves_nothing():
    p = A(100, 100, 200)
    adj, moves, unmoved = sim.shift_load(p, p, B(1, 1, 1), B(0, 0, 1), 0.1, CAP, "after")
    assert np.allclose(adj, p) and moves == [] and abs(unmoved - 20.0) < 1e-9


def test_after_received_then_alerted_cuts_adjusted_load():
    adj, _, unmoved = sim.shift_load(A(200, 100, 100), A(200, 100, 100),
                                     B(1, 1, 1), B(1, 1, 0), 0.1, CAP, "after")
    assert np.allclose(adj, [180, 110, 110])
    assert abs(unmoved - 11.0) < 1e-9          # 1번은 받은 뒤 110의 10% = 11을 깎으려 했고 2번은 한도가 참


def test_any_never_moves_into_alert_hour():
    adj, _, _ = sim.shift_load(A(200, 190, 100, 100), A(100, 100, 150, 150),
                               B(1, 1, 1, 1), B(1, 1, 0, 0), 0.1, CAP, "any")
    assert adj[0] < 200 and adj[1] < 190 and abs(adj.sum() - 590.0) < 1e-9


def test_unplanned_hours_never_receive():
    adj, _, unmoved = sim.shift_load(A(200, 50, 100), A(200, 50, 100),
                                     B(1, 0, 1), B(1, 0, 0), 0.1, CAP, "any")
    assert np.allclose(adj, [190, 50, 110]) and abs(unmoved - 10.0) < 1e-9


def test_no_planned_hours():
    p = A(190, 30, 30)
    adj, moves, unmoved = sim.shift_load(p, p, B(0, 0, 0), B(1, 0, 0), 0.2, CAP, "any")
    assert np.allclose(adj, p) and moves == [] and abs(unmoved - 38.0) < 1e-9


def test_no_room():
    p = A(200, 100, 100)
    adj, moves, unmoved = sim.shift_load(p, A(200, 179, 185), B(1, 1, 1), B(1, 0, 0), 0.1, CAP, "any")
    assert np.allclose(adj, p) and moves == [] and abs(unmoved - 20.0) < 1e-9


def test_simulate_keeps_daily_total_and_never_crosses_days():
    idx = pd.date_range("2021-09-01", periods=48, freq="h")
    power = np.full(48, 100.0); power[10] = 200.0; power[34] = 190.0
    frame = pd.DataFrame({"power": power, "planned_on": 1}, index=idx)
    alert = power >= CAP
    adj, moves, unmoved = sim.simulate(frame, alert, power, 0.1, "after", CAP)
    assert abs(adj[:24].sum() - power[:24].sum()) < 1e-6 and abs(adj[24:].sum() - power[24:].sum()) < 1e-6
    assert (moves["from"].dt.normalize() == moves["to"].dt.normalize()).all()
    assert list(moves.columns) == ["from", "to", "kw"] and unmoved == 0.0
    assert adj[10] == 180.0 and adj[34] == 171.0


def test_metrics_counts():
    power, adj = A(170, 190, 100), A(185, 171, 114)
    m = sim.metrics(power, adj, B(0, 1, 0), 1, CAP, 19.0, 0.0)
    assert (m["max_before"], m["max_after"], m["reduction_kw"]) == (190.0, 185.0, 5.0)
    assert (m["peaks_before"], m["peaks_after"], m["new_peaks"], m["missed_peaks"]) == (1, 1, 1, 0)
    assert (m["adjustments"], m["unnecessary"], m["adjustments_per_day"]) == (1, 0, 1.0)


def test_metrics_missed_and_unnecessary():
    power = A(190, 100, 185)
    m = sim.metrics(power, power, B(0, 1, 0), 2, CAP, 0.0, 0.0)
    assert (m["missed_peaks"], m["unnecessary"], m["adjustments_per_day"]) == (2, 1, 0.5)


def test_billing_kw_uses_earlier_winter_and_summer_months():
    assert sim.billing_kw({1: 222.0, 2: 198.0, 3: 230.0, 7: 210.0}, 7) == 222.0   # 3월은 대상 아님
    assert sim.billing_kw({7: 200.0, 8: 190.0}, 8) == 200.0
    assert sim.billing_kw({3: 222.0, 4: 199.0}, 4) == 199.0


def test_savings_month_basis_and_12m_basis():
    s = sim.savings(max_after=200.0, month=8, monthly_max={1: 222.0, 7: 222.0, 8: 218.0},
                    other_days_max=0.0, rate=1000.0)
    assert (s["saving_kw_month"], s["saving_won_month"]) == (18.0, 18000.0)
    assert abs(s["saving_rate_month"] - 18.0 / 218.0) < 1e-12
    assert (s["billing_kw_before"], s["billing_kw_after"], s["saving_won_12m"]) == (222.0, 222.0, 0.0)


def test_savings_unsimulated_day_limits_month_max():
    s = sim.savings(max_after=200.0, month=7, monthly_max={7: 222.0}, other_days_max=212.0, rate=1000.0)
    assert (s["month_max_after"], s["saving_kw_month"]) == (212.0, 10.0)


def _hours(start, n):
    return pd.date_range(start, periods=n, freq="h")


def test_require_missing_file():
    try:
        sim.require("model", "__no_such_file__.json", "09_peak_alert.py")
    except SystemExit as e:
        assert "__no_such_file__.json" in str(e) and "09_peak_alert.py" in str(e)
    else:
        raise AssertionError("SystemExit이 나야 함")


def test_build_frame_common_full_days():
    idx = _hours("2021-07-01", 72)                       # 3일
    m3 = pd.DataFrame({"actual_power": 100.0, "predicted_power": 90.0, "planned_on": 1}, index=idx)
    rnn = pd.Series(0, index=idx[:60])                   # 3일째는 12시간만 있음
    rf = pd.Series(0, index=idx)
    hist = pd.DataFrame({"lag_168": 100.0, "same_hour_2w_max": [80.0] * 36 + [120.0] * 36}, index=idx)
    f = sim.build_frame(m3, rnn, rf, hist)
    assert len(f) == 48 and f.index.normalize().nunique() == 2
    assert list(f.columns) == ["power", "m3_forecast", "planned_on", "rnn_alert", "rf_alert", "lag_168", "expected"]
    assert f.index.is_monotonic_increasing and not f.isna().any().any()
    assert f["expected"].iloc[0] == 90.0 and f["expected"].iloc[-1] == 120.0     # max(M3 예측, 2주 최대)


def _frame():
    idx = _hours("2021-07-01", 24)
    return pd.DataFrame({"power": [100.0] * 22 + [185.0, 170.0], "m3_forecast": [100.0] * 22 + [165.0, 175.0],
                         "planned_on": 1, "rnn_alert": [0] * 22 + [1, 1], "rf_alert": [0] * 22 + [1, 0],
                         "lag_168": [100.0] * 22 + [180.0, 100.0], "expected": [110.0] * 22 + [165.0, 175.0]}, index=idx)


def test_alert_sources_names_directions_and_rules():
    f = _frame()
    s = sim.alert_sources(f, {"주의": "RF 단독", "확정": "AND"}, 170, CAP)
    assert list(s) == ["완벽 예측 (1시간 전)", "완벽 예측 (하루 전)", "M2 주의 (RF 단독)", "M2 확정 (AND)",
                       "참고: RNN 단독", "M3 하루 전", "1주 전 같은 시각"]
    assert [v[1] for v in s.values()] == ["after", "any", "after", "after", "after", "any", "any"]
    for name, (_, _, exp) in s.items():          # 완벽 예측만 받는 쪽도 실제 전력을 안다
        assert (exp == f["power" if name.startswith("완벽 예측") else "expected"].to_numpy()).all()
    tail = lambda name: s[name][0][-2:].tolist()
    assert tail("완벽 예측 (1시간 전)") == [True, False] and tail("M2 주의 (RF 단독)") == [True, False]
    assert tail("M2 확정 (AND)") == [True, False] and tail("참고: RNN 단독") == [True, True]
    assert tail("M3 하루 전") == [False, True] and tail("1주 전 같은 시각") == [True, False]


def test_alert_sources_confirm_is_subset_of_caution():
    s = sim.alert_sources(_frame(), {"주의": "RF 단독", "확정": "RNN 단독"}, 170, CAP)
    assert s["M2 확정 (RNN 단독)"][0][-2:].tolist() == [True, False]


def test_alert_sources_unknown_rule():
    try:
        sim.alert_sources(_frame(), {"주의": "XGB 단독", "확정": "AND"}, 170, CAP)
    except ValueError as e:
        assert "XGB 단독" in str(e)
    else:
        raise AssertionError("ValueError가 나야 함")


def test_select_m3_cutoff_maximises_f1():
    f = _frame()
    f["power"] = [100.0] * 20 + [185.0, 190.0, 185.0, 170.0]
    f["m3_forecast"] = [100.0] * 20 + [160.0, 172.0, 165.0, 155.0]
    assert sim.select_m3_cutoff(f, CAP) == 156          # 156~160이 F1 1.0, 같으면 가장 낮은 값


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"\n{len(tests)} passed")
