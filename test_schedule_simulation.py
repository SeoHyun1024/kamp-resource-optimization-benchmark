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


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"ok  {name}")
    print(f"\n{len(tests)} passed")
