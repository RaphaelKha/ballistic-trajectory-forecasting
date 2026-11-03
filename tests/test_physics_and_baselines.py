import numpy as np

from ballistic_forecast.data import WindowConfig, make_windows
from ballistic_forecast.metrics import ade, fde, summarize
from ballistic_forecast.models.baselines import constant_velocity, parabola_fit
from ballistic_forecast.physics import PhysicsConfig, ShotParams, simulate


def _params(n, g=0.5, k=0.0, w=0.0):
    return ShotParams(np.full(n, g), np.full(n, k), np.full(n, w))


def test_vacuum_matches_closed_form():
    p0, v0, g, steps = np.array([[100.0, 900.0]]), np.array([[10.0, -20.0]]), 0.5, 50
    traj = simulate(p0, v0, _params(1, g), steps)[0]
    t = np.arange(1, steps + 1)
    expected_y = p0[0, 1] + t * v0[0, 1] + g * t * (t + 1) / 2  # semi-implicit Euler
    np.testing.assert_allclose(traj[:, 0], p0[0, 0] + t * v0[0, 0])
    np.testing.assert_allclose(traj[:, 1], expected_y)


def test_drag_shortens_range():
    p0, v0 = np.array([[0.0, 0.0]]), np.array([[15.0, -20.0]])
    vac = simulate(p0, v0, _params(1), 80)[0]
    drag = simulate(p0, v0, _params(1, k=0.001), 80)[0]
    assert drag[-1, 0] < vac[-1, 0]


def test_wind_pushes_downwind():
    p0, v0 = np.array([[0.0, 0.0]]), np.array([[0.0, -20.0]])
    traj = simulate(p0, v0, _params(1, k=0.001, w=10.0), 60)[0]
    assert traj[-1, 0] > 0


def test_parabola_is_exact_in_noise_free_vacuum():
    rng = np.random.default_rng(0)
    w = make_windows(200, "vacuum", WindowConfig(noise_px=0.0), PhysicsConfig(), rng)
    pred = parabola_fit(w["obs_clean"], 30)
    assert fde(pred, w["future"]).max() < 1e-6


def test_parabola_fails_under_drag():
    rng = np.random.default_rng(0)
    w = make_windows(500, "drag", WindowConfig(noise_px=0.0), PhysicsConfig(), rng)
    assert np.median(fde(parabola_fit(w["obs_clean"], 30), w["future"])) > 20


def test_constant_velocity_on_straight_line():
    obs = np.stack([np.arange(30.0), 2 * np.arange(30.0)], -1)[None]
    pred = constant_velocity(obs, 5)
    np.testing.assert_allclose(pred[0, -1], [34.0, 68.0])


def test_metrics():
    target = np.zeros((2, 3, 2))
    pred = target.copy()
    pred[0, :, 0] = [0.0, 0.0, 3.0]
    np.testing.assert_allclose(fde(pred, target), [3.0, 0.0])
    np.testing.assert_allclose(ade(pred, target), [1.0, 0.0])
    assert summarize(pred, target, hit_radius=2.0)["hit@2px"] == 0.5
