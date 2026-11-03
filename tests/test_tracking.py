import numpy as np

from ballistic_forecast.scene import Scene
from ballistic_forecast.tracking.detector import detect
from ballistic_forecast.tracking.tracker import MultiTargetTracker


def _scene(truth, occluders=()):
    return Scene(truth, np.zeros(len(truth), int), 400, 300, list(occluders))


def _run(scene):
    """Return, per frame, {track_id: position} for confirmed tracks."""
    tracker = MultiTargetTracker()
    out = []
    for f in range(scene.n_frames):
        tracker.update(detect(scene.render_sensor(f)))
        out.append({t.id: t.position.copy() for t in tracker.confirmed()})
    return out


def test_detector_subpixel_centroid():
    truth = np.array([[[100.3, 50.7]]])
    det = detect(_scene(truth).render_sensor(0))
    assert det.shape == (1, 2)
    np.testing.assert_allclose(det[0], truth[0, 0], atol=0.3)


def test_identities_survive_a_crossing():
    t = np.arange(60, dtype=float)
    a = np.stack([20 + 6 * t, 150 + 0 * t], -1)  # left -> right
    b = np.stack([380 - 6 * t, 150 + 0.02 * (t - 30) ** 2], -1)  # right -> left, crossing at t=30
    frames = _run(_scene(np.stack([a, b])))

    def owner(f, truth):
        ids = frames[f]
        return min(ids, key=lambda i: np.linalg.norm(ids[i] - truth[f]))

    assert owner(10, a) == owner(55, a)
    assert owner(10, b) == owner(55, b)
    assert owner(10, a) != owner(10, b)


def test_track_coasts_through_occlusion():
    t = np.arange(70, dtype=float)
    traj = np.stack([20 + 5 * t, 250 - 6 * t + 0.1 * t**2], -1)[None]
    frames = _run(_scene(traj, occluders=[(150, 0, 200, 300)]))  # hidden for ~10 frames
    before = set(frames[20])
    after = set(frames[60])
    assert len(before) == 1 and before == after
