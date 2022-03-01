import numpy as np
import pytest

from . import field, fieldtracer


def test_slab():
    mag_field = field.Slab(By=0.5, Bz=0.1, Bzprime=0.0)

    tracer = fieldtracer.FieldTracer(mag_field)

    coords = tracer.follow_field_lines(0.2, 0.3, [1.5, 2.5])

    assert coords.shape == (2, 1, 2)
    assert np.allclose(coords[:, 0, 0], 0.2)  # X coordinate

    assert np.allclose(coords[0, 0, 1], 0.3)
    assert np.allclose(coords[1, 0, 1], 0.3 + 0.1 / 0.5)  # Z coordinate

    coords = tracer.follow_field_lines([0.2, 0.3, 0.4], 0.5, [1.0, 2.5])

    assert coords.shape == (2, 3, 2)
    assert np.allclose(coords[:, 0, 0], 0.2)
    assert np.allclose(coords[:, 1, 0], 0.3)
    assert np.allclose(coords[:, 2, 0], 0.4)
    assert np.allclose(coords[1, :, 1], 0.5 + 1.5 * 0.1 / 0.5)  # Z coord


def test_FieldTracerReversible_slab():
    mag_field = field.Slab(By=0.5, Bz=0.1, Bzprime=0.0)

    tracer = fieldtracer.FieldTracerReversible(mag_field)

    coords = tracer.follow_field_lines(0.2, 0.3, [1.5, 2.5])

    assert coords.shape == (2, 1, 2)
    assert np.allclose(coords[:, 0, 0], 0.2)  # X coordinate

    assert np.allclose(coords[0, 0, 1], 0.3)
    assert np.allclose(coords[1, 0, 1], 0.3 + 0.1 / 0.5)  # Z coordinate

    coords = tracer.follow_field_lines([0.2, 0.3, 0.4], 0.5, [1.0, 2.5])

    assert coords.shape == (2, 3, 2)
    assert np.allclose(coords[:, 0, 0], 0.2)
    assert np.allclose(coords[:, 1, 0], 0.3)
    assert np.allclose(coords[:, 2, 0], 0.4)
    assert np.allclose(coords[1, :, 1], 0.5 + 1.5 * 0.1 / 0.5)  # Z coord


def test_poincare():
    mag_field = field.Slab(By=0.5, Bz=0.1, Bzprime=0.0)

    result, y_slices = fieldtracer.trace_poincare(
        mag_field, 0.0, 0.0, 1.0, nplot=3, revs=5, nover=1
    )

    assert y_slices.size == 3
    assert result.shape == (5, y_slices.size, 1, 2)


def test_traceweb():
    try:
        import osa
    except ImportError:
        pytest.skip("osa not installed")
    try:
        web = fieldtracer.FieldTracerWeb(configId=0)
    except:
        pytest.skip("Failed to initiallise - service not available?")

    num = 10
    start = np.linspace(5.7, 6, num), np.zeros(num)

    res = web.follow_field_lines(*start, np.linspace(0, 0.1, 10))
    assert np.all(res[1:, :, 1] < 0)
    dist = np.sqrt(np.sum((res[:, 1:, :] - res[:, :-1, :]) ** 2, axis=2))
    assert np.all(dist >= 0)
    assert np.max(dist) < 0.05

    # Check reverse direction
    res = web.follow_field_lines(*start, -np.linspace(0, 0.1, 10))
    assert np.all(res[1:, :, 1] > 0)
    dist = np.sqrt(np.sum((res[:, 1:, :] - res[:, :-1, :]) ** 2, axis=2))
    assert np.all(dist >= 0)
    # make sure step isn't to large
    assert np.max(dist) < 0.05

    res = web.follow_field_lines(*start, np.linspace(0, 0.1, 10) + np.pi * 4 / 5)
    # import matplotlib.pyplot as plt
    # plt.plot(res[..., 0], res[...,1])
    # plt.show()
    assert np.all(res[1:, :, 1] < 0)
    dist = np.sqrt(np.sum((res[:, 1:, :] - res[:, :-1, :]) ** 2, axis=2))
    assert np.all(dist >= 0)
    assert np.max(dist) < 0.05

    res = web.follow_field_lines(*start, -np.linspace(0, 0.1, 10) + np.pi * 6 / 5)
    assert np.all(res[1:, :, 1] > 0)
    dist = np.sqrt(np.sum((res[:, 1:, :] - res[:, :-1, :]) ** 2, axis=2))
    assert np.all(dist >= 0)
    assert np.max(dist) < 0.05


if __name__ == "__main__":
    test_traceweb()
