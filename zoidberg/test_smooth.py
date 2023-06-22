import numpy as np

from .smooth import find_outliers, smooth


def test_square():
    ### test smooth function on a square
    ### create square
    square_r = np.zeros(400)
    square_z = np.zeros(400)
    square_r[0:100] = np.linspace(0, 1, num=100)
    square_z[0:100] = np.linspace(0, 0, num=100)
    square_r[100:200] = np.linspace(1, 1, num=100)
    square_z[100:200] = np.linspace(0, 1, num=100)
    square_r[200:300] = np.linspace(1, 0, num=100)
    square_z[200:300] = np.linspace(1, 1, num=100)
    square_r[300:400] = np.linspace(0, 0, num=100)
    square_z[300:400] = np.linspace(1, 0, num=100)
    ### smooth square
    newr, newz = smooth(square_r, square_z, cutoff=0.75, bounds=10)
    outliers = find_outliers(newr, newz, 0.75)
    assert len(outliers) == 0


def run_npolygons(n=10, m=1000):
    ### test smooth function on a symmetric n-polygone
    ### create n-polygone
    circle = np.linspace(0, 2 * np.pi, num=n, endpoint=False)
    ends_r = np.cos(circle)
    ends_z = np.sin(circle)
    phi_r = np.linspace(0, 2 * np.pi, n, endpoint=False)
    phi_z = np.linspace(0, 2 * np.pi, m, endpoint=False)
    test_r = np.interp(phi_z, phi_r, ends_r, period=np.pi * 2)
    test_z = np.interp(phi_z, phi_r, ends_z, period=np.pi * 2)
    ### smooth n-polygone
    newr, newz = smooth(test_r, test_z, 0.96, 10)
    outliers = find_outliers(newr, newz, 0.96, 30)
    return outliers, test_r, test_z, newr, newz


def test_assert_npolygons(n=10, m=100):
    ### test the smooth function and evaluates the outcome
    ### finds nothing after smoothing
    outliers, testr, testz, newr, newz = run_npolygons(n=10, m=1000)
    assert not outliers and testz.all() != newz.all()
    ### no smoothing done
    outliers, testr, testz, newr, newz = run_npolygons(n=1000, m=10000)
    assert not outliers and testz.all() == newz.all()
    ### find outliers after smoothing
    outliers, testr, testz, newr, newz = run_npolygons(n=5, m=1000)
    assert outliers and testz.all() != newz.all()


if __name__ == "__main__":
    test_assert_npolygons(n=10, m=1000)
