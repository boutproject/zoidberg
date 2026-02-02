import numpy as np

from . import diff


def l2(f):
    return np.sqrt(np.mean(f**2))


def mms(testfunc, inp, ana, xmax=2 * np.pi, slices=None):
    if slices is None:
        slices = [slice(None)]
    l2s = []
    for nx in [64, 128]:
        x = np.linspace(0, xmax, nx, endpoint=False)
        err = testfunc(inp(x)) * nx / xmax - ana(x)
        l2s.append([l2(err[s]) for s in slices])
        if xmax == 2:
            import matplotlib.pyplot as plt

            plt.plot(x, testfunc(inp(x)) * nx / xmax, label="result")
            plt.plot(x, ana(x), label="ana")
            plt.legend()
            plt.show()
    return l2s


def check_l2s(l2s, exps):
    if isinstance(exps, int):
        exps = [exps]
    l2s = np.array(l2s).T
    success = True
    for (a, b), exp in zip(l2s, exps):
        ord = np.log(a / b) / np.log(2)
        s = np.isclose(ord, exp, atol=0.25)
        if not s:
            success = False
        s = "👍" if s else "❌"
        print(f"{ord:.3f} vs {exp} {s}")
    return success


def test_diff_c2_per():
    print("C2 periodic")
    l2 = mms(lambda x: diff.c2(x, 0, True), np.sin, np.cos)
    assert check_l2s(l2, 2)


def test_diff_c2_np():
    print("C2 periodic")
    l2 = mms(
        lambda x: diff.c2(x, 0, False),
        np.sin,
        np.cos,
        1,
        slices=[slice(0, 1), slice(1, -1), slice(-1)],
    )
    assert check_l2s(l2, [2, 2, 2])


def test_diff_c4_per():
    print("C4 periodic")
    l2 = mms(lambda x: diff.c4(x, 0, True), np.sin, np.cos)
    assert check_l2s(l2, 4)


def test_diff_c4_np():
    print("C4 non periodic")
    l2 = mms(
        lambda x: diff.c4(x, 0, False),
        np.sin,
        np.cos,
        1,
        slices=[slice(0, 2), slice(2, -2), slice(-2)],
    )
    assert check_l2s(l2, [4, 4, 4])

    l2 = mms(
        lambda x: diff.c4(x, 0, False),
        np.cos,
        lambda x: -np.sin(x),
        1,
    )
    assert check_l2s(l2, 4)


def test_diff_get_dist():
    for refine in 1, 10, 100:
        for nx in 16, 32, 64, 128:
            y = np.linspace(0, 1, nx + 1)
            x = y * 0 + 2
            z = y  # **3

            RZ = np.empty((nx + 1, 1, 2))
            RZ[:, 0, 0] = np.sqrt(x**2 + y**2)
            RZ[:, 0, 1] = z
            phi = np.atan2(y, x)

            # https://www.wolframalpha.com/input?i=integrate+sqrt%281%2B9*x**4%29+dx+from+0+to+1
            # result = 1.54786565468361014477533164606426061800663331794707225136859391655021114123260072669317296327301901
            result = np.sqrt(2)

            approx = diff.get_dist(RZ, phi, refine=refine)
            print(approx, result, np.abs(approx - result))
            tol = 1e-15 if refine == 1 else 1e-11
            assert np.abs(approx[0] - result) < tol

    for refine in 1, 10:
        l2 = []
        for nx in 16, 32, 64, 128:
            y = np.linspace(0, 1, nx + 1)
            x = y * 0 + 2
            z = y**3

            RZ = np.empty((nx + 1, 1, 2))
            RZ[:, 0, 0] = np.sqrt(x**2 + y**2)
            RZ[:, 0, 1] = z
            phi = np.atan2(y, x)

            # https://www.wolframalpha.com/input?i=integrate+sqrt%281%2B9*x**4%29+dx+from+0+to+1
            result = 1.54786565468361014477533164606426061800663331794707225136859391655021114123260072669317296327301901

            approx = diff.get_dist(RZ, phi, refine=refine)
            # print(approx, result, np.abs(approx - result))
            # tol = 1e-15 if refine == 1 else 1e-11
            # assert np.abs(approx[0] - result) < tol
            l2.append(np.abs(approx - result))
        for i in range(3):
            assert check_l2s(l2[i : i + 2], 2)
    for nx in 16, 32, 64, 128:
        l2 = []
        for refine in 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024:
            y = np.linspace(0, 1, nx + 1)
            x = y * 0 + 2
            z = y**3

            RZ = np.empty((nx + 1, 1, 2))
            RZ[:, 0, 0] = np.sqrt(x**2 + y**2)
            RZ[:, 0, 1] = z
            phi = np.atan2(y, x)

            # https://www.wolframalpha.com/input?i=integrate+sqrt%281%2B9*x**4%29+dx+from+0+to+1
            result = 1.54786565468361014477533164606426061800663331794707225136859391655021114123260072669317296327301901

            approx = diff.get_dist(RZ, phi, refine=refine)
            l2.append(np.abs(approx - result))
        for i in range(len(l2) - 1):
            # At a certain point further refinement is not helpful.
            chk = check_l2s(l2[i : i + 2], 2)
            if np.log2(nx) - i > 0.5:
                assert chk
            else:
                break


if __name__ == "__main__":
    test_diff_get_dist()
    test_diff_c4_np()
    test_diff_c4_per()
