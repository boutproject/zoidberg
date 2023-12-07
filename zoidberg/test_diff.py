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
    print(l2s.shape)
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


if __name__ == "__main__":
    test_diff_c4_np()
    test_diff_c4_per()
