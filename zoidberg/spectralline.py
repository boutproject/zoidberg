import warnings

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None
import numpy as np
from scipy.optimize import least_squares

from .rzline import RZline


def _fft(args, freq):
    out = args[0]
    for i0, (s, c) in enumerate(zip(args[1::2], args[2::2])):
        i = i0 + 1
        out += np.sin(i * freq) * s
        out += np.cos(i * freq) * c
    return out


def _err(args, data, freq):
    freq = np.linspace(-np.pi, np.pi, len(data), endpoint=False)
    return _fft(args, freq) - data


class SpectralLine:
    """
    similar to RZline, but based on fourier harmonics
    """

    def __init__(self, line, num):
        """
        Create an SpectralLine from a `line` using `num` modes.

        line: RZline
        num: integer
        """
        freq = line.distance()
        freq = freq[:-1] * np.pi * 2 / freq[-1]
        self.resR = least_squares(_err, np.zeros(num), args=(line.R, freq))
        self.resZ = least_squares(_err, np.zeros(num), args=(line.Z, freq))

        self.cost = self.resR.cost + self.resZ.cost
        self.orgLine = line

    def Rvalue(self, theta):
        return _fft(self.resR.x, theta)

    def Zvalue(self, theta):
        return _fft(self.resZ.x, theta)

    def equallySpaced(self, num=None):
        freq = np.linspace(0, np.pi * 2, num or len(self.orgLine.R))
        return RZline(self.Rvalue(freq), self.Zvalue(freq))

    def plot(self, axis=None, show=True):
        """Plot the RZline, either on the given axis or a new figure

        Parameters
        ----------
        axis : matplotlib axis, optional
            A matplotlib axis to plot on. By default a new figure
            is created
        show : bool, optional
            Calls plt.show() at the end

        Returns
        -------
        axis
            The matplotlib axis that was used

        """
        if plt is None:
            warnings.warn("matplotlib not available, unable to plot")
            return None

        if axis is None:
            fig = plt.figure()
            axis = fig.add_subplot(1, 1, 1)

        theta = np.linspace(0, 2 * np.pi, 100, endpoint=True)
        axis.plot(self.Rvalue(theta), self.Zvalue(theta), "k-")
        axis.plot(self.orgLine.R, self.orgLine.Z, "ro")

        if show:
            plt.show()

        return axis
