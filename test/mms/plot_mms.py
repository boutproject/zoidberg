import matplotlib.pyplot as plt
import xarray as xr
import sys


def plot(ds, var, i):
    print(ds[var].min().data, ds[var].max().data)
    plt.pcolormesh(ds.R[:, i], ds.Z[:, i], ds[var][:, i])
    plt.colorbar()
    plt.show()


with xr.open_dataset(sys.argv[1]) as ds:
    plot(ds, sys.argv[2], int(sys.argv[3]))
