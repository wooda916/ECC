# computation
import xarray as xr
import numpy as np

# plotting
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import matplotlib.patheffects as pe


def plot_check(data, min, max):
    ### Rough plot
    proj = ccrs.PlateCarree(central_longitude=0)
    plt.figure(figsize=(20, 10))
    ax = plt.subplot(111, projection=proj)

    data.plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap='RdBu_r', vmin=min, vmax=max)
    # Add coastlines
    ax.coastlines()
    plt.show()