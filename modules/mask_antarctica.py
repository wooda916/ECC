import numpy as np
import xarray as xr


def mask_antarctic_values(data_array):
    """
    Masks values in the Antarctic region of an xarray DataArray.
    
    Parameters:
        data_array (xr.DataArray): An xarray DataArray with dimensions "latitude" and "longitude".
        
    Returns:
        xr.DataArray: A new DataArray with values south of the Antarctic Circle masked as NaNs.
    """
    
    # Check if the required dimensions exist in the DataArray
    if 'lat' not in data_array.dims or 'lon' not in data_array.dims:
        raise ValueError("The DataArray must have dimensions named 'lat' and 'lon'.")
    
    antarctic_mask = data_array.lat < -66.5
    
    return data_array.where(~antarctic_mask, np.nan)


def land_cells_excluding_antarctic(dataset_path):
    """
    Returns the count of land cells excluding the cells in the Antarctic region.
    
    Parameters:
        dataset_path (str): Path to the xarray Dataset containing the 'mask' variable.
        
    Returns:
        int: Count of land cells excluding the Antarctic region cells.
    """
    
    # Load the land_mask from the provided dataset path
    land_mask = xr.open_dataset(dataset_path)['mask']
    
    # Count the total number of land cells
    total_land_cells = land_mask.count().item()
    
    # Count the number of Antarctic land cells
    antarctic_land_cells = (land_mask.where(land_mask.lat < -66.5)).count().item()
    
    # Subtract the Antarctic land cells from the total
    return total_land_cells - antarctic_land_cells