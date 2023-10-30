import os
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
from itertools import product
import glob
import logging
import argparse
import datetime
from time import sleep
from multiprocessing import SimpleQueue
from multiprocessing.pool import Pool

def load_data(metrics_paths, ipcc_regions_path, lat_lon_gdf_path, population_data_path):
    """
    Load the necessary data files and return them.
    Parameters:
        metrics_paths (list): Paths to the metrics dictionary files.
        ipcc_regions_path (str): Path to the IPCC regions GeoJSON file.
        lat_lon_gdf_path (str): Path to the latitude-longitude GeoDataFrame saved as GeoJSON.
        :param population_data_path: filepath to population data
    Returns:
        metrics_dicts (list): List of metrics dictionaries.
        ipcc_regions (GeoDataFrame): IPCC regions as a GeoDataFrame.
        land_based_gdf (GeoDataFrame):  Latitude-longitude grid as a GeoDataFrame, with regions assigned,
                                        filtered for land-only cells.
        population_array (np.array): array of populations for each region
    """
    # Load metrics dictionaries
    metrics_dicts = {path.split('f1_')[1].split('_p')[0].split('_ab')[0]:
                     np.load(path, allow_pickle=True).item() for path in metrics_paths}

    # Load IPCC regions
    ipcc_regions = gpd.read_file(ipcc_regions_path)

    # Load latitude-longitude grid
    lat_lon_gdf = gpd.read_file(lat_lon_gdf_path)

    # Load population data
    population_var_name = 'Population Count, v4.11 (2000, 2005, 2010, 2015, 2020): 1 degree'
    population_data_xr = xr.open_dataset(population_data_path)
    population_array = np.flipud(population_data_xr[population_var_name][0, :, :].values) # Flip the array vertically

    # Add population data to lat_lon_gdf
    lat_lon_gdf['population'] = population_array.flatten()

    def assign_regions_to_grid(lat_lon_gdf, ipcc_regions, land_based = True):
        for index, region in ipcc_regions.iterrows():
            mask = lat_lon_gdf.within(region.geometry)
            lat_lon_gdf.loc[mask, 'Region'] = region['Name']
            lat_lon_gdf.loc[mask, 'Acronym'] = region['Acronym']
            if land_based == True:
                land_based_gdf = lat_lon_gdf.dropna(subset=['population']).copy()
        return land_based_gdf

    land_based_gdf = assign_regions_to_grid(lat_lon_gdf, ipcc_regions)

    return metrics_dicts, ipcc_regions, land_based_gdf, population_array

def select_scenario_array(scenario_percentiles, metrics_dicts):
    """
    Select the arrays corresponding to the specified 'elbow' percentiles for each metric.

    Parameters: metrics_dict (list): List of metrics dictionaries. scenario_percentiles (list): List of core scenario
    percentiles for each metric (this will be the 'elbows' normally).

    Returns:
        selected_arrays (dict): Dictionary containing selected arrays for each metric based on elbow percentiles.
    """
    selected_arrays = {}
    for i, metric in enumerate(metrics_dicts):
        percentile = scenario_percentiles[i]
        selected_arrays[metric] = metrics_dicts[metric].get(percentile, None)
    return selected_arrays

def calculate_cell_failure_ratio(selected_arrays):
    """
    Calculate the failure ratio for each cell based on the selected arrays.

    Parameters: selected_arrays (dict): Dictionary containing selected arrays for each metric based on scenario
    (elbow) percentiles.

    Returns:
        failure_ratios (numpy.ndarray): Array containing the failure ratio for each cell.
    """
    # Stack the selected arrays along a new axis to form a 3D array
    stacked_arrays = np.stack(list(selected_arrays.values()), axis=-1)

    # Calculate the failure ratio for each cell (sum along the new axis, divided by number of metrics)
    failure_ratios = np.sum(stacked_arrays, axis=-1) / len(selected_arrays)

    return failure_ratios

def calculate_weighted_regional_average(failure_ratios, land_based_gdf, population_array):
    """
    Calculate the weighted average for each IPCC region based on the failure ratios.

    Parameters:
        failure_ratios (numpy.ndarray): Array containing the failure ratio for each cell.
        land_based_gdf (GeoDataFrame): Latitude-longitude grid with assigned IPCC regions and population weights.
        ```````````````````````````````With NaNs removed to show only land-based cells

    Returns:
        regional_weighted_average.index.to_numpy() (np.array): Array containing the IPCC region acronyms.
        regional_weighted_average.to_numpy() (np.array):  Array of regional weighted averages (corrresponding to region acronyms)
        regional_std.to_numpy() (np.array):  Array of regional weighted standard deviations (corrresponding to region acronyms)
    """
    # Filter failure_ratios to only include corresponding land-based cells
    failure_ratios_filtered = failure_ratios.flatten()[land_based_gdf.index]

    # filter population array (should be the same format as failure_ratios_filtered anyway)
    population_array_filtered = population_array.flatten()[land_based_gdf.index]

    # failure ratios weighted by population
    failure_ratios_filtered_weighted = failure_ratios_filtered * population_array_filtered

    # Add failure_ratios to land_based_gdf
    land_based_gdf['failure_ratios'] = failure_ratios_filtered

    # Add weighted_failure_ratios to land_based_gdf
    land_based_gdf['weighted_failure_ratios'] = failure_ratios_filtered_weighted

    # Group by region
    regional_grouped = land_based_gdf.groupby('Acronym')

    # Calculate weighted average of failure ratios, normalised by total population in the region
    regional_sum = regional_grouped['weighted_failure_ratios'].sum()  # sum of the population weighted failure ratios in region
    regional_population = regional_grouped['population'].sum() # sum of the population in the region
    regional_weighted_average = regional_sum / regional_population # weighted average in region

    # Calculate weighted standard deviation of failure ratios for each region
    def weighted_std(group):
        valid_data = group.dropna(subset=['population', 'failure_ratios'])
        non_zero_weights = valid_data['population'] > 0

        valid_weights = valid_data['population'][non_zero_weights]
        valid_failure_ratios = valid_data['failure_ratios'][non_zero_weights]

        # Check if the sum of valid_weights is zero to prevent division by zero
        if np.sum(valid_weights) == 0:
            return np.nan  # Return NaN or some other appropriate value

        mu_w = np.sum(valid_failure_ratios * valid_weights) / np.sum(valid_weights)
        diff = valid_failure_ratios - mu_w
        weighted_diff_sq = (diff ** 2) * valid_weights
        M = len(valid_weights)
        if M > 1:
            return np.sqrt(np.sum(weighted_diff_sq) / (((M - 1) / M) * np.sum(valid_weights)))
        else:
            return np.nan

    regional_std = regional_grouped.apply(weighted_std)

    return regional_weighted_average.index.to_numpy(), regional_weighted_average.to_numpy(), regional_std.to_numpy()

# initialize worker processes
def init_worker(metrics_dicts, ipcc_regions, land_based_gdf, all_combinations, bundle, population_array,
                root, output_dir):
    # declare scope of a new global variable
    global shared_metrics_dicts
    global shared_ipcc_regions
    global shared_land_based_gdf
    global shared_population_array
    global shared_all_combinations
    global shared_bundle
    global shared_root
    global shared_output_dir
    # store argument in the global variable for this process
    shared_metrics_dicts = metrics_dicts
    shared_ipcc_regions = ipcc_regions
    shared_land_based_gdf = land_based_gdf
    shared_population_array = population_array
    shared_all_combinations = all_combinations
    shared_bundle = bundle
    shared_root = root
    shared_output_dir = output_dir

def calculate_results_for_thresholds(id):
    # Select the elbow arrays based on the current combination of percentiles
    regional_means = {}
    regional_stds = {}
    for combo in shared_all_combinations[(id*shared_bundle)-shared_bundle:(id*shared_bundle)]:
        selected_scenario_arrays = select_scenario_array(scenario_percentiles=combo,
                                                         metrics_dicts=shared_metrics_dicts)

        # Calculate the failure ratios
        failure_ratios = calculate_cell_failure_ratio(selected_arrays=selected_scenario_arrays)

        # Calculate the regional statistics
        acronyms, means, stds = calculate_weighted_regional_average(failure_ratios=failure_ratios,
                                                             land_based_gdf=shared_land_based_gdf,
                                                             population_array=shared_population_array)

        regional_means[combo] = {acronyms[i]: mean for i, mean in enumerate(means)}
        regional_stds[combo] = {acronyms[i]: std for i, std in enumerate(stds)}

    np.save(f'{shared_output_dir}pc_regional_mean_population_weighted_partial_results_{(id*shared_bundle)-shared_bundle}_to_{(id*shared_bundle)-1}.npy', regional_means)
    np.save(f'{shared_output_dir}pc_regional_std_population_weighted_partial_results_{(id*shared_bundle)-shared_bundle}_to_{(id*shared_bundle)-1}.npy', regional_stds)

def arguments():
    parser = argparse.ArgumentParser(description="Parallel processing of scenario combinations")

    # add arguments
    parser.add_argument("-r", "--root", default='/users/ci1twx/DATA/', help="Path to the root directory")
    parser.add_argument("-o", "--output_dir", default='/users/ci1twx/OUTPUT/', help="Path to the output directory")
    parser.add_argument("-min", "--minimum", default=5, help="minimum percentile to iterate from")
    parser.add_argument("-max", "--maximum", default=96, help="maximum percentile to iterate to")
    parser.add_argument("-step", "--step", default=10, help="pecentile steps to iterate across")
    parser.add_argument("-c", "--cores", default=1, help="number of cores to use")
    parser.add_argument("-chunks", "--chunks", default=100, help="size of chunks")

    args = parser.parse_args()

    return(args)

def split_into_chunks(input_list, chunk_size):
    """
    Splits the input list into chunks of the specified size.

    :param input_list: The list to be split.
    :param chunk_size: The size of each chunk.
    :return: A list of lists representing the chunks.
    """
    # Calculate the number of chunks
    num_chunks = (len(input_list) + int(chunk_size) - 1) // int(chunk_size)

    # Split the list into chunks
    chunks = [input_list[i * int(chunk_size):(i + 1) * int(chunk_size)] for i in range(num_chunks)]

    return chunks

if __name__ == '__main__':
    args = arguments()

    root = args.root
    output_dir = args.output_dir
    print(output_dir)

    metrics_paths = glob.glob(f'{root}binary_cell_failure/*')
    ipcc_regions_path = f'{root}IPCC-WGI-reference-regions-v4.geojson'
    lat_lon_gdf_path = f'{root}lat_lon_gdf.geojson'
    population_data_path = f'{root}gpw_v4_population_count_rev11_1_deg.nc'

    metrics_dicts, ipcc_regions, land_based_gdf, population_array = load_data(metrics_paths,
                                                                              ipcc_regions_path,
                                                                              lat_lon_gdf_path,
                                                                              population_data_path)

    percentiles_range = list(range(int(args.minimum), int(args.maximum), int(args.step)))

    # Generate all combinations of 7 metrics
    all_combinations = list(product(percentiles_range, repeat=7))

    # Initialize the Process Pool Executor
    #cores = args.cores
    #num_workers = min(cores, os.cpu_count() - 1)
    num_workers = int(args.cores)

    # manual chunking
    chunk_size = int(args.chunks)  # Number of iterations to save after
    chunks = split_into_chunks(all_combinations, chunk_size)
    number_of_chunks = len(chunks)
    ids = range(1, number_of_chunks+1)

    def already_exists(ids, chunk_size):
        """
        Filters out ids for which the associated csv file already exists.

        :param ids: List of ids.
        :param shared_output_dir: Directory where the csv files are saved.
        :param shared_bundle: Shared bundle value.
        :return: List of ids for which the associated csv file does not exist.
        """
        return [id for id in ids
                if not os.path.exists(f'{output_dir}pc_regional_mean_population_weighted_partial_results_{(id * chunk_size) - chunk_size}_to_{(id * chunk_size) - 1}.npy')
                ]

    filtered_ids = already_exists(ids, chunk_size)

    # create and configure the process pool
    with Pool(processes=num_workers, initializer=init_worker, initargs=(metrics_dicts,
                                                                        ipcc_regions,
                                                                        land_based_gdf,
                                                                        all_combinations,
                                                                        chunk_size,
                                                                        population_array,
                                                                        root,
                                                                        output_dir)) as pool:
        # issue tasks into the process pool
        #pool.map(hadi_calculate_results_for_thresholds, range(100))
        pool.map(calculate_results_for_thresholds, filtered_ids)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f'END TIME: {current_time}')