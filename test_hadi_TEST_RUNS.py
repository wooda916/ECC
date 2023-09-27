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

    Returns:
        metrics_dicts (list): List of metrics dictionaries.
        ipcc_regions (GeoDataFrame): IPCC regions as a GeoDataFrame.
        lat_lon_gdf (GeoDataFrame): Latitude-longitude grid as a GeoDataFrame.
        :param population_data_path: filepath to population data
    """
    # Load metrics dictionaries
    # metrics_dicts = [np.load(path, allow_pickle=True).item() for path in metrics_paths]
    metrics_dicts = {path.split('f1_')[1].split('_p')[0]: np.load(path, allow_pickle=True).item() for path in
                    metrics_paths}
    #metrics_dicts = {path.split('f1_')[1].split('_p')[0]: {k:np.random.random((180, 360)) for (k, v) in np.load(path, allow_pickle=True).item().items()} for path in
    #                 metrics_paths}

    # Load IPCC regions
    ipcc_regions = gpd.read_file(ipcc_regions_path)

    # Load latitude-longitude grid
    lat_lon_gdf = gpd.read_file(lat_lon_gdf_path)

    # Load population data
    population_var_name = 'Population Count, v4.11 (2000, 2005, 2010, 2015, 2020): 1 degree'
    population_data_xr = xr.open_dataset(population_data_path)
    population_array = population_data_xr[population_var_name][0, :, :].values
    population_array = np.flipud(population_array)  # Flip the array vertically

    # Add population data to lat_lon_gdf
    lat_lon_gdf['population'] = population_array.flatten()

    return metrics_dicts, ipcc_regions, lat_lon_gdf

def assign_regions_to_grid(lat_lon_gdf, ipcc_regions):
    for index, region in ipcc_regions.iterrows():
        mask = lat_lon_gdf.within(region.geometry)
        lat_lon_gdf.loc[mask, 'Region'] = region['Name']
        lat_lon_gdf.loc[mask, 'Acronym'] = region['Acronym']
    return lat_lon_gdf

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
    # failure_ratios = np.nansum(stacked_arrays, axis=-1) / len(selected_arrays)
    failure_ratios = np.sum(stacked_arrays, axis=-1) / len(selected_arrays)

    return failure_ratios

def calculate_weighted_regional_average(failure_ratios, lat_lon_gdf, scenario_percentiles):
    """
    Calculate the weighted average for each IPCC region based on the failure ratios.

    Parameters:
        failure_ratios (numpy.ndarray): Array containing the failure ratio for each cell.
        lat_lon_gdf (GeoDataFrame): Latitude-longitude grid with assigned IPCC regions and population weights.

    Returns:
        regional_average_df (DataFrame): DataFrame containing the weighted average for each IPCC region.
    """
    # Flatten the failure_ratios array and add it to the GeoDataFrame
    lat_lon_gdf['failure_ratios'] = failure_ratios.flatten()
    # get rid of NaNs
    land_based_gdf = lat_lon_gdf.dropna(subset=['population']).copy()

    # Filter failure_ratios to only include corresponding land-based cells
    failure_ratios_filtered = failure_ratios.flatten()[land_based_gdf.index]

    # Add failure_ratios to land_based_gdf
    land_based_gdf['failure_ratios'] = failure_ratios_filtered

    # Group by region
    regional_grouped = land_based_gdf.groupby('Acronym')

    # Calculate weighted average of failure ratios, normalised by total population in the region
    # regional_sum = regional_grouped.apply(lambda group: np.sum(group['failure_ratios'] * group['population'])) # if none-population-weighted input data is used
    regional_sum = regional_grouped['failure_ratios'].sum()
    logging.info('regional_sum')
    regional_population = regional_grouped['population'].sum()
    regional_weighted_average = regional_sum / regional_population
    logging.info('regional_weighted_average')

    # Calculate weighted standard deviation of failure ratios for each region
    def weighted_std(group):
        diff = group['failure_ratios'] - regional_weighted_average.loc[group.name]
        weighted_diff_sq = (diff ** 2) * group['population']
        # Count the number of non-zero and non-NaN weights
        valid_weights = group['population'][(group['population'] != 0) & ~np.isnan(group['population'])]
        M = len(valid_weights)
        if M > 1:
            return np.sqrt(weighted_diff_sq.sum() / ((M - 1) / M * group['population'].sum()))
        else:
            return np.nan  # Return NaN if M is 1 or less

    regional_std = regional_grouped.apply(weighted_std)

    # Create an empty DataFrame with MultiIndex structure
    columns = pd.MultiIndex.from_product([regional_weighted_average.index, ['Mean', 'Std']])
    regional_stats_df = pd.DataFrame(columns=columns)

    # Fill in the DataFrame
    for region in regional_weighted_average.index:
        regional_stats_df.loc[0, (region, 'Mean')] = regional_weighted_average[region]
        regional_stats_df.loc[0, (region, 'Std')] = regional_std[region]

    # Set the index name based on elbow_percentiles
    ### as string...
    # percentile_str = ','.join(map(str, elbow_percentiles))
    # regional_stats_df.index = [percentile_str]
    ### as tuple...
    regional_stats_df.index = [tuple(scenario_percentiles)]

    return regional_stats_df

# initialize worker processes
def init_worker(metrics_dicts, ipcc_regions, lat_lon_gdf, all_combinations, bundle):
    # declare scope of a new global variable
    global shared_metrics_dicts
    global shared_ipcc_regions
    global shared_lat_lon_gdf
    global shared_all_combinations
    global shared_bundle
    global shared_root
    global shared_output_dir
    # store argument in the global variable for this process
    shared_metrics_dicts = metrics_dicts
    shared_ipcc_regions = ipcc_regions
    shared_lat_lon_gdf = lat_lon_gdf
    shared_all_combinations = all_combinations
    shared_bundle = bundle
    #shared_root = "G:/Shared drives/Workstation/1-research/script-command/par-test-python/DATA/"
    shared_root = "C:/Users/Tom_Wood/Documents/DATA/"
    #shared_output_dir = 'G:/Shared drives/Workstation/1-research/script-command/par-test-python/OUTPUT/'
    shared_output_dir = 'C:/Users/Tom_Wood/Documents/OUTPUT/TEST/'

def hadi_calculate_results_for_thresholds(id, shared_metrics_dicts, shared_lat_lon_gdf, shared_all_combinations, shared_bundle, shared_output_dir):
    logging.info(f"Running on process ID {os.getpid()}")
    # Select the elbow arrays based on the current combination of percentiles
    frames = []
    for combo in shared_all_combinations[(id*shared_bundle)-shared_bundle:(id*shared_bundle)]:
        selected_scenario_arrays = select_scenario_array(scenario_percentiles=combo,
                                                        metrics_dicts=shared_metrics_dicts)

        # Calculate the failure ratios
        failure_ratios = calculate_cell_failure_ratio(selected_arrays=selected_scenario_arrays)

        # Calculate the regional statistics
        regional_stats_df = calculate_weighted_regional_average(failure_ratios=failure_ratios,
                                                                lat_lon_gdf=shared_lat_lon_gdf,
                                                                scenario_percentiles=combo)

        # Set the row index based on the current combination of percentiles
        regional_stats_df.index = [tuple(combo)]
        frames.append(regional_stats_df)

    #batch job save
    results = pd.concat(frames)
    #results.to_csv(f'{shared_output_dir}partial_results_{(id*shared_bundle)-shared_bundle}_to_{(id*shared_bundle)-1}.csv')
    results.to_pickle(f'{shared_output_dir}partial_results_{(id*shared_bundle)-shared_bundle}_to_{(id*shared_bundle)-1}_TEST.pkl')
    #logging.info(f'{shared_output_dir}partial_results_{(id*shared_bundle)-shared_bundle}_to_{(id*shared_bundle)-1}.csv  ... SAVED!')
    logging.info(f'{shared_output_dir}partial_results_{(id*shared_bundle)-shared_bundle}_to_{(id*shared_bundle)-1}.pkl  ... SAVED!')

    frames = []   # Clear the list to save memory
    results = []  # Clear the list to save memory
    return "job done"

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='[%(asctime)s] [%(levelname)s] [%(process)d] %(message)s',
        filename='app_TEST.log',
        filemode='w'
    )
    #root = "G:/Shared drives/Workstation/1-research/script-command/par-test-python/DATA/"
    root = "C:/Users/Tom_Wood/Documents/DATA/"
    output_dir = 'C:/Users/Tom_Wood/Documents/OUTPUT/TEST/'

    metrics_paths = glob.glob(f'{root}population_weighted_metric_failure_percentiles/*')
    ipcc_regions_path = 'C:/Users/Tom_Wood/Documents/DATA/population/IPCC-WGI-reference-regions-v4.geojson'
    lat_lon_gdf_path = 'C:/Users/Tom_Wood/Documents/DATA/population/lat_lon_gdf.geojson'
    population_data_path = 'C:/Users/Tom_Wood/Documents/DATA/population/gpw_v4_population_count_rev11_1_deg.nc'

    metrics_dicts, ipcc_regions, lat_lon_gdf = load_data(metrics_paths, ipcc_regions_path, lat_lon_gdf_path,
                                                         population_data_path)

    # print(metrics_dicts)
    logging.info('data_loaded')

    lat_lon_gdf = assign_regions_to_grid(lat_lon_gdf, ipcc_regions)
    logging.info("'Acronym' assigned to lat_lon_gdf")

    percentiles_range = list(range(5, 26, 10))
    logging.info(f'percentiles_range:  \b{percentiles_range[:5]}')

    # Generate all combinations of 7 metrics
    all_combinations = list(product(percentiles_range, repeat=7))
    logging.info(f'combinations:  {all_combinations[:5]}')
    logging.info(len(all_combinations))

    logging.info('ready...')

    # Initialize the Process Pool Executor
    #num_workers = os.cpu_count() - 1
    cores = 80
    num_workers = min(cores, os.cpu_count() - 1)
    logging.info(f'num_workers: {num_workers}')

    # manual chunking
    def split_into_chunks(input_list, chunk_size):
        """
        Splits the input list into chunks of the specified size.

        :param input_list: The list to be split.
        :param chunk_size: The size of each chunk.
        :return: A list of lists representing the chunks.
        """
        # Calculate the number of chunks
        num_chunks = (len(input_list) + chunk_size - 1) // chunk_size

        # Split the list into chunks
        chunks = [input_list[i * chunk_size:(i + 1) * chunk_size] for i in range(num_chunks)]

        return chunks

    chunk_size = 100  # Number of iterations to save after
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
                if not os.path.exists(f'{output_dir}partial_results_{(id * chunk_size) - chunk_size}_to_{(id * chunk_size) - 1}.pkl')
                ]

    #filtered_ids = already_exists(ids, chunk_size)

    print('hello')
    from datetime import datetime
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f'START TIME: {current_time}')

    # using Dask
    from dask import delayed, compute
    import dask.bag as db
    from dask.distributed import Client, LocalCluster

    local_directory = "C:/Users/Tom_Wood/Documents/OUTPUT/dask-scratch"
    print(local_directory)

    try:
        print("Creating LocalCluster")
        # Create a LocalCluster with the specified number of cores
        cluster = LocalCluster(n_workers=num_workers, threads_per_worker=1, local_directory=local_directory)
        print("Created LocalCluster")

        print("Connecting Client to cluster")
        # Connect a client to the cluster
        client = Client(cluster)
        print("Connected Client to cluster")

        # Print the memory limit of each worker in the cluster
        for worker in cluster.workers.values():
            print(f"Worker {worker.name} has a memory limit of {worker.memory_limit} bytes")

        # Print out the address of the dashboard
        print(f"Dask dashboard is available at {client.dashboard_link}")

        print("creating Dask Bag")
        # Create a Dask Bag from the iterable
        dask_bag = db.from_sequence(ids)
        print("Dask Bag created")

        # Define a wrapper function to pass the necessary arguments and configurations
        def hadi_calculate_results_for_thresholds_wrapper(id):
            return hadi_calculate_results_for_thresholds(
                id,
                shared_metrics_dicts=metrics_dicts,
                shared_lat_lon_gdf=lat_lon_gdf,
                shared_all_combinations=all_combinations,
                shared_bundle=chunk_size,
                shared_output_dir='C:/Users/Tom_Wood/Documents/OUTPUT/'
            )

        # Map the function over the Dask Bag
        results = dask_bag.map(hadi_calculate_results_for_thresholds_wrapper)
        # Compute the results using Dask’s multiprocessing scheduler
        #computed_results = results.compute(scheduler='processes', num_workers=num_workers)
        computed_results = results.compute()

        client.close()
    finally:
        if 'client' in locals():
            client.close()

    # create and configure the process pool
    #with Pool(processes=num_workers, initializer=init_worker, initargs=(metrics_dicts, ipcc_regions, lat_lon_gdf, all_combinations, chunk_size)) as pool:
    #    # issue tasks into the process pool
    #    #pool.map(hadi_calculate_results_for_thresholds, range(100))
    #    pool.map(hadi_calculate_results_for_thresholds, ids)

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    print(f'END TIME: {current_time}')