import os
import glob

def create_data_dict(index_list, root):
    data_dict = {}
    
    for index in index_list:
        r_dict = {}
        data_list = glob.glob(os.path.join(root, f'{index}/regridded_{index.lower()}E*.nc'))
        r_list = set()
        
        for f in data_list:
            r_key = 'no-base' if 'no-base' in f else 'b'
            r = f.split(f'ssp585_')[1].split(f'_{r_key}')[0]
            r_list.add(r)
            
        for r in sorted(r_list):
            r_files = [f for f in data_list if r in f]
            r_dict[r] = r_files
        
        data_dict[index] = r_dict
    
    return data_dict


import geopandas as gpd

def load_region_geojson(regions_path = f'/shared/rise_group/User/ci1twx/geojson/IPCC-WGI-reference-regions-v4.geojson'):
    regions_data = gpd.read_file(regions)
    
    # create region object using the headings in the geojson file (i.e. check headings above and insert correct names)
    regions_g = regionmask.from_geopandas(regions_data, names="Name", abbrevs="Acronym", name = "ipcc_regions")
    
    return regions_g