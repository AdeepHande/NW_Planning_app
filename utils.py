import pandas as pd
import numpy as np
import streamlit as st
from numpy import array
import matplotlib.pyplot as plt
from tqdm import tqdm
from geopandas import GeoSeries
import haversine as hs
import geopandas as gpd
import shapely.geometry
from shapely.geometry import Point, asPoint, Polygon, MultiPoint 
import os
import json
import geog
import glob
import time
from sklearn import metrics
from geopy.distance import great_circle
from geopandas import GeoDataFrame
from shapely.geometry import MultiPoint
from tqdm import tqdm
import zipfile
import diameter_clustering as dc
from scipy.spatial.distance import pdist as pdist
from zipfile import ZipFile
import re
import plotly.express as px 

def my_hav(c1, c2):
    # calculate haversine distance for lat long co-oridnates
    return hs.haversine(c1, c2)


def compute_MaxDC(params,model, coords):
    """
    params: "m1" or "m2" or "m3" or "m4"
    coords: df_ac[['lat', 'lon']].to_numpy()
    """
    if params is None:
        
        params = {
        'm1': [30, 0.5, 2048],
        'm2': [60, 0.5, 1024],
        'm3': [75, 0.5, None],
        'm4': [10, 4, 20]
        }
    
    
    points, radius, bandwidth = params[model]
    max_dist = radius*2 
    
    if len(coords) < 2:
        labels  = 0
    
    else:
        model = dc.MaxDiameterClustering(max_distance = max_dist, metric = my_hav) 
        labels = model.fit_predict(coords)
    
    return labels, bandwidth
 
def cluster_model_block(df, city, ac_name, params,compute):
    
    
    # 
    """
    Parameters
    ----------
    model : str --> [m1, m2, m3, m4]     
        
    df :pd.DataFrame()
        Input DataFrame.
        
    ac_name: str --> ac_name  
        
    params: list--> parameters for the clustering    
    
    compute: bool --> [True, False]
    if True:
        
        compute clustering
    else:
        
        skip the block
        
    Returns
    -------
    final clusterd data
    """ 
    
    
    #df_ac = df[(df['CITY']==city)&(df['AC_NAME']==ac_name)]
    final_output = []
    bandwidth = []
    
    for comp in compute: 
        
        model, c = comp
        
        if c == 1:   
            
            points, radius, bandwidth = params[model]
            
            coords = df[['lat', 'lon']].to_numpy()
            
            if len(coords) < points:
                
                st.info(f'The number of data points in {ac_name} ')
                break
            
            else:
                
                labels = compute_MaxDC(params=params, model=model, coords=coords)
                
                df_ac_1 = df.copy()
                df_ac_1['cluster'] = labels
                df_ac_1['cluster'] = df_ac_1['cluster'].astype('str')
                df_ac_1['cluster_name'] = f'{ac_name}_m1_' + df_ac_1['cluster']
                
                sum_bw = df_ac_1[(df_ac_1['cluster_name']!=f'{ac_name}_{model}_0')& (df_ac_1['Bandwidth'] < int(bandwidth))]
                 
                m_noise = df_ac_1[df_ac_1.cluster_name == f'{ac_name}_{model}_0']
                
                if model == 'm3':
                    
                    m_output = pd.concat(m_noise)
                    
                elif len(m_noise) == 0:
                    m_output = pd.concat([sum_bw]) 
                    
                else:
                    m_output = pd.concat([sum_bw, m_noise]) 
             
                m_output.drop_duplicates(inplace=True)
                m_output[m_output.duplicated()]
                
                df = m_output
                
                final_output.append(m_output)
        
        else:
            
            break
        
    return [output for output in final_output]

    
    
# pass compute as a list
     
def blockwise_modeling(model, df, city, ac_name, params, compute):    
    
     
    """ 
    Parameters
    ----------
    model : str --> [m1, m2, m3, m4]     
        
    df :pd.DataFrame()
        Input DataFrame.
        
    ac_name: str --> ac_name  
        
    params: list--> parameters for the clustering    
    
    compute: list --> [(0/1), (0/1), (0/1), (0/1)] 
    if 1:
        
        compute clustering
        
    else:
        
        skip the block
        
    Returns
    -------
    final clusterd data
    
    """  
    if compute is None:
        
        compute = [['m1',1],
                   ['m2',1],
                   ['m3',1], 
                   ['m4',1]]
    
        
    df_ac = df[(df['CITY']==city)&(df['AC_NAME']==ac_name)]
    
    m1_output, m2_output, m3_output, m4_output = cluster_model_block(model, df, city, ac_name, params, compute)
    
    
        
        
    return m1_output
     
def cluster_model(df, city, ac_name, params):
    # drop cities if num(data points) < params[data_points]
    # while model!=0
    #ac_subset_master = []
    #if ac_name is None:
    #    ac_names = list(np.unique(city_subset['AC_NAME']))
    #    for ac in ac_names:
    #        ac_subset_master.append(city_subset[city_subset['AC_NAME']==ac])
            # Add for loop to iterate over the whole place
    
    #else: t1, 
    df_ac = df[(df['CITY']==city)&(df['AC_NAME']==ac_name)]
        
    print("-----------x---------Model 1 (High Bandwidth) -----------x--------------") # Add more comments
    
    coords = df_ac[['lat', 'lon']].to_numpy()
    
 
    labels, bw1 = compute_MaxDC(params = params, model='m1', coords=coords)
    #if labels  == 0:
   #     st.write('The number of data points')
    df_ac_1 = df_ac.copy()
    df_ac_1['cluster'] = labels
    df_ac_1['cluster'] = df_ac_1['cluster'].astype('str')
    df_ac_1['cluster_name'] = f'{ac_name}_m1_' + df_ac_1['cluster']
    
    sum_bw1 = df_ac_1[(df_ac_1['cluster_name']!=f'{ac_name}_m1_0')& (df_ac_1['Bandwidth'] < int(bw1))]
    print(sum_bw1.shape)
    m1_noise = df_ac_1[df_ac_1.cluster_name == f'{ac_name}_m1_0']
    print(m1_noise.shape)
    print(f'{len(sum_bw1)}/{len(df_ac_1)} data points in {ac_name} were clustered into {len(np.unique(df_ac_1.cluster_name))} clusters')
    print(f'There are {len(sum_bw1)} data points in {ac_name} where the bandwidth is less than 2 GB')
    print(f'Feeding {len(sum_bw1)}/{len(df_ac_1)} data points to model 2')
    print(len)
    if len(m1_noise) == 0:
        m1_output = pd.concat([sum_bw1])
        print('There are no noisy points from model1 (high bandwidth)')
        print('Feeding datapoints where bandwid < 2GB to model2')
    else:
        m1_output = pd.concat([sum_bw1, m1_noise])
        print(f'There are {len(m1_noise)} noisy/unallocated points from model1 (high bandwidth)')
        print('Merging noisy points and datapoints where bandwidth <2GB')

    print("Clusters with Bandwidth below 2 Gb ---> To be fed into Model 2")
    
    m1_output.drop_duplicates(inplace=True)
    m1_output[m1_output.duplicated()]
    
    print('-----------x---------Model 2 (Medium Bandwidth) -----------x--------------') # Add more comments 
    
    coords = m1_output[['lat', 'lon']].to_numpy()
     
    labels, bw2 = compute_MaxDC(params = params, model='m2', coords=coords)
    
    df_ac_2 = m1_output.copy()
    df_ac_2['cluster'] = labels
    df_ac_2['cluster'] = df_ac_2['cluster'].astype('str')
    df_ac_2['cluster_name'] = f'{ac_name}_m2_' + df_ac_2['cluster']
    
    df_ac_2.drop_duplicates(inplace = True)
    sum_bw2 = df_ac_2[(df_ac_2['cluster_name']!=f'{ac_name}_m2_0')& (df_ac_2['Bandwidth'] < bw2)]
    print(sum_bw2.shape)
    m2_noise = df_ac_2[df_ac_2.cluster_name == f'{ac_name}_m2_0']
    print(m2_noise.shape)
    print(f'{(len(df_ac_2) - len(m1_noise))}/{len(df_ac_2)} data points in {ac_name} were clustered into {len(np.unique(df_ac_2.cluster_name))} clusters')
    print(f'There are {len(sum_bw2)} data points in {ac_name} where the bandwidth is less than 1 GB')
    print(f'Feeding {len(sum_bw2)}/{len(df_ac_2)} data points to model 3')
    print(len)
    if len(m2_noise) == 0:
        m2_output = pd.concat([sum_bw2])
        print('There are no noisy points from model1 (high bandwidth)')
        print('Feeding datapoints where bandwid < 1GB to model2')

    else:
        print('Merging noisy points and datapoints where bandwidth <1 GB')
        print(f'There are {len(m2_noise)} noisy/unallocated points from model1 (medium bandwidth)')
        m2_output = pd.concat([sum_bw2, m2_noise])
        
    print("Clusters with Bandwidth below 2 Gb ---> To be fed into Model 2")
    m2_output.drop_duplicates(inplace=True)
    m2_output[m2_output.duplicated()]

    print('-----------x---------Model 3 (Crossover Model)-----------x--------------')
    # Add a test case if only 1 unallocated point is left.
    # Should add another case if all points are allocated and all bandwidth are are greater than 1 Gbps  
    
    coords = m2_output[['lat', 'lon']].to_numpy()
    
    labels, _ = compute_MaxDC(params = params, model='m3', coords=coords)
    
    df_ac_3 = m2_output.copy()
    df_ac_3['cluster'] = labels
    df_ac_3['cluster'] = df_ac_3['cluster'].astype('str')
    df_ac_3['cluster_name'] = f'{ac_name}_m3_' + df_ac_3['cluster']
    
    df_ac_3.drop_duplicates(inplace = True) 
    m3_noise = df_ac_3[df_ac_3.cluster_name == f'{ac_name}_m3_0']
    print(m3_noise.shape)
    print(f'{(len(df_ac_3) - len(m3_noise))}/{len(df_ac_3)} data points in {ac_name} were clustered into {len(np.unique(df_ac_3.cluster_name))} clusters')
    print('Merging noisy points')
    print(f'There are {len(m3_noise)} noisy/unallocated points from model1 (crossover model)')
    m3_output = pd.concat([m3_noise])
    
    
    print('-----------x---------Model 4 (Wireless Model)-----------x--------------')
    print('Min samples = 10, radius = 4km')
    print('Noise points from Model 3 is fed into Model 4')
    
    # Add a test case if only 1 unallocated point is left.
    # Should add another case if all points are allocated at m3
    coords = m3_output[['lat', 'lon']].to_numpy()
    
    labels, bw4 = compute_MaxDC(params = params, model='m4', coords=coords)
    
    df_ac_4 = m3_output.copy()
    df_ac_4['cluster'] = labels
    df_ac_4['cluster'] = df_ac_4['cluster'].astype('str')
    df_ac_4['cluster_name'] = f'{ac_name}_m4_' + df_ac_4['cluster']
    m4_noise = df_ac_4[df_ac_4.cluster_name == f'{ac_name}_m4_0']
    
    print('-----------x--------Combining Model Outputs-----------x--------------')
    
    sum_bw1 = df_ac_1[(df_ac_1['cluster_name']!=f'{ac_name}_m1_0')& (df_ac_1['Bandwidth'] >=bw1)]
    print(sum_bw1.shape)
    print(f'Model1 (High Bandwidth Model): {(len(df_ac_1) - len(m1_noise))}/{len(df_ac_1)} data points in {ac_name} were clustered into {len(np.unique(df_ac_1.cluster_name))} clusters')

    sum_bw2 =  df_ac_2[(df_ac_2['cluster_name']!=f'{ac_name}_m2_0')& (df_ac_2['Bandwidth'] >=bw2)]
    print(sum_bw2.shape)
    print(f'Model2 (Medium Bandwidth Model): {(len(df_ac_2) - len(m2_noise))}/{len(df_ac_2)} data points in {ac_name} were clustered into {len(np.unique(df_ac_2.cluster_name))} clusters')

    # Clusters without noise
    sum_bw3 =  df_ac_3[df_ac_3['cluster_name']!=f'{ac_name}_m3_0']
    print(f'Model3 (Crossover Model): {(len(df_ac_3) - len(m3_noise))}/{len(df_ac_3)} data points in {ac_name} were clustered into {len(np.unique(df_ac_3.cluster_name))} clusters')
    print(sum_bw3.shape)

    sum_bw4 =  df_ac_4[(df_ac_4['cluster_name']!=f'{ac_name}_m4_0')& (df_ac_4['Bandwidth'] >= bw4)]
    #sum_bw4 = df_ac_4[df_ac_4['Bandwidth']>=bw4]
    print(f'Model4 (Wireless Model): {(len(df_ac_4) - len(m4_noise))}/{len(df_ac_4)} data points in {ac_name} were clustered into {len(np.unique(df_ac_4.cluster_name))} clusters')
    print(sum_bw4.shape)
    
    final_data = pd.concat([sum_bw1, sum_bw2, sum_bw3, sum_bw4]) 
    
     
    #fig = px.scatter_mapbox(final_data, lat="lat", lon="lon", color="cluster_name",
    #                        hover_name='name', hover_data=["Bandwidth"], color_continuous_scale = px.colors.cyclical.IceFire)
    #fig.update_layout(mapbox_style="open-street-map")
    #fig.update_layout(margin={"r":1,"t":1,"l":1,"b":1})
    #fig.show()
    #fig.write_html(f'D:\\NW_Planning\\{city}-{ac_name}.html')
    
    return final_data

 

def compute_all(df, city_list, params):
    
    final_clustered_data = []
    final_fig = [] 
        
          
    if len(df['CITY'].unique()) > 1:
        
        
        for city in city_list:
    
            ac_names = np.unique(df[df['CITY']==city]['AC_NAME'])
            flag = 0
            for ac_name in ac_names:
                
                df = df[(df['CITY']==city)&(df['AC_NAME']==ac_name)]
                
            
                #fig, clustered_data = compute_clustering(df)
                
                 
                print('')
                print('')
            
                print(f'Computing Max Diameter Clustering for city:{city}, AC:{ac_name} ')
                
                print('')
                print('')
                print('')
                
                clustered_data = cluster_model(df,city,ac_name,params)
                print('')
                print('')
                print('')
                
                print(f'Completed Max Diameter Clustering for city:{city}, AC:{ac_name}')
                final_clustered_data.append(clustered_data) 
                info1 = st.info(f'We have clustered all data points in {ac_name} of {city}')
                time.sleep(2)
                info1.empty()
                info2 = st.info(f'{len(ac_names) - flag} cities left in {city}')
                time.sleep(2)
                info2.empty() 
                flag += 1  
                
        final_data = pd.concat(final_clustered_data)
    
    else:
        
        ac_names = np.unique(df['AC_NAME'])
        flag = 0
        for ac_name in ac_names:
            
            df = df[(df['AC_NAME']==ac_name)]
            
        
            #fig, clustered_data = compute_clustering(df)
            
             
            print('')
            print('')
        
            print(f'Computing Max Diameter Clustering for city:{city}, AC:{ac_name} ')
            
            print('')
            print('')
            print('')
            
            clustered_data = cluster_model(df,city,ac_name,params)
            print('')
            print('')
            print('')
            
            print(f'Completed Max Diameter Clustering for city:{city}, AC:{ac_name}')
            final_clustered_data.append(clustered_data) 
            info1 = st.info(f'We have clustered all data points in {ac_name} of {city}')
            time.sleep(2)
            info1.empty()
            info2 = st.info(f'{len(ac_names) - flag} cities left in {city}')
            time.sleep(2)
            info2.empty() 
            flag += 1 
        final_data = pd.concat(final_clustered_data)
         
        #fig = px.scatter_mapbox(final_data, lat="lat", lon="lon", color="cluster_name",
        #hover_name='name', hover_data=["Bandwidth"], color_continuous_scale = px.colors.cyclical.IceFire)
        #fig.update_layout(mapbox_style="open-street-map")
        #fig.update_layout(margin={"r":1,"t":1,"l":1,"b":1})
        #fig.show()
        return final_data


def compute_nearest_pop(df_type, df, parent_df):
    
    """
    df_type: wireline, wireless, handhole
    df: clustered data with columns: [Cluster_name, Lat, Lon]
    parent_df: internal data, should be in [wireless, wireline, handhole]
    """
    #if df_type == 'wireline':
    for index, row in df.iterrows():
    #     print(index, row['lat'], row['lon'])
        centermost = (row['lat'],row['lon'])
        min_dist = 10000 # equivalent to inf
        nearest_pop = ()
        for j_index, j_row in parent_df.iterrows():
            pop = (j_row['Lattitude'], j_row['Longitude'])
            cur_dist = great_circle(centermost, pop).km  
            if(cur_dist<=min_dist):
                min_dist = cur_dist
                nearest_pop = pop
                #owner = j_row['owner']
                if df_type == 'handhole':
                    site_id = j_row['System Id']
                else:
                    site_id = j_row['Site ID']

        print("centre ", index,f": Nearest {df_type} POP :",nearest_pop," Distance: ", min_dist)
        df.loc[index,f"{df_type}_site_id"] = site_id
        df.loc[index,f"{df_type}_pop_lat"] = nearest_pop[0]
        df.loc[index,f"{df_type}_pop_long"] = nearest_pop[1]
        df.loc[index,f"distance_from_{df_type}_pop"] = min_dist
        #df.loc[index,"wireline_owner"] = owner

    # Removing clusters within 0.5km of the POP for m1, m2, m3 output
    # Removing clusters within 3 km of the POP for m4 output
    df = df[((~df['cluster_name'].str.contains('m4')) & (df[f'distance_from_{df_type}_pop'] > 0.5)) | ((df['cluster_name'].str.contains('m4'))&(df[f'distance_from_{df_type}_pop'] > 3))]    
    
    # m1_m2_m3 =  df[(~df['cluster_name'].str.contains('m4'))& (df[f'distance_from_{df_type}_pop'] > 0.5)]
    # m4 = df[(df['cluster_name'].str.contains('m4'))&(df[f'distance_from_{df_type}_pop'] > 3)]
    #df = pd.concat([m1_m2_m3, m4])
    
    return df

def create_final_matrix_with_category(df_output, df_wireline, df_wireless, df_handhole):
    
    private_banks = ['axis bank ltd',
                 'bandhan bank ltd',
                 'csb bank ltd',
                 'city union bank ltd',
                 'dcb bank ltd',
                 'dhanlaxmi bank ltd',
                 'federal bank ltd',
                 'hdfc bank ltd',
                 'icici bank ltd',
                 'indusind bank ltd',
                 'idfc first bank ltd',
                 'jammu & kashmir bank ltd',
                 'karnataka bank ltd',
                 'karur vysya bank ltd',
                 'kotak mahindra bank ltd',
                 'lakshmi vilas bank ltd',
                 'nainital bank ltd',
                 'rbl bank ltd',
                 'south indian bank ltd',
                 'tamilnad mercantile bank ltd',
                 'yes bank ltd',
                 'idbi bank ltd']

    public_banks = ['bank of baroda',
                 'bank of india',
                 'bank of maharashtra',
                 'canara bank',
                 'central bank of india',
                 'indian bank',
                 'indian overseas bank',
                 'punjab & sind bank',
                 'punjab national bank',
                 'state bank of india',
                 'uco bank',
                 'union bank of india']

    df_output['category1'] = df_output['category'] 
    bank = df_output[df_output['category'] == 'Bank']
    bank['name'] = bank['name'].str.lower()
    bank['name'] = bank['name'].str.replace(r"limited|ltd.",'ltd', regex = True)
    bank['category'] = np.where(bank['name'].isin(public_banks) == True, 'Public Bank', bank['name'])
    bank['category'] = np.where(bank['category'].isin(private_banks) == True, 'Private Bank', bank['category'])
    bank['category'] = np.where((bank['category'] != 'Public Bank')&(bank['category'] != 'Private Bank'), 'Other Banks', bank['category'])
    
    for i in tqdm(list(bank.index)):
        df_output['category'].loc[i] = bank['category'].loc[i]
    
    # Categorical Count
    cat_count = pd.DataFrame(df_output.groupby(['cluster_name'])['category'].value_counts())
    cat_count.rename(columns = {'category':'count_of_each_category'}, inplace = True)
    cat_count.reset_index(inplace = True)
    
    mat = pd.pivot_table(cat_count, index = ['cluster_name'], columns = 'category', values = 'count_of_each_category')
    mat = mat.fillna(int(0))
    mat.reset_index(inplace = True)
    
    merged_data1 = pd.merge(mat, df_wireline, on = 'cluster_name', how = 'inner', indicator = True)
    merged_data1.rename(columns = {'_merge':'wireline_indicator1'}, inplace = True)
    
    merged_data2 = pd.merge(merged_data1, df_wireless, on = 'cluster_name', how = 'inner', indicator = True)
    merged_data2.rename(columns = {'_merge':'wireless_indicator1'}, inplace = True)
    
    merged_data3 = pd.merge(merged_data2, df_handhole, on = 'cluster_name', how = 'inner', indicator = True)
    merged_data3.rename(columns = {'_merge':'handhole_indicator1'}, inplace = True)
    
    sum_bw1 = pd.DataFrame(df_output.groupby(['cluster_name'])['Bandwidth'].sum())
    sum_bw1.reset_index(inplace = True)
    
    merged_final = pd.merge(merged_data3, sum_bw1, on = 'cluster_name', how = 'inner', indicator = True)
    merged_final.rename(columns = {'_merge':'indicator1'}, inplace = True)
    
    categories = ['ATM', 'Automobile Dealer', 'Clinic', 'Private Bank', 'Public Bank',
       'Corporate Office', 'Hospital', 'Offnet', 'Onnet', 'Other Amenities',
       'Other Banks', 'Railway station', 'SEZ', 'Supermarket', 'University']
    
    cluster_matrix = merged_final
    cluster_matrix['Bandwidth_bins'] = pd.cut(x = cluster_matrix['Bandwidth'], bins = [0, 1000, 2000, 5000,1000000])
    cluster_matrix['Bandwidth_bins'] = cluster_matrix['Bandwidth_bins'].astype('str') 
            
    included_category = []
    for column in cluster_matrix.columns:
        if column in categories:
            included_category.append(column)
            cluster_matrix[column] = cluster_matrix[column].astype('int')
             
     
    cluster_matrix['count_of_categories_in_a_cluster'] = cluster_matrix[included_category].sum(axis=1)
    
    req_category = ['cluster_name', 'ATM', 'Automobile Dealer', 'Clinic',
       'Corporate Office', 'Hospital', 'Offnet', 'Onnet', 'Other Amenities',
       'Other Banks', 'Railway station', 'SEZ', 'Supermarket', 'University',  
       'distance_from_wireline_pop_y', 'distance_from_wireless_pop_x', 'lat', 'lon', 'wireline_site_id',
       'distance_from_handhole_pop_y' 'Bandwidth', 'Bandwidth_bins',
       'count_of_categories_in_a_cluster']
    
    categories_present = []
    for column in cluster_matrix.columns:
        if column in req_category:
            categories_present.append(column)
    cluster_matrix = cluster_matrix[categories_present]
    return cluster_matrix    

 
def data_concat(internal_df, external_df, ac_shape_file):
     
    """

    Parameters
    ----------
    internal_df : pd.DataFrame()
        Master CSV T1, T2, RoI data
        
    external_df :pd.DataFrame()
        User Input DataFrame.
        
    ac_shape_file: gpd.DataFrame()
        Internal GeoDataFrame for spatially joining AC_Names to external_df
        
    
    Returns
    -------
    Concatenated DataFrame with relevant columns.

    """ 
    
    columns = ['CITY', 'name', 'lat', 'lon', 'Bandwidth', 'category', 'identifier','ST_NAME', 'AC_NAME']
       #'geometry',  
    df_dash = internal_df[columns]
    df_dash['type'] = 'internal'
    
    # Comment the next line before deploying it for production
    ext_df_v2 = external_df.rename(columns={'City':'CITY', 'Opportunity Account Name':'name'}) 
    
    ext_df_v2 = ext_df_v2[['name','CITY','Bandwidth', 'category', 'identifier', 'lat', 'lon']]
    ext_df_v2.drop_duplicates()
    geometry = [Point(xy) for xy in zip(ext_df_v2.lon, ext_df_v2.lat)] 
    
    ext_df_v2 = GeoDataFrame(ext_df_v2, crs="EPSG:4326", geometry=geometry)
    
        
    ext_df_v3 = ac_shape_file.sjoin(ext_df_v2, how='inner')
    
    ext_df_v3['type'] = 'user_input'
    if not (set(columns).issubset(set(ext_df_v3.columns))):         
        raise KeyError('There is a column mismatch. Kindly structure the dataframe with the following columns')
        st.write(columns)
        
    else:
    #matching the same columns as internal_df
        ext_df_v3 = ext_df_v3[columns]
        
        final_df = pd.concat([df_dash, ext_df_v3])
        
        # Dropping duplicates
        # Throwing off an error for the timebeing
        #final_df.drop_duplicates()
        final_df.dropna() 
        return final_df
    
    
def convert_df(df):
   return df.to_csv(index=False).encode('utf-8')


def generate_summary(city_summ, file_len, final_df, external_df):
    
     
    
    nm1 = final_df.cluster_name.str.contains('m1').sum()
    nm2 = final_df.cluster_name.str.contains('m2').sum()
    nm3 = final_df.cluster_name.str.contains('m3').sum()
    nm4 = final_df.cluster_name.str.contains('m4').sum()
    
    total_clusters = final_df['cluster_name'].nunique()
    
    pre_num = file_len
    post_num = len(final_df)
    
    city_type, cities = city_summ
    if (external_df == 'yes'):
        if cities == 'All':
            
            summary = f'The user selected {cities}, {city_type} cities. After combining the internal master data and user input, the dataset has {pre_num} data points (feasibility requests). After clustering, out of {pre_num} data points, {post_num} feasibility requests were clustered into {total_clusters} clusters. '
        else:
            
            summary = f'The user selected {cities}, {city_type} cities. After combining the internal master data and user input, the dataset has {pre_num} data points (feasibility requests). After clustering, out of {pre_num} data points, {post_num} feasibility requests were clustered into {total_clusters} clusters. '
    
    else: 
        
        summary = f'The user selected {cities}, {city_type} cities. After combining the internal master data and user input, the dataset has {pre_num} data points (feasibility requests). After clustering, out of {pre_num} data points, {post_num} feasibility requests were clustered into {total_clusters} clusters. '
        
    return summary