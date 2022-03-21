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
        'm1': [30, 0.5],
        'm2': [60, 0.5],
        'm3': [75, 0.5],
        'm4': [10, 4]
        }
    
    
    points, radius = params[model]
    max_dist = radius*2 
    model = dc.MaxDiameterClustering(max_distance = max_dist, metric = my_hav) 
    labels = model.fit_predict(coords)
    
    return labels


def cluster_model(df, city, ac_name, params):
    
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
    
    if len(coords) == 1:
        labels = 0
    else:
        labels = compute_MaxDC(params = params, model='m1', coords=coords)
    
    df_ac_1 = df_ac.copy()
    df_ac_1['cluster'] = labels
    df_ac_1['cluster'] = df_ac_1['cluster'].astype('str')
    df_ac_1['cluster_name'] = f'{ac_name}_m1_' + df_ac_1['cluster']
    
    sum_bw1 = df_ac_1[(df_ac_1['cluster_name']!=f'{ac_name}_m1_0')& (df_ac_1['Bandwidth'] < 2048)]
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
    if len(coords) == 1:
        labels = 0
    else:
        labels = compute_MaxDC(params = params, model='m2', coords=coords)
    
    df_ac_2 = m1_output.copy()
    df_ac_2['cluster'] = labels
    df_ac_2['cluster'] = df_ac_2['cluster'].astype('str')
    df_ac_2['cluster_name'] = f'{ac_name}_m2_' + df_ac_2['cluster']
    
    df_ac_2.drop_duplicates(inplace = True)
    sum_bw2 = df_ac_2[(df_ac_2['cluster_name']!=f'{ac_name}_m2_0')& (df_ac_2['Bandwidth'] < 1024)]
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
    if len(coords) == 1:
        labels = 0
    else:
        labels = compute_MaxDC(params = params, model='m3', coords=coords)
    
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
    if len(coords) == 1:
        labels = 0
    else:
        labels = compute_MaxDC(params = params, model='m4', coords=coords)
    
    df_ac_4 = m3_output.copy()
    df_ac_4['cluster'] = labels
    df_ac_4['cluster'] = df_ac_4['cluster'].astype('str')
    df_ac_4['cluster_name'] = f'{ac_name}_m4_' + df_ac_4['cluster']
    m4_noise = df_ac_4[df_ac_4.cluster_name == f'{ac_name}_m4_0']
    
    print('-----------x--------Combining Model Outputs-----------x--------------')
    
    sum_bw1 = df_ac_1[(df_ac_1['cluster_name']!=f'{ac_name}_m1_0')& (df_ac_1['Bandwidth'] >=2048)]
    print(sum_bw1.shape)
    print(f'Model1 (High Bandwidth Model): {(len(df_ac_1) - len(m1_noise))}/{len(df_ac_1)} data points in {ac_name} were clustered into {len(np.unique(df_ac_1.cluster_name))} clusters')

    sum_bw2 =  df_ac_2[(df_ac_2['cluster_name']!=f'{ac_name}_m2_0')& (df_ac_2['Bandwidth'] >=1024)]
    print(sum_bw2.shape)
    print(f'Model2 (Medium Bandwidth Model): {(len(df_ac_2) - len(m2_noise))}/{len(df_ac_2)} data points in {ac_name} were clustered into {len(np.unique(df_ac_2.cluster_name))} clusters')

    # Clusters without noise
    sum_bw3 =  df_ac_3[df_ac_3['cluster_name']!=f'{ac_name}_m3_0']
    print(f'Model3 (Crossover Model): {(len(df_ac_3) - len(m3_noise))}/{len(df_ac_3)} data points in {ac_name} were clustered into {len(np.unique(df_ac_3.cluster_name))} clusters')
    print(sum_bw3.shape)

    #sum_bw4 =  df_ac_4[(df_ac_4['cluster_name']!=f'{ac_name}_m4_0')& (df_ac_4['Bandwidth'] >= 20)]
    sum_bw4 = df_ac_4[df_ac_4['Bandwidth']>=20]
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


 