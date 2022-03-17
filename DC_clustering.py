import pandas as pd
import numpy as np
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
 
from utils import *


def main():
    
    
    """
    Button-1: Choose from T1 or T2 or RoI (rest of India):
    Button-2: Provide option from T1 or T2 or ROI based on Button-1
    (Also add 'ALL' test case if T1 or T2)
    
    Params: Provide option for choosing the appropriate parameter for cluster
    
    
    User Input-1:
        city;
        check if city lies in t1/t2 or t3:
        if t1/t2:
            User input-2:
            ac_name; (provide option None)
            
    Add a test case to check if AC falls in the same T1/T2 city else raise Error
    add option # t1, t2, rest of India
    add a dropdown
    1)user should add more data, and drop duplicates (add info about data)
    2) created in Tata Comm
    22nd
    """
    
    
    
   
   
    #city = 'Delhi'
    #ac_name='Noida'
    #k = 1 if city!=ac_name:else 0
        
    t1_t2 = list(np.unique(df['CITY']))
    t3 = list(np.unique(df['AC_NAME'])) 
    
    # Button 1 
    city = input('Please Enter the name of the city ')
    if city in t1_t2:
        #city_subset = df[df['CITY']== city] 
        
    
        # Button 2:
        print('You have entered T1 or T2 cities;\n Do you want to include an Assembly Constituency?')
        j = input('enter [Y]/[N] ')
        if j == 'Y':
            ac_name = input('Enter the name of the assembly constituency ')
             
            if ac_name in t3:
                
                df_ac = df[(df['CITY']==city)&(df['AC_NAME']==ac_name)]
            else:
                raise('Key Error: Invalid Assembly Constituency name')
        
        if j == 'N':
            
            ac_names = list(np.unique(df[df['CITY']==city]['AC_NAME']))
            
    elif city in t3:
        print('You have entered a T3 city')
        #df_ac = df[df['AC_NAME']== city]
    
    elif city not in set(t1_t2).union(set(t3)) :
        print('The city you entered does not exist, please check the spelling')
        print('Kindly provide a city name among the ones we have')
        print(np.unique(df['CITY'])) 
        
        
    if 'ac_names' in locals():
        
        print('No')
        final_clustered_data = []
        final_fig = []
        #fig, clustered_data = compute_clustering(df)
        
        for ac_name in ac_names:
            
            print(f'Computing Max Diameter Clustering for city:{city}, AC:{ac_name} ')
            fig, clustered_data = compute_clustering(df, city, ac_name)
            
            final_clustered_data.append(clustered_data)
            final_fig.append(fig)
            del fig, clustered_data
        
        final_data = pd.concat(final_clustered_data)
        fig = px.scatter_mapbox(final_clustered_data, lat="lat", lon="lon", color="cluster_name",
                                hover_name='name', hover_data=["Bandwidth"], color_continuous_scale = px.colors.cyclical.IceFire)
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":1,"t":1,"l":1,"b":1})
        fig.show()
        #fig.write_html(f'C:\\Users\\clouduser1\\NW Planning\\{city}-{ac_name}.html')
            
    
    else:
        
        print(f'Computing Max Diameter Clustering for city:{city}, AC:{ac_name} ')
        fig, clustered_data = compute_clustering(df, city, ac_name)  
        fig.show()
        
if __name__ == "__main__":

"""
    Button-1: Choose from T1 or T2 or RoI (rest of India):
    Button-2: Provide option from T1 or T2 or ROI based on Button-1
    (Also add 'ALL' test case if T1 or T2)
    
    Params: Provide option for choosing the appropriate parameter for cluster
    
    Default params:
                params = {
            'm1': [30, 0.5],
            'm2': [60, 0.5],
            'm3': [75, 0.5],
            'm4': [10, 4]
            }
    
    User Input-1:
        city;
        check if city lies in t1/t2 or t3:
        if t1/t2:
            User input-2:
            ac_name; (provide option None)
            
    Add a test case to check if AC falls in the same T1/T2 city else raise Error
    add option # t1, t2, rest of India
    add a dropdown
    1)user should add more data, and drop duplicates (add info about data)
    2) created in Tata Comm
    22nd
"""
    df = pd.read_csv(r'C:\Users\clouduser1\NW Planning\master_t1_t2_t3.csv')
    
    main()