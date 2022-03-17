import streamlit as st
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
import time
from utils import *

def main():
    
    
    st.title('AI Based Network Expansion Planning')
    
    city_type = ['Tier 1', 'Tier 2', 'Rest of India']
    
    c_type = st.selectbox('Select the type of city you want to cluster in', city_type) 

    st.write('You selected:', c_type)
    
    t1_all = False
    t2_all = False
    t3_city = None
    if c_type == 'Tier 1':
        
        # Add 'All' as the first option
        t1_list = ['All'] + list(np.unique(df_t1['CITY']))
        
        city = st.selectbox('Select the Tier 1 city of your choice', t1_list)
        
        if city == 'All':
            t1_all = True
            city = t1_list[1:]
            st.write('You selected:', city)
        else:
             
            st.write('You selected', city)
            
    elif c_type == 'Tier 2':
        
        t2_list = ['All'] + list(np.unique(df_t2['CITY']))
        
        city = st.selectbox('Select the Tier 2 city of your choice', t2_list)
        if city == 'All':
            
             
            city = t2_list[1:]
            st.write('You selected:', city)
            
        else:
            t2_all = False
            st.write('You selected', city)
    
    else:
        
        #t3_list = ['All'] + 
        t3_list = list(np.unique(df_t123['AC_NAME']))
        
        city = st.selectbox('Select the Tier 3 city of your choice', t3_list)
        t3_city = city
        
        #if city == 'All':
            
        #    city = t3_list[1:]
        #    st.write('You selected:', city)
        #else:
            
        st.write('You selected', city)  
    
    ## Ask if they want to upload data?
    
    external_data = st.radio(
     "Do you want to upload external data?",
     ('Yes', 'No'))

    if external_data == 'Yes':
        
        uploaded_file = st.file_uploader("Choose a CSV file", accept_multiple_files=False)
        
        if uploaded_file is not None:
            #for uploaded_file in uploaded_files:
            external_df = pd.read_csv(uploaded_file)
            #st.write(dataframe)
        
            st.write('The file has been uploaded successfully')
            st.write('Checking for duplicates after combining the data')
            
            if list(external_df.columns) != list(df_t123.columns):
                
                st.error('The columns of the uploaded data and our internal data does not match')
                st.error('The columns of the uploaded data and our internal data does not match')
                st.write('Try to structure the external data in the following order', df_t123.columns)
            
            else:
                
                st.write('we will concatenate it with the internal data')
                st.write('Checking for duplicates after combining the data')
                master_data = pd.concat([df_t123, external_df])
                master_data.drop_duplicates() 
                
                
    else:
        
        st.write("We will proceed with the internal data on our end")
        st.write('A snippet of our internal data:')
        st.write(df_t123.head(5))
        
    
    # Provide optional parameters
    #param_bool = st.radio(
    # "Do you want to provide parameters to compute the clusters?",
    # ('Yes', 'No'))
    #if param_bool == 'Yes':
        
    #    st.sidebar.subheader('Optional Parameters')
    
    
    # Add a run button
    
    if st.button('Generate Clusters'):
        
        if t1_all == True:
            
            final_data = compute_all(df_t123, t1_list[1:])
            
            fig = px.scatter_mapbox(final_data, lat="lat", lon="lon", color="cluster_name",
                                hover_name='name', hover_data=["Bandwidth"], color_continuous_scale = px.colors.cyclical.IceFire)
            fig.update_layout(mapbox_style="open-street-map")
            fig.update_layout(margin={"r":1,"t":1,"l":1,"b":1})
            fig.show()
            
        elif t2_all == True:
            
            final_data = compute_all(df_t123, t2_list[1:])
    
            fig = px.scatter_mapbox(final_data, lat="lat", lon="lon", color="cluster_name",
                                hover_name='name', hover_data=["Bandwidth"], color_continuous_scale = px.colors.cyclical.IceFire)
            fig.update_layout(mapbox_style="open-street-map")
            fig.update_layout(margin={"r":1,"t":1,"l":1,"b":1})
            fig.show()
            
        elif t3_city is not None:
            
            df = df_t123[df_t123['AC_NAME']==t3_city]
            
            city = np.unique(df[df['AC_NAME']==t3_city]['CITY'])
            
            final_data = cluster_model(df, city[0], t3_city)
            
            fig = px.scatter_mapbox(final_data, lat="lat", lon="lon", color="cluster_name",
                                hover_name='name', hover_data=["Bandwidth"], color_continuous_scale = px.colors.cyclical.IceFire)
            fig.update_layout(mapbox_style="open-street-map")
            fig.update_layout(margin={"r":1,"t":1,"l":1,"b":1})
            fig.show()           
            
        else:
            
            #flag = 0
            final_clustered_data = []
            ac_names = np.unique(df_t123[df_t123['CITY']==city]['AC_NAME'])
            for ac_name in ac_names: 
                 
                
                df = df_t123[(df_t123['CITY']==city)&(df_t123['AC_NAME']==ac_name)] 
            
                print(f'Computing Max Diameter Clustering for city:{city}, AC:{ac_name} ')
                 
                
                clustered_data = cluster_model(df, city, ac_name) 
                print(f'Completed Max Diameter Clustering for city:{city}, AC:{ac_name}')
                
                final_clustered_data.append(clustered_data) 
                 
                #my_bar = st.progress(0)
    
                #flag += int(100/len(ac_names))
                #time.sleep(1)
                #my_bar.progress(flag)  
                info = st.info(f'We have clustered all data points in {ac_name} of {city}')
                #time.sleep(2)
                #info.empty()
                 
            final_data = pd.concat(final_clustered_data)
        
        fig = px.scatter_mapbox(final_data, lat="lat", lon="lon", color="cluster_name",
                                hover_name='name', hover_data=["Bandwidth"], color_continuous_scale = px.colors.cyclical.IceFire)
        fig.update_layout(mapbox_style="open-street-map")
        fig.update_layout(margin={"r":1,"t":1,"l":1,"b":1})
        fig.show()
        
        
        st.success('The clusters have been generated. Please go ahead and download the clusters, along with the confusion matrices')
    else:
        st.write('Click here to generate the clusters and confusion matrices')
         

    
if __name__ == "__main__":
    
    df_t123 = pd.read_csv('D:/NW_Planning/NWP_v2/data/master_t1_t2_t3.csv')
    df_t11 = pd.read_csv('D:/NW_Planning/NWP_v2/data/Tier1/master_csv_1_tier1_mod1.csv')
    df_t12 = pd.read_csv('D:/NW_Planning/NWP_v2/data/Tier1/master_csv_1_tier1_mod2.csv')
    df_t21 = pd.read_csv('D:/NW_Planning/NWP_v2/data/Tier2/master_csv_1_tier2_mod1.csv')
    df_t22 = pd.read_csv('D:/NW_Planning/NWP_v2/data/Tier2/master_csv_1_tier2_mod2.csv')
    
    df_t1 = pd.concat([df_t11,df_t12])
    df_t2 = pd.concat([df_t21,df_t22])
    
    main()