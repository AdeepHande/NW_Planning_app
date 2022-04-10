### AI Based Network Expansion Planning App

The app is structured as follows:
* Page 1: Landing Page
  
      User has to select among
            * Duplicate Checker (Page 2)
            * Uploading external data (Page 3)
            * Clustering App (Page 4)
* Page 2: Duplicate Checker (In dev)

      Input: dataframe df to remove duplicates (Coordinates are necessary)
      Output: Returns dataframe df after checking for duplicates.
* Page 3: Uploading external data 

      Input: 
            A) Dataframe df (from Duplicate Checker)
            B) Wireline Point of Presence ( Wireline PoP)
            C) Wireless Point of Presence (Wireless PoP)
            D) Handhole Point of Presence ( Handhole PoP)
      Default data: 
            * Internal data (Master data)
      
      Output:
            * Spatially Join (Master Data, df) 🠆  df_dash
* Page 4: Clustering App
      
      Input data:
            * df_dash 
            * Configurable Parameters (set by the user)
      Returns:
            * Output.zip
                  |__ Final_clustered_data.csv
                  |__ Merged_final_data.csv  (categorized clustered data)
                  |__ Bandwidth_potential_per_cluster.csv
                  |__ plots.html 
                  |__ plots.kml (In dev)

### Dev

1. Create an environment ```conda create -n streamlit-template python=3.8.5 pip```.

2. Install requirements: ```pip install -r app/requirements.txt```

#### Run the App
* ```cd app```
* ```streamlit run multipage.py```
