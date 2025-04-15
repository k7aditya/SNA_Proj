import pandas as pd
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import folium_static
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import time
import matplotlib.dates as mdates
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from shapely.geometry import Point
from scipy.spatial import cKDTree
from haversine import haversine, Unit
from sklearn.preprocessing import StandardScaler
import streamlit as st
from branca.element import Template, MacroElement
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from folium.plugins import MarkerCluster, HeatMap
from datetime import timedelta
from collections import Counter

# --- Load and Preprocess ---
st.title("🚨 Hoax Call Analysis Dashboard")

uploaded_file = st.file_uploader("Upload LAPD Calls-for-Service CSV", type=["csv"])
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    st.warning("Please upload the LAPD Calls-for-Service CSV file.")
    st.stop()

# Use only 0.01% of data for memory efficiency
df = df.sample(frac=0.0001, random_state=42).reset_index(drop=True)
st.metric("Rows after sampling", len(df))
# Parse datetime
df['Dispatch Date'] = pd.to_datetime(df['Dispatch Date'], errors='coerce')
df['Dispatch Time'] = pd.to_timedelta(df['Dispatch Time'], errors='coerce')
df['Datetime'] = df['Dispatch Date'] + df['Dispatch Time']
df.fillna({'Call Type Code': 'UNKNOWN', 'Call Type Description': 'UNKNOWN'}, inplace=True)

# --- Hoax Call Identification Rules ---
st.subheader("🔍 Hoax Call Identification")

# Create rules UI with expandable section
with st.expander("Hoax Call Identification Rules"):
    st.markdown("""
    We'll identify potential hoax calls based on the following rules:
    
    1. *Known Hoax Call Types*: Certain call codes are more likely to be hoaxes
    2. *Rapid Succession Rule*: Multiple calls from the same district in short time period
    3. *Pattern Matching Rule*: Similar call types in different districts within short time
    4. *No Response Needed Rule*: Calls that were closed without dispatching units
    5. *Call Frequency Rule*: Districts with abnormally high call volume in short period
    """)

    # Allow users to configure rules
    st.subheader("Configure Hoax Detection Rules")
    col1, col2 = st.columns(2)
    
    with col1:
        hoax_call_types = st.multiselect(
            "Suspicious Call Types",
            options=['900', '906B1', 'CODE 6', 'CODE 30 RINGER', 'UNKNOWN TROUBLE'],
            default=['900', '906B1', 'CODE 30 RINGER']
        )
        time_window_minutes = st.slider("Time Window for Pattern Detection (minutes)", 5, 120, 30)
    
    with col2:
        distance_threshold = st.slider("Geographic Distance Threshold (km)", 1.0, 20.0, 5.0)
        call_frequency_threshold = st.slider("Call Frequency Threshold (calls per hour)", 2, 10, 3)

# Geocode subset
geolocator = Nominatim(user_agent="hoax-call-analysis")
df_geo_subset = df.sample(frac=0.1, random_state=42)
real_coords = {}

with st.spinner("Geocoding locations..."):
    for _, row in df_geo_subset.iterrows():
        location_query = f"Los Angeles, CA {row['Address']}" if 'Address' in df.columns else None
        if location_query:
            try:
                location = geolocator.geocode(location_query, timeout=10)
                if location:
                    real_coords[row['Reporting District']] = (location.latitude, location.longitude)
                # time.sleep(1)
            except Exception:
                pass

# Assign Coordinates
df['Latitude'] = df['Reporting District'].map(lambda x: real_coords.get(x, (np.nan, np.nan))[0])
df['Longitude'] = df['Reporting District'].map(lambda x: real_coords.get(x, (np.nan, np.nan))[1])

# Fill missing with synthetic
unique_districts = df['Reporting District'].dropna().unique()
mock_coords = {
    district: (np.random.uniform(33.776979, 34.241551), np.random.uniform(-118.378099, -117.806759))
    for district in unique_districts if district not in real_coords
}

df['Latitude'] = df.apply(lambda row: mock_coords.get(row['Reporting District'], (row['Latitude'], row['Longitude']))[0] if np.isnan(row['Latitude']) else row['Latitude'], axis=1)
df['Longitude'] = df.apply(lambda row: mock_coords.get(row['Reporting District'], (row['Latitude'], row['Longitude']))[1] if np.isnan(row['Longitude']) else row['Longitude'], axis=1)

# GeoDataFrame
gdf = gpd.GeoDataFrame(df, geometry=gpd.points_from_xy(df['Longitude'], df['Latitude']), crs="EPSG:4326")
gdf[['Latitude', 'Longitude']] = gdf[['Latitude', 'Longitude']].bfill().ffill()

# --- HOAX CALL IDENTIFICATION FUNCTION ---
def identify_hoax_calls(gdf, hoax_call_types, time_window_minutes, distance_threshold, call_frequency_threshold):
    """
    Identify potential hoax calls based on multiple rules
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        The calls data
    hoax_call_types : list
        List of call types likely to be hoaxes
    time_window_minutes : int
        Time window in minutes for pattern detection
    distance_threshold : float
        Distance threshold in km for spatial patterns
    call_frequency_threshold : int
        Threshold for abnormal call frequency
        
    Returns:
    --------
    GeoDataFrame with hoax probability scores
    """
    # Create a copy to avoid modifying original
    gdf_hoax = gdf.copy()
    
    # Initialize hoax probability score column (0-1 scale)
    gdf_hoax['hoax_probability'] = 0.0
    
    # --- Rule 1: Known Hoax Call Types ---
    def get_call_type_score(call_type):
        if call_type in hoax_call_types:
            return 0.7
        # Add partial matches
        elif 'CODE' in str(call_type) or 'UNKNOWN' in str(call_type):
            return 0.4
        return 0.1
    
    gdf_hoax['rule1_score'] = gdf_hoax['Call Type Description'].apply(get_call_type_score)
    
    # --- Rule 2: Rapid Succession Rule ---
    # Sort by district and time
    gdf_sorted = gdf_hoax.sort_values(['Reporting District', 'Datetime'])
    
    # Calculate time difference between calls in same district
    gdf_sorted['prev_time'] = gdf_sorted.groupby('Reporting District')['Datetime'].shift(1)
    gdf_sorted['time_diff_minutes'] = (gdf_sorted['Datetime'] - gdf_sorted['prev_time']).dt.total_seconds() / 60
    
    # Score based on rapid succession within same district
    def get_rapid_succession_score(time_diff):
        if pd.isna(time_diff):
            return 0.0
        elif time_diff < 15:  # Very suspicious if less than 15 minutes
            return 0.8
        elif time_diff < 60:  # Somewhat suspicious if less than 1 hour
            return 0.4
        return 0.0
    
    gdf_sorted['rule2_score'] = gdf_sorted['time_diff_minutes'].apply(get_rapid_succession_score)
    
    # --- Rule 3: Pattern Matching Rule ---
    # Look for similar calls across districts within time window
    def check_pattern_match(row, gdf, time_window):
        call_time = row['Datetime']
        call_district = row['Reporting District']
        call_type = row['Call Type Description']
        
        # Find calls with same type in different districts within time window
        time_window_calls = gdf[
            (gdf['Datetime'] >= call_time - pd.Timedelta(minutes=time_window)) &
            (gdf['Datetime'] <= call_time + pd.Timedelta(minutes=time_window)) &
            (gdf['Reporting District'] != call_district) &
            (gdf['Call Type Description'] == call_type)
        ]
        
        if len(time_window_calls) >= 2:
            return 0.9  # Very likely hoax if similar calls in different districts
        elif len(time_window_calls) == 1:
            return 0.5
        return 0.0
    
    # Apply pattern matching but only to a subset for efficiency
    pattern_sample = gdf_sorted.sample(min(len(gdf_sorted), 100), random_state=42)
    pattern_scores = {}
    
    for idx, row in pattern_sample.iterrows():
        pattern_scores[idx] = check_pattern_match(row, gdf_sorted, time_window_minutes)
    
    # Apply scores to all rows (approximation)
    gdf_sorted['rule3_score'] = gdf_sorted.index.map(lambda x: pattern_scores.get(x, 0.0))
    
    # --- Rule 4: Call Frequency Rule ---
    # Identify districts with abnormally high call frequency
    district_freq = gdf_sorted.groupby('Reporting District').size()
    avg_district_calls = district_freq.mean()
    high_freq_districts = district_freq[district_freq > call_frequency_threshold * avg_district_calls].index.tolist()
    
    gdf_sorted['rule4_score'] = gdf_sorted['Reporting District'].apply(
        lambda x: 0.6 if x in high_freq_districts else 0.0
    )
    
    # --- Calculate Final Hoax Probability ---
    # Weighted combination of all rules
    gdf_sorted['hoax_probability'] = (
        0.3 * gdf_sorted['rule1_score'] +
        0.3 * gdf_sorted['rule2_score'] +
        0.3 * gdf_sorted['rule3_score'] +
        0.1 * gdf_sorted['rule4_score']
    )
    
    # Classify calls as likely hoax, suspicious, or normal
    gdf_sorted['hoax_classification'] = pd.cut(
        gdf_sorted['hoax_probability'],
        bins=[0, 0.3, 0.6, 1.0],
        labels=['Normal', 'Suspicious', 'Likely Hoax']
    )
    
    return gdf_sorted

# Run the hoax identification algorithm
gdf_with_hoax = identify_hoax_calls(
    gdf, 
    hoax_call_types, 
    time_window_minutes, 
    distance_threshold, 
    call_frequency_threshold
)

# Display summary
st.subheader("📊 Hoax Call Analysis Summary")
hoax_counts = gdf_with_hoax['hoax_classification'].value_counts()

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Total Calls", len(gdf_with_hoax))
with col2:
    st.metric("Suspicious Calls", 
             int(hoax_counts.get('Suspicious', 0)), 
             f"{100 * hoax_counts.get('Suspicious', 0) / len(gdf_with_hoax):.1f}%")
with col3:
    st.metric("Likely Hoax Calls", 
             int(hoax_counts.get('Likely Hoax', 0)), 
             f"{100 * hoax_counts.get('Likely Hoax', 0) / len(gdf_with_hoax):.1f}%")

# --- Hoax Call Map ---
st.subheader("🗺️ Hoax Call Map")

# Create base map
m = folium.Map(location=[33.954, -118.2], zoom_start=11, tiles="CartoDB positron")

# Add custom legend
legend_html = """
<div style="position: fixed; 
     bottom: 50px; right: 50px; width: 180px; height: 160px; 
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white; padding: 10px;
     ">
     <b>Hoax Classification</b><br>
     <i class="fa fa-circle" style="color:red"></i> Likely Hoax<br>
     <i class="fa fa-circle" style="color:orange"></i> Suspicious<br>
     <i class="fa fa-circle" style="color:green"></i> Normal<br>
     <hr>
     Size indicates hoax probability
</div>
"""
m.get_root().html.add_child(folium.Element(legend_html))

# Add marker cluster group for improved performance
marker_cluster = MarkerCluster().add_to(m)

# Add points to map
for idx, row in gdf_with_hoax.iterrows():
    # Determine color based on classification
    if row['hoax_classification'] == 'Likely Hoax':
        color = 'red'
    elif row['hoax_classification'] == 'Suspicious':
        color = 'orange'
    else:
        color = 'green'
    
    # Determine size based on probability
    size = 4 + (row['hoax_probability'] * 8)
    
    # Create popup content
    popup_content = f"""
    <b>Incident:</b> {row['Incident Number']}<br>
    <b>District:</b> {row['Reporting District']}<br>
    <b>Area:</b> {row['Area Occurred']}<br>
    <b>Call Type:</b> {row['Call Type Description']}<br>
    <b>Time:</b> {row['Datetime']}<br>
    <b>Hoax Probability:</b> {row['hoax_probability']:.2f}<br>
    <b>Classification:</b> {row['hoax_classification']}
    """
    
    # Add circle marker
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=size,
        color=color,
        fill=True,
        fill_opacity=0.7,
        popup=folium.Popup(popup_content, max_width=300)
    ).add_to(marker_cluster)

# Add heat map layer for hoax probability
heat_data = [
    [row['Latitude'], row['Longitude'], float(row['hoax_probability'])] 
    for _, row in gdf_with_hoax.iterrows()
]

heatmap_layer = HeatMap(
    heat_data,
    name="Hoax Probability Heatmap",
    radius=15,
    blur=10,
    gradient={"0.2": 'blue', "0.5": 'lime', "0.8": 'orange', "1": 'red'}  # Convert float keys to strings
)

# Add heatmap to a separate layer group
heatmap_group = folium.FeatureGroup(name="Heatmap View")
heatmap_layer.add_to(heatmap_group)
heatmap_group.add_to(m)

# Add layer control
folium.LayerControl().add_to(m)

# Display map
folium_static(m)

# --- Hoax Call Details ---
st.subheader("📋 Hoax Call Details")

filter_type = st.selectbox(
    "Filter by classification",
    options=["All Calls", "Likely Hoax", "Suspicious", "Normal"]
)

if filter_type == "All Calls":
    filtered_calls = gdf_with_hoax
else:
    filtered_calls = gdf_with_hoax[gdf_with_hoax['hoax_classification'] == filter_type]

st.dataframe(
    filtered_calls[['Incident Number', 'Reporting District', 'Area Occurred', 
                   'Datetime', 'Call Type Description', 'hoax_probability', 
                   'hoax_classification']].sort_values('hoax_probability', ascending=False)
)

# --- Clustering ---
coords = gdf[['Latitude', 'Longitude']].values
scaler = StandardScaler()
scaled_coords = scaler.fit_transform(coords)

kmeans = KMeans(n_clusters=5, random_state=42)
gdf['Cluster_KMeans'] = kmeans.fit_predict(coords)

dbscan = DBSCAN(eps=0.005, min_samples=3)
gdf['Cluster_DBSCAN'] = dbscan.fit_predict(scaled_coords)

spectral = SpectralClustering(n_clusters=5, affinity='nearest_neighbors', assign_labels='kmeans', random_state=42)
gdf['Cluster_Spectral'] = spectral.fit_predict(scaled_coords)

# --- Clustering Visualization ---
st.subheader("📍 Cluster Maps")
cluster_algos = ['Cluster_KMeans', 'Cluster_DBSCAN', 'Cluster_Spectral']
cluster_colors = ['red', 'blue', 'green', 'purple', 'orange', 'gray', 'black']

algo_choice = st.selectbox("Select Clustering Algorithm", cluster_algos)
m = folium.Map(location=[33.780882000869195, -118.380483], zoom_start=10)

for _, row in gdf.iterrows():
    cluster_val = int(row[algo_choice])
    color = cluster_colors[cluster_val % len(cluster_colors)] if cluster_val != -1 else 'black'
    folium.CircleMarker(
        location=[row['Latitude'], row['Longitude']],
        radius=5,
        color=color,
        fill=True,
        popup=f"{algo_choice} | {row['Call Type Code']}"
    ).add_to(m)

folium_static(m)

# --- Graph Construction ---
threshold_km = 10
G = nx.Graph()

for idx, row in gdf.iterrows():
    G.add_node(row['Reporting District'], lat=row['Latitude'], lon=row['Longitude'])

coords = np.array(list(zip(gdf['Latitude'], gdf['Longitude'])))
tree = cKDTree(coords)
pairs = tree.query_pairs(threshold_km / 111)

for i, j in pairs:
    d1 = gdf.iloc[i]['Reporting District']
    d2 = gdf.iloc[j]['Reporting District']
    distance = geodesic(
        (gdf.iloc[i]['Latitude'], gdf.iloc[i]['Longitude']),
        (gdf.iloc[j]['Latitude'], gdf.iloc[j]['Longitude'])
    ).kilometers
    if distance < threshold_km:
        G.add_edge(d1, d2, weight=distance)

# --- Propagation Analysis ---
gdf_sorted = gdf.sort_values('Datetime')
gdf_sorted['Time_diff'] = gdf_sorted['Datetime'].diff().dt.total_seconds().fillna(0)

def detect_propagation_chains(gdf_sorted, time_threshold=120, distance_threshold_km=10):
    chains = []
    current_chain = [gdf_sorted.iloc[0]]

    for i in range(1, len(gdf_sorted)):
        t_diff = (gdf_sorted.iloc[i]['Datetime'] - gdf_sorted.iloc[i-1]['Datetime']).total_seconds() / 60
        d_diff = geodesic(
            (gdf_sorted.iloc[i]['Latitude'], gdf_sorted.iloc[i]['Longitude']),
            (gdf_sorted.iloc[i-1]['Latitude'], gdf_sorted.iloc[i-1]['Longitude'])
        ).kilometers

        if t_diff <= time_threshold and d_diff <= distance_threshold_km:
            current_chain.append(gdf_sorted.iloc[i])
        else:
            chains.append(current_chain)
            current_chain = [gdf_sorted.iloc[i]]

    if current_chain:
        chains.append(current_chain)

    return chains

chains = detect_propagation_chains(gdf_sorted)

st.subheader("📊 Propagation Chain Summary")
st.markdown(f"*Detected {len(chains)} propagation chains.*")

for idx, chain in enumerate(chains[:10]):
    st.markdown(f"### Chain {idx+1}:")
    for call in chain:
        st.write(f"- {call['Datetime']} | District: {call['Reporting District']} | Call Type: {call['Call Type Code']}")

# --- Time Analysis --- #
st.subheader("⏱ Call Time Analysis")

fig1, ax1 = plt.subplots(figsize=(10, 5))
ax1.hist(gdf_sorted['Time_diff'] / 60, bins=50, color='skyblue', edgecolor='black')
ax1.set_xlabel("Time Difference Between Calls (minutes)")
ax1.set_ylabel("Frequency")
ax1.set_title("Distribution of Time Differences Between Hoax Calls")
ax1.grid(True)
st.pyplot(fig1)

# Dropdown for time resolution
time_view = st.selectbox("Select Time Resolution", options=["Daily", "Hourly (Sample of 20)"])

if time_view == "Daily":
    # Calls per day
    calls_per_day = gdf_sorted.set_index('Datetime').resample('D').size()
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    calls_per_day.plot(ax=ax2, kind='line', marker='o', color='red')
    ax2.set_title('Hoax Call Frequency Over Time (Daily)')
    ax2.set_xlabel('Date')
    ax2.set_ylabel('Number of Calls')
    ax2.grid(True)
    ax2.set_xticklabels(ax2.get_xticklabels(), rotation=45)
    st.pyplot(fig2)

else:
    # Hourly distribution from a sample
    import matplotlib.dates as mdates

    gdf_sample = gdf_sorted.sample(n=20, random_state=42)
    if 'Datetime' in gdf_sample.columns:
        gdf_sample = gdf_sample.sort_values('Datetime').set_index('Datetime')
    else:
        gdf_sample = gdf_sample.sort_index()

    calls_per_hour = gdf_sample.resample('h').size()

    fig_hourly, ax_hourly = plt.subplots(figsize=(12, 6))
    ax_hourly.plot(calls_per_hour.index, calls_per_hour.values, marker='o', linestyle='-', color='red')
    ax_hourly.set_title("Hoax Call Frequency Over Time (Sample of 20 Calls - Hourly)")
    ax_hourly.set_xlabel("Time")
    ax_hourly.set_ylabel("Number of Calls")
    ax_hourly.grid(True)

    ax_hourly.set_facecolor("lightgray")
    ax_hourly.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
    ax_hourly.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d\n%H:%M"))
    plt.xticks(rotation=45, fontsize=8)
    plt.tight_layout()
    st.pyplot(fig_hourly)

# --- Correlation ---
st.subheader("📌 Cross-Region Correlation")
pivot = pd.crosstab(gdf['Reporting District'], gdf['Reporting District'])
corr = pivot.T.corr()
fig3, ax3 = plt.subplots(figsize=(12, 8))
sns.heatmap(corr, cmap='coolwarm', ax=ax3)
ax3.set_title("Cross-Region Correlation Based on Call Type Patterns")
st.pyplot(fig3)

# --- Graph ---
st.subheader("📡 Distance-Based Relationship Graph")
fig4, ax4 = plt.subplots(figsize=(12, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=500, node_color='lightblue', edge_color='gray', ax=ax4)
ax4.set_title("Distance-based Relationship Graph")
st.pyplot(fig4)

# --- Clustering --- #
coords = gdf[['Latitude', 'Longitude']].values
scaled_coords = StandardScaler().fit_transform(coords)

kmeans = KMeans(n_clusters=3, random_state=42).fit(coords)
dbscan = DBSCAN(eps=0.15, min_samples=5).fit(scaled_coords)
spectral = SpectralClustering(n_clusters=3, assign_labels="discretize", random_state=42).fit(scaled_coords)

gdf['Cluster_KMeans'] = kmeans.labels_
gdf['Cluster_DBSCAN'] = dbscan.labels_
gdf['Cluster_Spectral'] = spectral.labels_

# --- Cluster Evaluation --- #
st.subheader("📊 Clustering Evaluation")
st.text("Silhouette Score:")
st.text(f"KMeans: {silhouette_score(coords, gdf['Cluster_KMeans']):.3f}")
st.text(f"DBSCAN: {silhouette_score(scaled_coords, gdf['Cluster_DBSCAN']) if len(set(gdf['Cluster_DBSCAN'])) > 1 else 'N/A'}")
st.text(f"Spectral: {silhouette_score(scaled_coords, gdf['Cluster_Spectral']):.3f}")

st.text("\nCalinski-Harabasz Index:")
st.text(f"KMeans: {calinski_harabasz_score(coords, gdf['Cluster_KMeans']):.2f}")
st.text(f"DBSCAN: {calinski_harabasz_score(scaled_coords, gdf['Cluster_DBSCAN']) if len(set(gdf['Cluster_DBSCAN'])) > 1 else 'N/A'}")
st.text(f"Spectral: {calinski_harabasz_score(scaled_coords, gdf['Cluster_Spectral']):.2f}")

st.text("\nDavies-Bouldin Index:")
st.text(f"KMeans: {davies_bouldin_score(coords, gdf['Cluster_KMeans']):.3f}")
st.text(f"DBSCAN: {davies_bouldin_score(scaled_coords, gdf['Cluster_DBSCAN']) if len(set(gdf['Cluster_DBSCAN'])) > 1 else 'N/A'}")
st.text(f"Spectral: {davies_bouldin_score(scaled_coords, gdf['Cluster_Spectral']):.3f}")

# --- Centrality Measures --- #
st.subheader("🔗 Node Centrality Measures")

# Compute centralities
betweenness = nx.betweenness_centrality(G)
closeness = nx.closeness_centrality(G)
try:
    eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
except nx.PowerIterationFailedConvergence:
    eigenvector = {k: 0 for k in G.nodes()}

centrality_options = {
    'Betweenness Centrality': betweenness,
    'Closeness Centrality': closeness,
    'Eigenvector Centrality': eigenvector
}

selected_metric = st.selectbox("Select Centrality Metric", list(centrality_options.keys()))
selected_dict = centrality_options[selected_metric]

# Top N nodes
sorted_centrality = sorted(selected_dict.items(), key=lambda item: item[1], reverse=True)
top_n = st.slider("Number of top nodes to show", 5, 20, 10)

st.markdown(f"### Top {top_n} Nodes by {selected_metric}")
for node, val in sorted_centrality[:top_n]:
    st.write(f"District {node}: {val:.4f}")
