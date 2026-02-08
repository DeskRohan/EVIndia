import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import numpy as np

# 1. Config & Style
st.set_page_config(page_title="EV India Analysis", layout="wide", page_icon="⚡")

# Custom CSS for green energy theme with light mode
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }
    
    h1 {
        font-weight: 500;
        font-size: 2.8rem;
        margin-bottom: 0.5rem;
        color: #1a472a;
        letter-spacing: -0.02em;
    }
    
    h2 {
        font-weight: 500;
        font-size: 1.6rem;
        margin-top: 2rem;
        color: #2d5f3f;
    }
    
    h3 {
        font-weight: 500;
        font-size: 1.3rem;
        color: #2d5f3f;
    }
    
    .stMetric {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #bbf7d0;
    }
    
    .stMetric label {
        color: #166534 !important;
        font-weight: 500;
        font-size: 0.9rem;
    }
    
    .stMetric [data-testid="stMetricValue"] {
        color: #15803d !important;
        font-weight: 600;
    }
    
    [data-testid="stHorizontalBlock"] {
        gap: 1rem;
    }
    
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background-color: #f9fafb;
        padding: 0.5rem;
        border-radius: 10px;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        color: #4b5563;
        font-weight: 500;
        padding: 0.6rem 1.2rem;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #dcfce7;
        color: #166534;
    }
    
    .stButton button {
        background-color: #16a34a;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.6rem 1.5rem;
        font-weight: 500;
        transition: all 0.2s;
    }
    
    .stButton button:hover {
        background-color: #15803d;
        box-shadow: 0 4px 12px rgba(22, 163, 74, 0.2);
    }
    
    hr {
        border: none;
        border-top: 1px solid #e5e7eb;
        margin: 1.5rem 0;
    }
    
    .stDataFrame {
        border: 1px solid #e5e7eb;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# 2. Data Loading
@st.cache_data
def load_data():
    try:
        sales = pd.read_csv('electric_vehicle_sales_by_state.csv')
    except:
        sales = pd.read_csv('electric_vehicle_sales_by_state.csv', encoding='latin1')
    
    stations = pd.read_csv('charging_stations_india.csv')
    
    sales['date'] = pd.to_datetime(sales['date'], format='%d-%b-%y', errors='coerce')
    sales.dropna(subset=['date'], inplace=True)
    return sales, stations

try:
    sales_df, stations_df = load_data()
except Exception as e:
    st.error(f"Error loading data: {e}")
    st.stop()

# 3. Header
st.title("VoltGrid: EV India")
st.caption("Adoption Trends & Infrastructure Optimization")

# 4. Global Stats
total_evs = sales_df['electric_vehicles_sold'].sum()
total_stations = len(stations_df)
c1, c2, c3 = st.columns(3)
c1.metric("Total EVs Sold", f"{total_evs:,.0f}")
c2.metric("Charging Stations", f"{total_stations:,.0f}")
c3.metric("Data Period", f"{sales_df['date'].min().year} - {sales_df['date'].max().year}")

st.divider()

# 5. Main Tabs
tab_market, tab_infra, tab_opt, tab_forecast = st.tabs(["📈 Market", "🔋 Infrastructure", "🎯 Optimize", "🔮 Forecast"])

# --- TAB 1: MARKET ---
with tab_market:
    st.markdown("### Adoption Trends")
    
    monthly_sales = sales_df.groupby('date')['electric_vehicles_sold'].sum().reset_index()
    
    fig_line = px.line(monthly_sales, x='date', y='electric_vehicles_sold', 
                       template='plotly_white', height=400)
    fig_line.update_layout(
        xaxis_title=None, 
        yaxis_title="Units Sold",
        margin=dict(l=0, r=0, t=20, b=0),
        font=dict(color='#1f2937', size=12),
        xaxis=dict(showgrid=False, linecolor='#e5e7eb'),
        yaxis=dict(gridcolor='#f3f4f6', linecolor='#e5e7eb')
    )
    fig_line.update_traces(line_color='#16a34a', line_width=3)
    st.plotly_chart(fig_line, use_container_width=True)
    
    st.markdown("### Top States")
    
    top_n = 10
    state_sales = sales_df.groupby('state')['electric_vehicles_sold'].sum().sort_values(ascending=False).head(top_n).reset_index()
    
    fig_bar = px.bar(state_sales, x='electric_vehicles_sold', y='state', orientation='h',
                     template='plotly_white', height=400)
    fig_bar.update_layout(
        xaxis_title=None, 
        yaxis_title=None, 
        margin=dict(l=0, r=0, t=10, b=0),
        font=dict(color='#1f2937', size=12),
        xaxis=dict(showgrid=True, gridcolor='#f3f4f6', linecolor='#e5e7eb'),
        yaxis=dict(categoryorder='total ascending', linecolor='#e5e7eb')
    )
    fig_bar.update_traces(marker_color='#22c55e', marker_line_color='#16a34a', marker_line_width=1)
    st.plotly_chart(fig_bar, use_container_width=True)

# --- TAB 2: INFRASTRUCTURE ---
with tab_infra:
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### Station Map")
        
        # Define green-themed color palette for charger types
        color_map = {
            'Type 1': '#22c55e',
            'Type 2': '#16a34a',
            'CCS': '#15803d',
            'CHAdeMO': '#14532d'
        }
        
        fig_map = px.scatter_mapbox(
            stations_df, 
            lat="Latitude", 
            lon="Longitude", 
            color="Charger_Type",
            hover_name="Station_ID", 
            hover_data=["City", "Capacity_KW"],
            zoom=3.5, 
            height=500, 
            mapbox_style="open-street-map",
            color_discrete_map=color_map
        )
        fig_map.update_layout(
            margin=dict(l=0, r=0, t=0, b=0),
            font=dict(color='#1f2937', size=11),
            legend=dict(
                title_text='Charger Type',
                orientation="h",
                yanchor="bottom",
                y=0.02,
                xanchor="left",
                x=0.02,
                bgcolor='rgba(255,255,255,0.9)',
                bordercolor='#e5e7eb',
                borderwidth=1
            )
        )
        st.plotly_chart(fig_map, use_container_width=True)
    
    with col2:
        st.markdown("### Charger Types")
        
        type_counts = stations_df['Charger_Type'].value_counts().reset_index()
        type_counts.columns = ['Type', 'Count']
        
        fig_pie = px.pie(
            type_counts, 
            names='Type', 
            values='Count', 
            hole=0.65, 
            template='plotly_white',
            color_discrete_sequence=['#22c55e', '#16a34a', '#15803d', '#14532d']
        )
        fig_pie.update_layout(
            showlegend=False, 
            margin=dict(t=20, b=20, l=20, r=20),
            font=dict(color='#1f2937', size=11)
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent')
        st.plotly_chart(fig_pie, use_container_width=True)
        
        # Custom metrics
        for i, r in type_counts.iterrows():
            st.caption(f"**{r['Type']}**: {r['Count']}")

# --- TAB 3: OPTIMIZATION ---
with tab_opt:
    st.markdown("### 🎯 Strategic Station Placement")
    
    col_settings, col_map = st.columns([1, 3])
    
    with col_settings:
        st.markdown("#### Configuration")
        
        state_options = ["All India"] + sorted(stations_df['State'].unique())
        selected_region = st.selectbox("Select Region", state_options, index=0)
        
        if selected_region == "All India":
            state_data = stations_df.copy()
            zoom_level = 3.5
        else:
            state_data = stations_df[stations_df['State'] == selected_region].copy()
            zoom_level = 5.5
            
        count = len(state_data)
        st.metric("Analyzed Stations", count)
        
        if count > 5:
            k = st.slider("Proposed Hubs", 2, 15, 5, help="Number of new charging hubs to locate")
            run_btn = st.button("Generate Plan", type="primary", use_container_width=True)
        else:
            st.warning("Insufficient data for clustering.")
            run_btn = False

    with col_map:
        if run_btn and count > 5:
            with st.spinner("Optimizing locations..."):
                X = state_data[['Latitude', 'Longitude']]
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                state_data['Cluster'] = kmeans.fit_predict(X)
                centers = kmeans.cluster_centers_
                
                proposed_hubs = []
                for idx, center in enumerate(centers):
                    ref_data = stations_df if selected_region == "All India" else state_data
                    dists = np.sqrt((ref_data['Latitude'] - center[0])**2 + (ref_data['Longitude'] - center[1])**2)
                    nearest_idx = dists.idxmin()
                    nearest_loc = ref_data.loc[nearest_idx]
                    
                    proposed_hubs.append({
                        "Hub ID": f"HUB-{idx+1:02d}",
                        "Proposed Location": f"Near {nearest_loc['City']}",
                        "State": nearest_loc['State'],
                        "Latitude": round(center[0], 4),
                        "Longitude": round(center[1], 4)
                    })
                
                results_df = pd.DataFrame(proposed_hubs).set_index("Hub ID")

                # Create color palette for clusters
                cluster_colors = ['#bbf7d0', '#86efac', '#4ade80', '#22c55e', '#16a34a', 
                                '#15803d', '#14532d', '#a7f3d0', '#6ee7b7', '#34d399']
                
                fig_opt = px.scatter_mapbox(
                    state_data, 
                    lat="Latitude", 
                    lon="Longitude", 
                    color=state_data['Cluster'].astype(str),
                    hover_name="City", 
                    zoom=zoom_level, 
                    mapbox_style="open-street-map", 
                    opacity=0.4,
                    color_discrete_sequence=cluster_colors
                )
                
                # Add new hub markers
                fig_opt.add_trace(go.Scattermapbox(
                    lat=centers[:, 0], 
                    lon=centers[:, 1], 
                    mode='markers+text',
                    marker=go.scattermapbox.Marker(
                        size=18, 
                        color='#dc2626',
                        symbol='star'
                    ),
                    text=[f"Hub {i+1}" for i in range(k)], 
                    textposition="top right",
                    textfont=dict(size=11, color='#1f2937', family='Inter'),
                    name='New Hubs'
                ))
                
                fig_opt.update_layout(
                    margin=dict(l=0, r=0, t=0, b=0), 
                    showlegend=False, 
                    height=500,
                    font=dict(color='#1f2937')
                )
                st.plotly_chart(fig_opt, use_container_width=True)
                
                st.markdown("#### 📋 Proposed Infrastructure Plan")
                st.dataframe(results_df, use_container_width=True, column_config={
                    "Latitude": st.column_config.NumberColumn(format="%.4f"),
                    "Longitude": st.column_config.NumberColumn(format="%.4f")
                })
        else:
            fig_def = px.scatter_mapbox(
                state_data, 
                lat="Latitude", 
                lon="Longitude", 
                color_discrete_sequence=['#22c55e'],
                zoom=zoom_level, 
                mapbox_style="open-street-map", 
                opacity=0.5
            )
            fig_def.update_layout(
                margin=dict(l=0, r=0, t=0, b=0), 
                height=500,
                font=dict(color='#1f2937')
            )
            st.plotly_chart(fig_def, use_container_width=True)
            if not run_btn:
                st.info("👈 Select settings and click 'Generate Plan' to identify strategic locations.")

# --- TAB 4: FORECAST ---
with tab_forecast:
    st.markdown("### Sales Prediction")
    
    monthly_sales = sales_df.groupby('date')['electric_vehicles_sold'].sum().reset_index()
    monthly_sales['date_ordinal'] = monthly_sales['date'].map(pd.Timestamp.toordinal)
    
    X = monthly_sales[['date_ordinal']]
    y = monthly_sales['electric_vehicles_sold']
    
    model = LinearRegression()
    model.fit(X, y)
    
    last_date = monthly_sales['date'].max()
    future_dates = [last_date + pd.DateOffset(months=i) for i in range(1, 13)]
    future_ordinals = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    future_predictions = model.predict(future_ordinals)
    
    future_df = pd.DataFrame({
        'date': future_dates, 
        'electric_vehicles_sold': future_predictions, 
        'Type': 'Forecast'
    })
    history_df = monthly_sales[['date', 'electric_vehicles_sold']].copy()
    history_df['Type'] = 'Historical'
    combined = pd.concat([history_df, future_df], ignore_index=True)
    
    fig_cast = px.line(
        combined, 
        x='date', 
        y='electric_vehicles_sold', 
        color='Type', 
        template='plotly_white', 
        height=450,
        color_discrete_map={'Historical': '#6b7280', 'Forecast': '#16a34a'}
    )
    
    # Add trendline
    fig_cast.add_scatter(
        x=monthly_sales['date'], 
        y=model.predict(X), 
        mode='lines', 
        name='Trend', 
        line=dict(dash='dot', color='#1f2937', width=2)
    )
    
    fig_cast.update_layout(
        xaxis_title=None, 
        yaxis_title="Sales", 
        legend_title=None, 
        margin=dict(l=0, r=0, t=20, b=0), 
        legend=dict(orientation="h", y=1.1),
        font=dict(color='#1f2937', size=12),
        xaxis=dict(showgrid=False, linecolor='#e5e7eb'),
        yaxis=dict(gridcolor='#f3f4f6', linecolor='#e5e7eb')
    )
    fig_cast.update_traces(line_width=3)
    st.plotly_chart(fig_cast, use_container_width=True)
    
    r2 = model.score(X, y)
    st.caption(f"Model Confidence (R²): {r2:.2f}")

