import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import plotly.express as px

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="VoltGrid – EV Charging Optimization",
    layout="wide",
    page_icon="⚡"
)

# -------------------------------------------------
# LIGHT GREEN THEME (CSS)
# -------------------------------------------------
st.markdown("""
<style>
.block-container {max-width: 1400px; padding-top: 2rem;}
h1 {color:#14532d; font-weight:600;}
h2,h3 {color:#166534;}
.stMetric {
    background: linear-gradient(135deg,#f0fdf4,#dcfce7);
    padding:1rem;
    border-radius:12px;
    border:1px solid #bbf7d0;
}
.stButton button {
    background:#16a34a;
    color:white;
    border-radius:8px;
    font-weight:500;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------
# LOAD DATA
# -------------------------------------------------
@st.cache_data
def load_data():
    sales = pd.read_csv("electric_vehicle_sales_by_state.csv")
    stations = pd.read_csv("charging_stations_india.csv")
    sales["date"] = pd.to_datetime(sales["date"], format="%d-%b-%y", errors="coerce")
    sales.dropna(subset=["date"], inplace=True)
    return sales, stations

sales_df, stations_df = load_data()

# -------------------------------------------------
# HEADER
# -------------------------------------------------
st.title("⚡ VoltGrid – EV Infrastructure Intelligence")
st.caption("EV Adoption • Charging Infrastructure • AI Optimization")

# -------------------------------------------------
# KPI METRICS
# -------------------------------------------------
c1, c2, c3 = st.columns(3)
c1.metric("Total EVs Sold", f"{sales_df['electric_vehicles_sold'].sum():,.0f}")
c2.metric("Charging Stations", f"{len(stations_df):,}")
c3.metric(
    "Data Period",
    f"{sales_df['date'].min().year} – {sales_df['date'].max().year}"
)

st.divider()

# -------------------------------------------------
# TABS
# -------------------------------------------------
tab_market, tab_infra, tab_opt, tab_forecast = st.tabs(
    ["📈 Market", "🔋 Infrastructure", "🎯 Optimize", "🔮 Forecast"]
)

# =================================================
# TAB 1: MARKET
# =================================================
with tab_market:
    st.subheader("EV Adoption Trends")

    monthly = sales_df.groupby("date")["electric_vehicles_sold"].sum().reset_index()
    fig = px.line(
        monthly,
        x="date",
        y="electric_vehicles_sold",
        template="plotly_white"
    )
    fig.update_traces(line_color="#16a34a", line_width=3)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top EV Adopting States")
    top_states = (
        sales_df.groupby("state")["electric_vehicles_sold"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
    )
    fig2 = px.bar(
        top_states,
        x="electric_vehicles_sold",
        y="state",
        orientation="h",
        template="plotly_white"
    )
    fig2.update_traces(marker_color="#22c55e")
    st.plotly_chart(fig2, use_container_width=True)

# =================================================
# TAB 2: INFRASTRUCTURE (LEAFLET)
# =================================================
with tab_infra:
    st.subheader("Charging Station Distribution (Leaflet Map)")

    m = folium.Map(
        location=[
            stations_df["Latitude"].mean(),
            stations_df["Longitude"].mean()
        ],
        zoom_start=5,
        tiles="CartoDB positron"
    )

    cluster = MarkerCluster().add_to(m)

    for _, r in stations_df.iterrows():
        folium.CircleMarker(
            location=[r["Latitude"], r["Longitude"]],
            radius=3,
            popup=f"""
            <b>{r['Station_ID']}</b><br>
            {r['City']}, {r['State']}<br>
            {r['Charger_Type']} – {r['Capacity_KW']} kW
            """,
            color="#16a34a",
            fill=True,
            fill_opacity=0.6
        ).add_to(cluster)

    st_folium(m, height=520, use_container_width=True)

# =================================================
# TAB 3: OPTIMIZATION (LEAFLET + K-MEANS)
# =================================================
with tab_opt:
    st.subheader("AI-Based Charging Hub Optimization")

    col1, col2 = st.columns([1, 3])

    with col1:
        region = st.selectbox(
            "Select Region",
            ["All India"] + sorted(stations_df["State"].unique())
        )

        data = stations_df if region == "All India" else stations_df[stations_df["State"] == region]
        st.metric("Stations Analyzed", len(data))

        if len(data) > 10:
            k = st.slider("Proposed Charging Hubs", 2, 15, 5)
            run = st.button("Generate Plan", type="primary", use_container_width=True)
        else:
            run = False
            st.warning("Insufficient data.")

    with col2:
        if run:
            X = data[["Latitude", "Longitude"]]
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            data["Cluster"] = kmeans.fit_predict(X)
            centers = kmeans.cluster_centers_

            m2 = folium.Map(
                location=[data["Latitude"].mean(), data["Longitude"].mean()],
                zoom_start=6 if region != "All India" else 5,
                tiles="CartoDB positron"
            )

            cluster2 = MarkerCluster().add_to(m2)

            for _, r in data.iterrows():
                folium.CircleMarker(
                    location=[r["Latitude"], r["Longitude"]],
                    radius=3,
                    color="#86efac",
                    fill=True,
                    fill_opacity=0.4
                ).add_to(cluster2)

            hubs = []
            for i, c in enumerate(centers):
                folium.Marker(
                    location=[c[0], c[1]],
                    icon=folium.Icon(color="red", icon="flash", prefix="fa"),
                    popup=f"Proposed Hub {i+1}"
                ).add_to(m2)

                hubs.append({
                    "Hub": f"HUB-{i+1}",
                    "Latitude": round(c[0], 4),
                    "Longitude": round(c[1], 4)
                })

            st_folium(m2, height=520, use_container_width=True)

            st.subheader("Proposed Charging Hubs")
            st.dataframe(pd.DataFrame(hubs), use_container_width=True)

        else:
            st.info("👈 Select region and generate optimization plan.")

# =================================================
# TAB 4: FORECAST
# =================================================
with tab_forecast:
    st.subheader("EV Sales Forecast")

    monthly = sales_df.groupby("date")["electric_vehicles_sold"].sum().reset_index()
    monthly["ordinal"] = monthly["date"].map(pd.Timestamp.toordinal)

    X = monthly[["ordinal"]]
    y = monthly["electric_vehicles_sold"]

    model = LinearRegression()
    model.fit(X, y)

    future_dates = [monthly["date"].max() + pd.DateOffset(months=i) for i in range(1, 13)]
    future_ord = np.array([d.toordinal() for d in future_dates]).reshape(-1, 1)
    preds = model.predict(future_ord)

    forecast = pd.DataFrame({
        "date": future_dates,
        "electric_vehicles_sold": preds,
        "Type": "Forecast"
    })

    hist = monthly[["date", "electric_vehicles_sold"]]
    hist["Type"] = "Historical"

    combined = pd.concat([hist, forecast])

    fig = px.line(
        combined,
        x="date",
        y="electric_vehicles_sold",
        color="Type",
        template="plotly_white",
        color_discrete_map={
            "Historical": "#6b7280",
            "Forecast": "#16a34a"
        }
    )
    fig.update_traces(line_width=3)
    st.plotly_chart(fig, use_container_width=True)

    st.caption(f"Model Confidence (R²): {model.score(X, y):.2f}")
