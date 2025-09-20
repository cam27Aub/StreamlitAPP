
# streamlit_app.py
# Ready-to-run Streamlit template with two Plotly visualizations and impactful interactivity
from pandas.core.arrays import numpy_


import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# ------------------------------
# Page config
# ------------------------------
st.set_page_config(
    page_title="Interactive Visualizations Demo",
    page_icon="📊",
    layout="wide"
)

# ------------------------------
# Header
# ------------------------------
st.title("Building Interactive Visualizations with Streamlit")
st.caption("A polished example app you can adapt to your dataset. Includes two related charts, multiple interactivity controls, and explanatory context.")

# ------------------------------
# Data
# ------------------------------
# Using Plotly's built-in Gapminder dataset to avoid external downloads
@st.cache_data
def load_data():
    df = px.data.gapminder()
    # Precompute a few helpers
    df["gdp_billions"] = (df["gdpPercap"] * df["pop"]) / 1e9
    return df

df = load_data()

# ------------------------------
# Sidebar controls
# ------------------------------
st.sidebar.header("Controls")

years = sorted(df["year"].unique().tolist())
min_year, max_year = min(years), max(years)

selected_year = st.sidebar.slider("Year", min_value=min_year, max_value=max_year, value=2007, step=5)

continents = sorted(df["continent"].unique().tolist())
selected_continents = st.sidebar.multiselect("Continents", continents, default=continents)

size_by = st.sidebar.selectbox("Bubble size", ["pop", "gdp_billions"], format_func=lambda x: "Population" if x=="pop" else "GDP (Billions)")

use_log_x = st.sidebar.checkbox("Log scale for GDP per capita", value=True)

search_countries = st.sidebar.text_input("Highlight countries (comma separated)")

# Filter by year and continents
mask = (df["year"] == selected_year) & (df["continent"].isin(selected_continents))
filtered = df.loc[mask].copy()

# Highlight selection
highlights = [c.strip() for c in search_countries.split(",") if c.strip()]
filtered["highlight"] = np.where(filtered["country"].isin(highlights), "Highlighted", "Other")

# ------------------------------
# KPI summary row
# ------------------------------
col_a, col_b, col_c, col_d = st.columns(4)
with col_a:
    st.metric("Countries", int(filtered["country"].nunique()))
with col_b:
    st.metric("Total population", f"{filtered['pop'].sum():,}")
with col_c:
    st.metric("Avg life expectancy", f"{filtered['lifeExp'].mean():.1f} years")
with col_d:
    st.metric("Total GDP", f"{filtered['gdp_billions'].sum():.1f} B")

st.markdown("""
### What this page shows
We are exploring how economic growth and demographics relate to **life expectancy** in the selected year and continents.
Use the controls to change the **year**, **continents**, axis scale, and to **highlight specific countries**.
""")

# ------------------------------
# Chart 1: Scatter plot
# ------------------------------
left, right = st.columns([1.2, 1])

with left:
    st.subheader("Life Expectancy vs GDP per Capita")
    st.caption("Each bubble is a country. Size reflects Population or GDP. Color indicates continent. Use the sidebar to filter and tweak scale.")

    fig_scatter = px.scatter(
        filtered,
        x="gdpPercap",
        y="lifeExp",
        size=size_by,
        color="continent",
        hover_name="country",
        size_max=60,
        labels={
            "gdpPercap": "GDP per capita",
            "lifeExp": "Life expectancy",
        },
        custom_data=["country", "pop", "gdp_billions"]
    )

    # Emphasize highlighted countries
    if highlights:
        fig_scatter.update_traces(
            marker=dict(line=dict(width=np.where(filtered["highlight"].eq("Highlighted"), 2.5, 0))),
            selector=dict(mode="markers")
        )

    if use_log_x:
        fig_scatter.update_xaxes(type="log")

    fig_scatter.update_layout(margin=dict(t=10, b=0, l=0, r=0))
    st.plotly_chart(fig_scatter, use_container_width=True)

with right:
    st.subheader("Distribution of Life Expectancy")
    st.caption("Adjust year or continents to see how the distribution shifts.")
    fig_hist = px.histogram(
        filtered,
        x="lifeExp",
        nbins=20,
        color="continent",
        barmode="overlay",
        opacity=0.75,
        labels={"lifeExp": "Life expectancy"}
    )
    fig_hist.update_layout(margin=dict(t=10, b=0, l=0, r=0))
    st.plotly_chart(fig_hist, use_container_width=True)

# ------------------------------
# Chart 2: Choropleth map
# ------------------------------
st.subheader("Global Life Expectancy Map")
st.caption("A choropleth view helps spot regional patterns and outliers for the chosen year.")

fig_map = px.choropleth(
    filtered,
    locations="iso_alpha",
    color="lifeExp",
    hover_name="country",
    color_continuous_scale="Viridis",
    labels={"lifeExp": "Life expectancy"},
)
fig_map.update_layout(margin=dict(t=0, b=0, l=0, r=0))
st.plotly_chart(fig_map, use_container_width=True)

# ------------------------------
# Insights and notes
# ------------------------------
st.markdown(
    f"""
#### Insights
In {selected_year}, countries with higher GDP per capita generally cluster at higher life expectancy values, though there are notable exceptions.
Smaller populations can still achieve strong outcomes while some populous countries show wide variance across continents.
Use the histogram to examine dispersion and the map to identify regional pockets above or below the global median.
"""
)

# ------------------------------
# Data download + how to adapt
# ------------------------------
st.download_button(
    label="Download filtered data as CSV",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name=f"gapminder_{selected_year}.csv",
    mime="text/csv",
)

with st.expander("How to adapt this to your dataset"):
    st.markdown(
        """
1. Replace the `load_data()` function with your own data source. You can read a CSV with `pd.read_csv('yourfile.csv')` or query a database.
2. Update the sidebar controls to match the fields in your data.
3. Adjust the `px.scatter`, `px.histogram`, and `px.choropleth` calls to use your column names.
4. Keep at least two **distinct** interactive features that change the visuals meaningfully.
5. Write 3 to 5 sentences of context and insights below the charts, explaining what users should notice.
        """
    )

st.success("Template ready. Tweak columns and text then deploy to Streamlit Community Cloud.")