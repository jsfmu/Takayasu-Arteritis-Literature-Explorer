import pandas as pd
import streamlit as st
import plotly.express as px
from pathlib import Path

DATA_PATH = Path("data/processed/takayasu_annotated_clustered.csv")

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    return df.dropna(subset=["year"])

def main():
    st.set_page_config(page_title="Takayasu Arteritis Literature Explorer", layout="wide")
    st.title("Takayasu Arteritis Literature Explorer")

    df = load_data()

    # Sidebar filters
    st.sidebar.header("Filters")

    min_year, max_year = int(df["year"].min()), int(df["year"].max())
    year_range = st.sidebar.slider("Publication year range", min_year, max_year, (min_year, max_year))
    mask_year = df["year"].between(year_range[0], year_range[1])

    imaging_options = ["mentions_ct", "mentions_mri", "mentions_pet", "mentions_ultrasound"]
    selected_imaging = st.sidebar.multiselect("Imaging modalities", imaging_options)

    treatment_options = ["mentions_steroids", "mentions_biologics", "mentions_surgery"]
    selected_treatments = st.sidebar.multiselect("Treatments", treatment_options)

    # keyword search
    keyword = st.sidebar.text_input("Search in title/abstract", "")

    filtered = df[mask_year].copy()

    for col in selected_imaging:
        filtered = filtered[filtered[col] == True]
    for col in selected_treatments:
        filtered = filtered[filtered[col] == True]

    if keyword:
        kw = keyword.lower()
        filtered = filtered[
            filtered["title"].fillna("").str.lower().str.contains(kw)
            | filtered["abstract"].fillna("").str.lower().str.contains(kw)
        ]

    st.subheader("Summary")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total papers (all)", len(df))
    col2.metric("Papers (filtered)", len(filtered))
    col3.metric("Year span", f"{min_year}–{max_year}")

    # Time series
    st.subheader("Papers per year")
    ts = filtered.groupby("year").size().reset_index(name="count")
    fig_ts = px.line(ts, x="year", y="count", markers=True)
    st.plotly_chart(fig_ts, use_container_width=True)

    # Imaging bar chart
    st.subheader("Imaging modality mentions (filtered set)")
    imaging_counts = {
        col: int(filtered[col].sum())
        for col in imaging_options
    }
    df_imaging = pd.DataFrame({
        "modality": list(imaging_counts.keys()),
        "count": list(imaging_counts.values())
    })
    fig_img = px.bar(df_imaging, x="modality", y="count")
    st.plotly_chart(fig_img, use_container_width=True)

    # Country / affiliation
    st.subheader("Top affiliations (rough proxy for country)")
    top_aff = (
        filtered["affiliations"]
        .fillna("Unknown")
        .value_counts()
        .head(15)
        .reset_index()
    )
    top_aff.columns = ["affiliation", "count"]
    fig_aff = px.bar(top_aff, x="affiliation", y="count")
    st.plotly_chart(fig_aff, use_container_width=True)

    # Table of papers
    st.subheader("Filtered papers")
    st.dataframe(
        filtered[[
            "year", "journal", "title", "mentions_ct", "mentions_mri",
            "mentions_pet", "mentions_steroids", "mentions_biologics", "lda_topic", "kmeans_cluster"
        ]],
        use_container_width=True,
        height=400
    )

    # Optional: show abstract for selected paper
    st.subheader("Inspect abstract")
    options = filtered["title"].tolist()
    if options:
        selected_title = st.selectbox("Select a paper", options)
        row = filtered[filtered["title"] == selected_title].iloc[0]
        st.write(f"**{row['title']}**")
        st.write(f"*{row.get('journal', '')}, {int(row['year'])}*")
        st.write(row.get("abstract", ""))

if __name__ == "__main__":
    main()
