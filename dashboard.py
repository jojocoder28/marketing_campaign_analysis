import streamlit as st
import pandas as pd
import plotly.express as px

# Load dataset from hardcoded file
data_file = "./marketing_campaign.xlsx"

def load_data(file):
    return pd.read_excel(file)

def main():
    st.set_page_config(page_title="Marketing Dashboard", layout="wide")
    st.title("📊 Marketing Campaign Analysis Dashboard")
    
    df = load_data(data_file)
    
    # Key metrics
    st.subheader("📌 Key Metrics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Customers", df.shape[0])
    col2.metric("Average Income", f"${df['Income'].mean():,.2f}")
    col3.metric("Average Recency", f"{df['Recency'].mean():.2f} days")
    
    st.markdown("---")
    
    # Interactive filtering
    st.sidebar.header("🔍 Filter Options")
    education_levels = df["Education"].unique()
    selected_education = st.sidebar.multiselect("Select Education Level", education_levels, default=education_levels)
    df_filtered = df[df["Education"].isin(selected_education)]
    
    # Show raw data
    if st.expander("📄 Show Raw Data").checkbox("Show Data Table"):
        st.write(df_filtered)
    
    # Visualizations
    st.subheader("📈 Data Visualizations")
    col4, col5 = st.columns(2)
    
    with col4:
        selected_column = st.selectbox("Select Column for Distribution", ["Income", "Recency", "MntWines", "MntMeatProducts", "NumWebPurchases", "NumStorePurchases"])
        fig = px.histogram(df_filtered, x=selected_column, title=f"Distribution of {selected_column}", marginal="box", color_discrete_sequence=['#636EFA'])
        st.plotly_chart(fig, use_container_width=True)
    
    with col5:
        selected_column_2 = st.selectbox("Select Second Column for Scatter Plot", ["Income", "Recency", "MntWines", "MntMeatProducts", "NumWebPurchases", "NumStorePurchases"], index=1)
        fig2 = px.scatter(df_filtered, x=selected_column, y=selected_column_2, title=f"{selected_column} vs {selected_column_2}", trendline="ols", color_discrete_sequence=['#EF553B'])
        st.plotly_chart(fig2, use_container_width=True)
    
    st.markdown("---")
    
    # Customer Spending Overview
    st.subheader("💰 Customer Spending")
    fig3 = px.bar(df_filtered, x="Education", y="MntWines", color="Marital_Status", title="Wine Spending by Education Level", text_auto=True)
    st.plotly_chart(fig3, use_container_width=True)
    
    st.markdown("---")
    
    # Summary statistics
    st.subheader("📋 Summary Statistics")
    st.write(df_filtered.describe())
    
if __name__ == "__main__":
    main()
