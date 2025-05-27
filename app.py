import streamlit as st
import pandas as pd
import numpy as np
try:
    import matplotlib.pyplot as plt
except ImportError:
    import sys
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "matplotlib"])
    import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO

# Set page config
st.set_page_config(
    page_title="Sanjai Chips Sales Prediction",
    page_icon="🥔",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 5px;
        border: none;
        padding: 10px 24px;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .header-text {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .subheader-text {
        font-size: 1.2rem;
        color: #7f8c8d;
        margin-bottom: 2rem;
    }
    .result-card {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
    }
    .plot-container {
        background-color: white;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.markdown('<p class="header-text">Sanjai Chips Sales Prediction</p>', unsafe_allow_html=True)
st.markdown('<p class="subheader-text">Using Monte Carlo Simulation to forecast future sales of Sanjai Chips</p>', unsafe_allow_html=True)

# Sidebar for parameters
with st.sidebar:
    st.header("Simulation Parameters")
    
    st.subheader("Monte Carlo Parameters")
    a = st.number_input("Multiplier (a)", min_value=1, value=45)
    c = st.number_input("Increment (c)", min_value=1, value=78)
    m = st.number_input("Modulus (m)", min_value=1, value=99)
    z0 = st.number_input("Initial Seed (Z0)", min_value=0, value=10)
    simulations = st.number_input("Number of Simulations", min_value=100, max_value=10000, value=1000)
    
    st.subheader("Data Options")
    use_sample_data = st.checkbox("Use Sample Data from Research", value=True)
    
    st.markdown("---")
    st.markdown("""
    **About the Method:**
    The Monte Carlo method uses random sampling to predict future sales based on historical data.
    This implementation follows the research by Aldo Eko Syaputra (2023).
    """)

# Sample data from the research
sample_data_2021 = {
    "Month": ["January", "February", "March", "April", "May", "June", 
              "July", "August", "September", "October", "November", "December"],
    "Sales": [1523, 1662, 1701, 1122, 1844, 1742, 
              1100, 1986, 1655, 1717, 1870, 1764]
}

sample_data_2022 = {
    "Month": ["January", "February", "March", "April", "May", "June", 
              "July", "August", "September", "October", "November", "December"],
    "Sales": [1942, 1873, 1711, 1564, 1732, 1890, 
              1376, 1840, 1847, 1754, 1899, 1341]
}

# Main content
tab1, tab2, tab3 = st.tabs(["Data Input", "Simulation", "Results"])

with tab1:
    st.header("Input Historical Sales Data")
    
    if use_sample_data:
        st.info("Using sample data from the research paper (2021 and 2022 sales data)")
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("2021 Sales Data")
            df_2021 = pd.DataFrame(sample_data_2021)
            edited_df_2021 = st.data_editor(df_2021, num_rows="dynamic")
            
        with col2:
            st.subheader("2022 Sales Data")
            df_2022 = pd.DataFrame(sample_data_2022)
            edited_df_2022 = st.data_editor(df_2022, num_rows="dynamic")
    else:
        st.warning("Please upload your historical sales data")
        uploaded_file = st.file_uploader("Upload CSV file with monthly sales data", type=["csv"])
        
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
            st.dataframe(df)
        else:
            st.info("No file uploaded. Using sample data for demonstration.")
            df_2021 = pd.DataFrame(sample_data_2021)
            df_2022 = pd.DataFrame(sample_data_2022)

with tab2:
    st.header("Monte Carlo Simulation")
    
    if use_sample_data or ('df_2021' in locals() and 'df_2022' in locals()):
        # Calculate probability distributions
        def calculate_probabilities(df):
            total_sales = df['Sales'].sum()
            df['Probability'] = df['Sales'] / total_sales
            df['Cumulative Probability'] = df['Probability'].cumsum()
            return df
        
        df_2021 = calculate_probabilities(df_2021)
        df_2022 = calculate_probabilities(df_2022)
        
        # Display probability distributions
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("2021 Probability Distribution")
            st.dataframe(df_2021)
            
        with col2:
            st.subheader("2022 Probability Distribution")
            st.dataframe(df_2022)
        
        # Generate random numbers using LCG algorithm
        def generate_random_numbers(a, c, m, z0, n):
            random_numbers = []
            z = z0
            for _ in range(n):
                z = (a * z + c) % m
                random_numbers.append(z / m)  # Normalize to [0,1)
            return random_numbers
        
        # Monte Carlo simulation
        def monte_carlo_simulation(historical_df, random_numbers):
            predictions = []
            for r in random_numbers:
                # Find which month corresponds to this random number
                for i, row in historical_df.iterrows():
                    if r <= row['Cumulative Probability']:
                        predictions.append(row['Sales'])
                        break
            return predictions
        
        if st.button("Run Simulation", key="run_sim"):
            with st.spinner("Running Monte Carlo simulation..."):
                # Generate random numbers
                random_numbers = generate_random_numbers(a, c, m, z0, simulations)
                
                # Run simulation for 2023 based on 2022 data
                predictions_2023 = monte_carlo_simulation(df_2022, random_numbers)
                
                # Calculate monthly predictions (group by month)
                monthly_predictions = []
                for i, month in enumerate(df_2022['Month']):
                    start_prob = 0 if i == 0 else df_2022.iloc[i-1]['Cumulative Probability']
                    end_prob = df_2022.iloc[i]['Cumulative Probability']
                    
                    # Count how many random numbers fall in this range
                    count = sum(1 for r in random_numbers if start_prob < r <= end_prob)
                    predicted_sales = df_2022.iloc[i]['Sales'] * (count / simulations)
                    monthly_predictions.append(predicted_sales)
                
                # Create results dataframe
                results_df = pd.DataFrame({
                    'Month': df_2022['Month'],
                    'Predicted Sales': monthly_predictions,
                    'Last Year Sales': df_2022['Sales']
                })
                
                # Calculate accuracy (simple comparison to last year)
                accuracy = np.mean([min(p/l, l/p) for p, l in zip(monthly_predictions, df_2022['Sales'])]) * 100
                
                # Store results in session state
                st.session_state.results_df = results_df
                st.session_state.accuracy = accuracy
                st.session_state.random_numbers = random_numbers
                
            st.success("Simulation completed successfully!")
            
            # Show results in tab3
            st.experimental_rerun()

with tab3:
    st.header("Simulation Results")
    
    if 'results_df' in st.session_state:
        results_df = st.session_state.results_df
        accuracy = st.session_state.accuracy
        
        # Display results
        st.subheader("Predicted Sales for Next Year")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.dataframe(results_df.style.format({"Predicted Sales": "{:.0f}", "Last Year Sales": "{:.0f}"}))
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Download button
            csv = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Predictions as CSV",
                data=csv,
                file_name='sanjai_sales_predictions.csv',
                mime='text/csv'
            )
        
        with col2:
            st.markdown('<div class="result-card">', unsafe_allow_html=True)
            st.metric("Average Predicted Sales", f"{results_df['Predicted Sales'].mean():.0f}")
            st.metric("Estimated Accuracy", f"{accuracy:.1f}%")
            st.metric("Total Predicted Annual Sales", f"{results_df['Predicted Sales'].sum():.0f}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Visualization
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("Sales Comparison")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.lineplot(data=results_df, x='Month', y='Last Year Sales', label='Last Year Sales', marker='o', ax=ax)
        sns.lineplot(data=results_df, x='Month', y='Predicted Sales', label='Predicted Sales', marker='o', ax=ax)
        ax.set_title('Monthly Sales Comparison')
        ax.set_ylabel('Number of Sales')
        ax.set_xlabel('Month')
        ax.legend()
        plt.xticks(rotation=45)
        st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Random number distribution visualization
        st.markdown('<div class="plot-container">', unsafe_allow_html=True)
        st.subheader("Random Number Distribution")
        
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        sns.histplot(st.session_state.random_numbers, bins=20, kde=True, ax=ax2)
        ax2.set_title('Distribution of Generated Random Numbers')
        ax2.set_xlabel('Random Number Value')
        ax2.set_ylabel('Frequency')
        st.pyplot(fig2)
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.warning("Please run the simulation first from the 'Simulation' tab.")

# Footer
st.markdown("---")
st.markdown("""
**Research Reference:**  
Syaputra, A. E. (2023). Akumulasi Metode Monte Carlo dalam Memperkirakan Tingkat Penjualan Keripik Sanjai.  
*Jurnal Informatika Ekonomi Bisnis*, 5(1), 209-216.
""")
