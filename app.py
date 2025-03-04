import streamlit as st

# Create a sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page", ["Response Prediction", "Dashboard"])

# Load the respective page based on user selection
if page == "Response Prediction":
    # Run the response prediction code
    

    # Import your existing Response_Prediction code
    # Optionally, you can copy the code from Response_Prediction.py here directly, 
    # or import it as a module (if it's structured in a function-based manner)
    import Response_Prediction  # Make sure it's in the same directory or install it as a module

elif page == "Dashboard":
    # Run the dashboard code
    
    # Import your existing dashboard code
    # Similarly, you can either copy the code from dashboard.py here, 
    # or import it as a module
    import dashboard  # Make sure it's in the same directory or install it as a module
