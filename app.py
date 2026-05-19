import streamlit as st
from streamlit_option_menu import option_menu
from src.ui_components import (
    render_data_overview,
    render_predictive_analytics,
    render_resource_optimization,
    render_ai_insights
)

def main():
    """
    main function to run the streamlit application.
    """
    st.set_page_config(page_title="Seshat AI - HMS", layout="wide")
    
    st.title("Seshat AI - Hospital Management System")
    
    #horizontal navigation menu
    selected = option_menu(
        menu_title=None,
        options=["Data Overview", "Predictive Analytics", "Resource Optimization", "AI Insights"],
        icons=["", "", "", ""],
        menu_icon="",
        default_index=0,
        orientation="horizontal",
    )
    
    if selected == "Data Overview":
        render_data_overview()
    elif selected == "Predictive Analytics":
        render_predictive_analytics()
    elif selected == "Resource Optimization":
        render_resource_optimization()
    elif selected == "AI Insights":
        render_ai_insights()

if __name__ == "__main__":
    main()
