# In data_loader.py

import pandas as pd
import streamlit as st
import os

@st.cache_data
def load_excel_data(file_path):
    """
    A cached function to load all sheets from a specific Excel file.
    The cache key is based on the file_path.
    """
    if not os.path.exists(file_path):
        st.error(f"Error: Data file not found at '{file_path}'")
        return None
    try:
        all_sheets = pd.read_excel(file_path, sheet_name=None)
        return all_sheets
    except Exception as e:
        st.error(f"Error loading or parsing data from '{file_path}': {e}")
        return None

class DataLoader:
    """Handles data access after it has been loaded."""
    def __init__(self, file_path):
        self.file_path = file_path
        # Call the standalone cached function to load data
        self.data_sheets = load_excel_data(self.file_path)
        # Dynamically get the category names
        self.categories = list(self.data_sheets.keys()) if self.data_sheets else []

    def get_dataframe(self, name):
        """Returns the dataframe for a given category name."""
        if self.data_sheets and name in self.data_sheets:
            return self.data_sheets[name]
        return None