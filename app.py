import streamlit as st
import json
import os
from data_loader import DataLoader
from chart_generator import ChartGenerator

# --- Configuration ---
RECOMMENDATIONS_FILE = 'custom_recommendations.json'
CHART_OPTIONS = [
    'Distribution', 'KPI Performance', 'Performance Multiple', 
    'Top vs Bottom Performers', 'Time to First Sale', 'CAR2CATPO Ratio',
    'Attrition Count', 'Average Residency', 'Infant Attrition', 'Retention', 'Cost of Hire'
]

# In app.py

class DashboardApp:
    """The main Streamlit application class that manages the dashboard."""

    def __init__(self):
        """Initializes the plotter. Data loading is now handled in run()."""
        self.plotter = ChartGenerator()

    def _initialize_session_state(self, channel_key):
        """Initializes or loads recommendations for a SPECIFIC channel."""
        session_key = f"recs_{channel_key}"
        filename = f"recs_{channel_key}.json"

        if session_key not in st.session_state:
            if os.path.exists(filename):
                try:
                    with open(filename, 'r') as f:
                        st.session_state[session_key] = json.load(f)
                except (json.JSONDecodeError, FileNotFoundError):
                    st.session_state[session_key] = {}
            else:
                st.session_state[session_key] = {}

    def _save_recommendation(self, channel_key, rec_key, text):
        """Saves a recommendation to the session state and its channel-specific file."""
        session_key = f"recs_{channel_key}"
        filename = f"recs_{channel_key}.json"

        st.session_state[session_key][rec_key] = text
        with open(filename, 'w') as f:
            json.dump(st.session_state[session_key], f, indent=2)
        st.success("Recommendation saved!")

    def _display_recommendation_box(self, channel_key, category_name, chart_name, edit_mode=False):
        """Displays the recommendation box, either as an editor or read-only text."""
        session_key = f"recs_{channel_key}"
        rec_key = f"{category_name}_{chart_name}"
        
        existing_rec = st.session_state.get(session_key, {}).get(rec_key, "")
        
        if edit_mode:
            # ADMIN VIEW: Show an editable text area and save button
            st.markdown("**Edit Recommendation:**")
            custom_rec = st.text_area(
                "Recommendation:", value=existing_rec, height=150, key=f"text_{channel_key}_{rec_key}"
            )
            if st.button("Save", key=f"save_{channel_key}_{rec_key}"):
                self._save_recommendation(channel_key, rec_key, custom_rec)
        elif existing_rec:
            # NORMAL VIEW: Show read-only text if it exists
            st.markdown("**Recommendation:**")
            st.info(existing_rec)

    def run(self):
        """Executes the main application flow."""
        st.set_page_config(page_title="SLIC Analysis Dashboard", page_icon="📊", layout="wide")

        # --- Check for Admin Mode ---
        edit_mode = st.query_params.get("mode") == "edit"

        st.markdown("""
            <div style='text-align: center; margin-bottom: 40px;'>
                <h1 style='font-weight: 800; color: #0047AB; font-size: 46px;'>SLIC Analysis Dashboard</h1>
            </div>
        """, unsafe_allow_html=True)
        
        if edit_mode:
            st.warning("You are in Admin Edit Mode.", icon="✍️")

        import glob
        data_files = glob.glob("data/*.xlsx")
        if not data_files:
            st.error("No Excel files found in the 'data' folder.")
            return
        
        channel_names = {os.path.basename(f).replace('_', ' ').replace('.xlsx', '').title(): f for f in data_files}
        selected_channel_name = st.selectbox("Select a Channel:", options=list(channel_names.keys()))
        selected_file_path = channel_names[selected_channel_name]
        
        channel_key = os.path.basename(selected_file_path).replace('.xlsx', '').lower()
        
        self._initialize_session_state(channel_key)
        loader = DataLoader(selected_file_path)
        
        if not loader.data_sheets:
            st.warning("Dashboard could not be loaded.")
            return

        category = st.selectbox("Select a category to analyze:", options=loader.categories)
        selected_df = loader.get_dataframe(category)
        
        if selected_df is None:
            st.error(f"Could not retrieve data for the selected category: {category}")
            return
            
        all_options = selected_df['Category'].unique()
        selected_options = st.multiselect(
            f"Filter {category} categories:",
            options=all_options,
            default=all_options
        )
        filtered_df = selected_df[selected_df['Category'].isin(selected_options)].copy()
        filtered_df['Category'] = filtered_df['Category'].str.replace(' ', '<br>')

        st.markdown(f"<h2 style='text-align: center; color: #444;'>{category} Analysis Dashboard</h2>", unsafe_allow_html=True)

        selected_charts = st.multiselect(
            '📊 Select visualizations to display:',
            options=CHART_OPTIONS,
            default=['Distribution', 'KPI Performance', 'Retention']
        )
        
        chart_functions = {
            'Distribution': self.plotter.create_distribution_chart,
            'KPI Performance': self.plotter.create_kpi_performance_chart,
            'Performance Multiple': self.plotter.create_performance_multiple_chart,
            'Top vs Bottom Performers': self.plotter.create_top_bottom_performers_chart,
            'Time to First Sale': self.plotter.create_time_to_first_sale_chart,
            'CAR2CATPO Ratio': self.plotter.create_car2catpo_ratio_chart,
            'Attrition Count': self.plotter.create_attrition_count_chart,
            'Average Residency': self.plotter.create_average_residency_chart,
            'Infant Attrition': self.plotter.create_infant_attrition_chart,
            'Retention': self.plotter.create_retention_chart,
            'Cost of Hire': self.plotter.create_cost_of_hire_chart
        }

        num_charts = len(selected_charts)
        if num_charts > 0:
            num_rows = (num_charts + 2) // 3
            for row in range(num_rows):
                cols = st.columns(3)
                for col_idx in range(3):
                    chart_idx = row * 3 + col_idx
                    if chart_idx < num_charts:
                        chart_name = selected_charts[chart_idx]
                        with cols[col_idx]:
                            with st.container(border=True):
                                try:
                                    chart_func = chart_functions.get(chart_name)
                                    if chart_func:
                                        chart_func(filtered_df, category)
                                        # Pass the edit_mode flag here
                                        self._display_recommendation_box(channel_key, category, chart_name, edit_mode)
                                except Exception as e:
                                    st.error(f"Error generating {chart_name} chart: {e}")

if __name__ == "__main__":
    app = DashboardApp()
    app.run()