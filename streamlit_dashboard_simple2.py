import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sys
import json
import os

# Set seaborn style
sns.set_theme(style="whitegrid")


# Initialize session state for storing custom recommendations
if 'custom_recommendations' not in st.session_state:
    # Try to load existing recommendations from file
    recommendations_file = 'custom_recommendations.json'
    if os.path.exists(recommendations_file):
        try:
            with open(recommendations_file, 'r') as f:
                st.session_state.custom_recommendations = json.load(f)
        except:
            st.session_state.custom_recommendations = {}
    else:
        st.session_state.custom_recommendations = {}

# Helper function to setup consistent chart styling
def setup_chart_style():
    """Set up consistent styling for all charts with executive-level polish."""
    # Set global font size and weight - make everything bolder and more professional
    plt.rcParams.update({
        'font.size': 14,
        'font.weight': 'bold',
        'axes.titlesize': 20,
        'axes.titleweight': 'bold',
        'axes.labelsize': 16,
        'axes.labelweight': 'bold',
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'figure.constrained_layout.use': True,  # Use constrained layout for better spacing
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.edgecolor': '#333333',
        'axes.linewidth': 1.5,
        'figure.facecolor': '#ffffff',
        'axes.facecolor': '#f9f9f9',
    })
    
    # Create figure with consistent size for all charts
    # Using a fixed aspect ratio to ensure all charts have the same height
    fig, ax = plt.subplots(figsize=(12, 8), dpi=120)  # Increased DPI for sharper images
    
    # Set figure face color to white for better appearance
    fig.set_facecolor('white')
    
    # Adjust the bottom margin to create more space for x-axis labels
    plt.subplots_adjust(bottom=0.15)
    
    # Add a subtle background color to enhance readability
    ax.set_facecolor('#f9f9f9')
    
    # Add a border to the figure for a more polished look
    fig.patch.set_edgecolor('#e0e0e0')
    fig.patch.set_linewidth(2)
    
    return fig, ax

def extend_y_limits(ax, top_extension=0.2):
    """Extend the y-axis limits to add more room at the top for labels."""
    y_min, y_max = ax.get_ylim()
    extra_space = (y_max - y_min) * top_extension
    ax.set_ylim(y_min, y_max + extra_space)

def get_recommendation_key(category, chart_name):
    """Generate a unique key for storing chart recommendations."""
    return f"{category}_{chart_name}"

def save_recommendation(key, text):
    """Save a recommendation to the session state and to a file."""
    st.session_state.custom_recommendations[key] = text
    
    # Save to file
    recommendations_file = 'custom_recommendations.json'
    with open(recommendations_file, 'w') as f:
        json.dump(st.session_state.custom_recommendations, f, indent=2)
    
    # Show a success message
    st.success("Recommendation saved successfully!")

def main():    # Set page configuration
    st.set_page_config(
        page_title="SLIC Analysis Dashboard",
        page_icon="📊",
        layout="wide"
    )
    
    st.markdown("""
        <div style='text-align: center; margin-bottom: 40px;'>
            <h1 style='font-weight: 800; color: #0047AB; font-size: 46px; margin-bottom: 10px; text-shadow: 1px 1px 3px rgba(0,0,0,0.2);'>
                SLIC Analysis Dashboard
            </h1>
            <div style='width: 100px; height: 5px; background-color: #0047AB; margin: 0 auto 10px auto; border-radius: 2px;'></div>
            <p style='color: #555; font-size: 18px; font-weight: 500;'>Executive Summary Report</p>
        </div>
    """, unsafe_allow_html=True)
      # Add information about the custom recommendation feature
    with st.expander("About Custom Recommendations", expanded=False):
        st.info("""
        **How to Use Custom Recommendations:**
        
        1. Each chart has a recommendation input box where you can write your own analysis and insights.
        2. Your recommendations will be saved when you click the "Save" button.
        3. Recommendations are saved per chart and per category (Gender, Education, Experience, Age).
        4. All recommendations are stored in a file called "custom_recommendations.json" which persists between dashboard sessions.
        """)
    
    # Load all dataframes
    with st.spinner('Loading data from SLIC_modified.xlsx...'):
        try:
            st.info("Loading data from Excel file. This may take a moment...")
            Gender = pd.read_excel('SLIC_modified.xlsx', sheet_name='Gender')
            Education = pd.read_excel('SLIC_modified.xlsx', sheet_name='Education')
            Experience = pd.read_excel('SLIC_modified.xlsx', sheet_name='Experience')
            Age = pd.read_excel('SLIC_modified.xlsx', sheet_name='Age')
            Designation = pd.read_excel('SLIC_modified.xlsx', sheet_name='Designation')

            st.success("Data loaded successfully!")
            
            # Create list of dataframes with their display names
            all_dataframes = [
                {"df": Gender, "name": "Gender"},
                {"df": Education, "name": "Education"},
                {"df": Experience, "name": "Experience"},
                {"df": Age, "name": "Age"},
                {"df": Designation, "name": "Designation"}

            ]
              # Create a dropdown to select the category with clear styling
            st.markdown("<h2 style='text-align: center; color: #444; margin: 20px 0;'>Select a category to analyze:</h2>", unsafe_allow_html=True)
            category = st.selectbox(
                "",
                ['Gender', 'Education', 'Experience', 'Age', 'Designation'],
                index=1 if 'Education' in sys.argv else 0,
                format_func=lambda x: f"{x}"
            )
            
            # Find the corresponding dataframe
            selected_df = next(data for data in all_dataframes if data["name"] == category)
            
            # Create the dashboard for the selected category
            create_dashboard(selected_df["df"], selected_df["name"])
                    
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            st.exception(e)  # This will display the full traceback

def create_dashboard(df, name):
    """Create a dashboard visualization for the given dataframe in Streamlit."""
    st.markdown(f"<h1 style='text-align: center; font-weight: 800; color: #0A2472; margin-bottom: 20px; text-shadow: 1px 1px 2px #ccc;'>{name} Analysis Dashboard</h1>", unsafe_allow_html=True)
    
    # Show information about the data with improved styling
    st.markdown(f"<h3 style='text-align: center; color: #444; background-color: #f8f9fa; padding: 10px; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);'>Executive Dashboard • {len(df)} Data Points</h3>", unsafe_allow_html=True)
    
    # Enable user to select which charts to display with improved styling
    st.markdown("<div style='background-color: #f0f2f6; padding: 15px; border-radius: 10px; margin-bottom: 20px;'>", unsafe_allow_html=True)
    selected_charts = st.multiselect(
        '📊 Select visualizations to display:',
        ['Distribution', 'KPI Performance', 'Performance Multiple', 
         'Top vs Bottom Performers', 'Time to First Sale', 'CAR2CATPO Ratio',
         'Attrition Count', 'Average Residency', 'Infant Attrition', 'Retention', 'Cost of Hire'],
        default=['Distribution', 'KPI Performance', 'Performance Multiple', 'Retention']
    )
    st.markdown("</div>", unsafe_allow_html=True)
    
    # Create a list of chart creation functions
    chart_functions = {
        'Distribution': create_distribution_chart,
        'KPI Performance': create_kpi_performance_chart,
        'Performance Multiple': create_performance_multiple_chart,
        'Top vs Bottom Performers': create_top_bottom_performers_chart,
        'Time to First Sale': create_time_to_first_sale_chart,
        'CAR2CATPO Ratio': create_car2catpo_ratio_chart,
        'Attrition Count': create_attrition_count_chart,
        'Average Residency': create_average_residency_chart,
        'Infant Attrition': create_infant_attrition_chart,
        'Retention': create_retention_chart,
        'Cost of Hire': create_cost_of_hire_chart
    }
      # Organize charts into rows with equal heights
    # Determine how many rows we need (3 charts per row)
    num_charts = len(selected_charts)
    num_rows = (num_charts + 2) // 3  # Integer division rounded up
      # Create each row of charts
    for row in range(num_rows):
        # Create columns for this row
        cols = st.columns(3)
        
        # Add charts to this row
        for col_idx in range(3):
            chart_idx = row * 3 + col_idx
            if chart_idx < num_charts:
                chart_name = selected_charts[chart_idx]
                with cols[col_idx]:
                    try:
                        with st.container():
                            st.markdown(f"""
                                <div style='border: 1px solid #e0e0e0; border-radius: 10px; overflow: hidden; margin-bottom: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
                                    <h2 style='text-align: center; font-weight: 700; color: #0A2472; background-color: #f0f2f6; 
                                    padding: 15px; margin: 0; border-bottom: 2px solid #e0e0e0;'>
                                    {chart_name}</h2>
                                    <div style='padding: 10px 0;'>
                            """, unsafe_allow_html=True)
                            chart_functions[chart_name](df, name)
                            st.markdown("</div></div>", unsafe_allow_html=True)
                    except Exception as e:
                        st.error(f"Error generating {chart_name} chart: {str(e)}")


def create_distribution_chart(df, name):
    """Create the distribution chart."""
    fig, ax = setup_chart_style()
    
    # Extract the first 3 columns
    first_cols = df.columns[:3]
    # Reshape data for seaborn
    df_melted = pd.melt(df, 
                        id_vars=[first_cols[0]], 
                        value_vars=[first_cols[1], first_cols[2]], 
                        var_name='Metric', 
                        value_name='Count')

    # Calculate percentages for each cohort
    for metric in [first_cols[1], first_cols[2]]:
        total = df[metric].sum()
        # Avoid division by zero if a cohort has no members
        if total > 0:
            df_melted.loc[df_melted['Metric'] == metric, 'Percentage'] = df_melted.loc[df_melted['Metric'] == metric, 'Count'] / total * 100
        else:
            df_melted.loc[df_melted['Metric'] == metric, 'Percentage'] = 0
    
    # Using seaborn barplot with grouped bars and better colors
    bars = sns.barplot(x='Category', y='Count', hue='Metric', 
                       data=df_melted, 
                       palette=['#1f77b4', '#9ecae1'], 
                       ax=ax, width=0.7)

    ax.set_title(f'{name} Distribution by Cohort', pad=20)
    ax.set_xlabel(f'{name}', labelpad=15)
    ax.set_ylabel('Head Count', labelpad=15)

    # --- Start of Modified Section ---
    # Adding value labels dynamically inside or outside the bars
    base_fontsize = 16
    # Set a dynamic threshold: 10% of the max y-axis value
    threshold = ax.get_ylim()[1] * 0.10 

    for container_idx, container in enumerate(ax.containers):
        for bar_idx, bar in enumerate(container):
            count = bar.get_height()
            
            # Skip labeling for bars with zero height
            if count == 0:
                continue

            x_pos = bar.get_x() + bar.get_width() / 2
            label_text = f'{int(count)}'

            # If the bar is too short, place the text ON TOP
            if count < threshold:
                # Add a small offset based on the chart's scale for padding
                offset = ax.get_ylim()[1] * 0.02 
                y_pos = count + offset # Position text above the bar
                color = 'black'
                
                ax.text(x_pos, y_pos, label_text,
                        ha='center', va='bottom', 
                        color=color,
                        fontsize=base_fontsize,
                        fontweight='bold')
                
            # Otherwise, place the text INSIDE the bar
            else:
                y_pos = count / 2  # Position text in the middle
                color = 'white'
                ax.text(x_pos, y_pos, label_text,
                        ha='center', va='center',
                        color=color,
                        fontsize=base_fontsize,
                        fontweight='bold')
    # --- End of Modified Section ---

    # Enhance grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust legend with better styling
    legend = ax.legend(title='Cohort Type', fontsize=14)
    plt.setp(legend.get_title(), fontsize=16, fontweight='bold')
    # Extend y-axis to provide more space for labels
    extend_y_limits(ax, 0.2)
    
    # Rotate x-axis labels for Education dashboard
    if name == "Education":
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
    # Add more padding around figure
    plt.tight_layout(pad=3.0)
    
    # Use full width in Streamlit
    st.pyplot(fig, use_container_width=True)
    
    # Generate a unique key for this chart's recommendation
    rec_key = get_recommendation_key(name, "Distribution")
    
    # Get any existing recommendation
    existing_rec = st.session_state.custom_recommendations.get(rec_key, "")
    
    # Create text area for custom recommendation
    custom_rec = st.text_area(
        "Recommendation:",
        value=existing_rec,
        height=150,
        key=f"text_{rec_key}"
    )
    
    # Add a save button
    if st.button("Save", key=f"save_{rec_key}"):
        save_recommendation(rec_key, custom_rec)

def create_retention_chart(df, name):
    """Create the Retention chart."""
    fig, ax = setup_chart_style()
    
    # Get retention columns
    col23 = df.columns[22]  # Retention at CAP 3
    col24 = df.columns[23]  # Retention at CAP 6
    col25 = df.columns[24]  # Retention at CAP 9
    col26 = df.columns[25]  # Retention at CAP 12

    # Create shorter column names for display
    col23_short = "Retention at CAP 3"
    col24_short = "Retention at CAP 6"
    col25_short = "Retention at CAP 9"
    col26_short = "Retention at CAP 12"

    # Create a DataFrame with the data, converting to percentages
    retention_data = pd.DataFrame({
        'Category': df['Category'],
        col23_short: df[col23] * 100,
        col24_short: df[col24] * 100,
        col25_short: df[col25] * 100,
        col26_short: df[col26] * 100
    })

    # Reshape data for seaborn
    retention_melted = pd.melt(retention_data, 
                          id_vars=['Category'], 
                          value_vars=[col23_short, col24_short, col25_short, col26_short],
                          var_name='Retention Period', 
                          value_name='Retention Rate (%)')
    
    # Using seaborn barplot with grouped bars
    sns.barplot(x='Category', y='Retention Rate (%)', hue='Retention Period', 
                data=retention_melted, 
                palette='viridis', 
                ax=ax, width=0.8)

    ax.set_title(f'Employee Retention Rate by {name}', pad=20)
    ax.set_xlabel(f'{name}', labelpad=15)
    ax.set_ylabel('Retention Rate (%)', labelpad=15)

    # Adding value labels dynamically inside or outside the bars
    base_fontsize = 12
    threshold = ax.get_ylim()[1] * 0.10

    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            if height == 0:
                continue
            
            label_text = f'{height:.0f}%'
            x_pos = bar.get_x() + bar.get_width() / 2

            if height < threshold:
                y_pos = height
                color = 'black'
                ax.text(x_pos, y_pos, label_text,
                        ha='center', va='bottom',
                        color=color,
                        fontsize=base_fontsize,
                        fontweight='bold')

            else:
                y_pos = height / 2
                color = 'white'
                ax.text(x_pos, y_pos, label_text,
                        ha='center', va='center',
                        color=color,
                        fontsize=base_fontsize,
                        fontweight='bold')

    # Enhance grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust legend with better styling
    legend = ax.legend(title='Retention Period', fontsize=12)
    plt.setp(legend.get_title(), fontsize=14, fontweight='bold')

    # Extend y-axis to provide more space for labels
    extend_y_limits(ax, 0.2)
    
    # Rotate x-axis labels for Education dashboard
    if name == "Education":
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
    # Add more padding around figure
    plt.tight_layout(pad=3.0)
    
    # Use full width in Streamlit
    st.pyplot(fig, use_container_width=True)
    
    # Generate a unique key for this chart's recommendation
    rec_key = get_recommendation_key(name, "Retention")
    
    # Get any existing recommendation
    existing_rec = st.session_state.custom_recommendations.get(rec_key, "")
    
    # Create text area for custom recommendation
    custom_rec = st.text_area(
        "Recommendation:",
        value=existing_rec,
        height=150,
        key=f"text_{rec_key}"
    )
    
    # Add a save button
    if st.button("Save", key=f"save_{rec_key}"):
        save_recommendation(rec_key, custom_rec)


def create_cost_of_hire_chart(df, name):
    """Create the Cost of Hire chart."""
    fig, ax = setup_chart_style()
    
    # Get the 21st and 22nd columns (indices 20, 21)
    col21 = df.columns[20]  # Cost of Wrong Hire
    col22 = df.columns[21]  # Cost of Back Fill

    # Create shorter column names for display
    col21_short = "Cost of Wrong Hire"
    col22_short = "Cost of Back Fill"

    # Create a DataFrame for the chart
    cost_data = pd.DataFrame({
        'Category': df['Category'],
        col21_short: df[col21],
        col22_short: df[col22]
    })

    # Reshape data for seaborn
    cost_melted = pd.melt(cost_data, 
                             id_vars=['Category'], 
                             value_vars=[col21_short, col22_short],
                             var_name='Cost Type', 
                             value_name='Cost of Hire')

    # Using seaborn barplot with grouped bars
    bars = sns.barplot(x='Category', y='Cost of Hire', hue='Cost Type', 
                      data=cost_melted, 
                      palette=['#d62728', '#ff9896'], 
                      ax=ax, width=0.7)

    ax.set_title(f'Cost of Hire by {name}', pad=20)
    ax.set_xlabel(f'{name}', labelpad=15)
    ax.set_ylabel('Cost of Hire (in thousands)', labelpad=15)

    # Adding value labels dynamically inside or outside the bars
    base_fontsize = 16
    threshold = ax.get_ylim()[1] * 0.10

    for bar in bars.patches:
        height = bar.get_height()
        if height == 0:
            continue
            
        label_text = f'{height:,.1f}'
        x_pos = bar.get_x() + bar.get_width() / 2

        if height < threshold:
            y_pos = height
            color = 'black'
            ax.text(x_pos, y_pos, label_text,
                    ha='center', va='bottom',
                    color=color,
                    fontsize=base_fontsize,
                    fontweight='bold',
                    xytext=(0, 5),
                    textcoords='offset points')
        else:
            y_pos = height / 2
            color = 'white'
            ax.text(x_pos, y_pos, label_text,
                    ha='center', va='center',
                    color=color,
                    fontsize=base_fontsize,
                    fontweight='bold')

    # Enhance grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust legend
    ax.legend(title='Cost Type', loc='upper right')
    
    # Extend y-axis to provide more space for labels
    extend_y_limits(ax, 0.2)
    
    # Rotate x-axis labels for Education dashboard
    if name == "Education":
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
    # Add more padding around figure
    plt.tight_layout(pad=3.0)
    
    # Use full width in Streamlit
    st.pyplot(fig, use_container_width=True)
    
    # Generate a unique key for this chart's recommendation
    rec_key = get_recommendation_key(name, "Cost of Hire")
    
    # Get any existing recommendation
    existing_rec = st.session_state.custom_recommendations.get(rec_key, "")
    
    # Create text area for custom recommendation
    custom_rec = st.text_area(
        "Recommendation:",
        value=existing_rec,
        height=150,
        key=f"text_{rec_key}"
    )
    
    # Add a save button
    if st.button("Save", key=f"save_{rec_key}"):
        save_recommendation(rec_key, custom_rec)

def create_kpi_performance_chart(df, name):
    """Create the KPI performance chart."""
    fig, ax = setup_chart_style()
    
    # Get the 8th column (index 7)
    col8 = df.columns[7]

    # Create shorter column names for display
    col8_short = "Cumulative KPI 1"
    
    # Create a DataFrame with the data and specified columns, multiplying KPI values by 100 to show as percentages
    kpi_data = pd.DataFrame({
        'Category': df['Category'],
        col8_short: df[col8] * 100   # Multiply by 100 to convert to percentage
    })

    # Using seaborn barplot with a single metric
    bars = sns.barplot(x='Category', y=col8_short, 
                      data=kpi_data, 
                      palette=['#ff7f0e'], 
                      ax=ax, width=0.7)

    ax.set_title(f'KPI 1 Performance by {name} CAP LRM', pad=20)
    ax.set_xlabel(f'{name}', labelpad=15)
    ax.set_ylabel('Achievement %', labelpad=15)
    
    # Increased font size by 20% (from 13 to 16)
    base_fontsize = 16  # Increased from 13
    
    # Adding value labels dynamically inside or outside the bars
    base_fontsize = 16
    threshold = ax.get_ylim()[1] * 0.10
    
    for bar in bars.patches:
        height = bar.get_height()
        if height == 0:
            continue
        
        label_text = f'{height:.0f}%'
        x_pos = bar.get_x() + bar.get_width() / 2
        
        if height < threshold:
            y_pos = height
            color = 'black'
            ax.text(x_pos, y_pos, label_text,
                   ha='center', va='bottom',
                   color=color,
                   fontsize=base_fontsize,
                   fontweight='bold',
                   xytext=(0, 5),
                   textcoords='offset points')
        else:
            y_pos = height / 2
            color = 'white'
            ax.text(x_pos, y_pos, label_text,
                   ha='center', va='center',
                   color=color,
                   fontsize=base_fontsize,
                   fontweight='bold')

    # Enhance grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Extend y-axis to provide more space for labels
    extend_y_limits(ax, 0.3)  # Increased from 0.2 to 0.3 to make room for ticks and annotations
    
    # Rotate x-axis labels for Education dashboard
    if name == "Education":
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
    # Add more padding around figure
    plt.tight_layout(pad=3.0)
    
    # Use full width in Streamlit
    st.pyplot(fig, use_container_width=True)
    
    # Generate a unique key for this chart's recommendation
    rec_key = get_recommendation_key(name, "KPI Performance")
    
    # Get any existing recommendation
    existing_rec = st.session_state.custom_recommendations.get(rec_key, "")
    
    # Create text area for custom recommendation
    custom_rec = st.text_area(
        "Recommendation:",
        value=existing_rec,
        height=150,
        key=f"text_{rec_key}"
    )
    
    # Add a save button
    if st.button("Save", key=f"save_{rec_key}"):
        save_recommendation(rec_key, custom_rec)

def create_performance_multiple_chart(df, name):
    """Create the performance multiple chart."""
    fig, ax = setup_chart_style()
    
    # Get the 11th column (index 10)
    col11 = df.columns[10]

    # Create shorter column names for display
    col11_short = "Performance Multiple KPI 1"

    # Create a DataFrame with the data and specified columns
    perf_data = pd.DataFrame({
        'Category': df['Category'],
        col11_short: df[col11]
    })

    # Using seaborn barplot
    bars = sns.barplot(x='Category', y=col11_short, 
                      data=perf_data, 
                      palette=['#2ca02c'], 
                      ax=ax, width=0.7)

    ax.set_title(f'Performance Multiple by {name}', pad=20)
    ax.set_xlabel(f'{name}', labelpad=15)
    ax.set_ylabel('Multiple Value', labelpad=15)
    
    # Adding value labels dynamically inside or outside the bars
    base_fontsize = 16
    threshold = ax.get_ylim()[1] * 0.10

    for bar in bars.patches:
        height = bar.get_height()
        if height == 0:
            continue
        
        label_text = f'{height:.1f}x'
        x_pos = bar.get_x() + bar.get_width() / 2

        if height < threshold:
            y_pos = height
            color = 'black'
            ax.text(x_pos, y_pos, label_text,
                   ha='center', va='bottom',
                   color=color,
                   fontsize=base_fontsize,
                   fontweight='bold',
                   xytext=(0, 5),
                   textcoords='offset points')
        else:
            y_pos = height / 2
            color = 'white'
            ax.text(x_pos, y_pos, label_text,
                   ha='center', va='center',
                   color=color,
                   fontsize=base_fontsize,
                   fontweight='bold')

    # Enhance grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Extend y-axis to provide more space for labels
    extend_y_limits(ax, 0.2)
    
    # Rotate x-axis labels for Education dashboard
    if name == "Education":
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
    # Add more padding around figure
    plt.tight_layout(pad=3.0)
    
    # Use full width in Streamlit
    st.pyplot(fig, use_container_width=True)
    
    # Generate a unique key for this chart's recommendation
    rec_key = get_recommendation_key(name, "Performance Multiple")
    
    # Get any existing recommendation
    existing_rec = st.session_state.custom_recommendations.get(rec_key, "")
    
    # Create text area for custom recommendation
    custom_rec = st.text_area(
        "Recommendation:",
        value=existing_rec,
        height=150,
        key=f"text_{rec_key}"
    )
    
    # Add a save button
    if st.button("Save", key=f"save_{rec_key}"):
        save_recommendation(rec_key, custom_rec)

def create_top_bottom_performers_chart(df, name):
    """Create the top and bottom performers chart."""
    fig, ax = setup_chart_style()
    
    # Get the 9th and 10th columns (indices 8, 9)
    col9 = df.columns[8]  # Top performers KPI 1
    col10 = df.columns[9]  # Bottom performers KPI 1

    # Create shorter column names for display
    col9_short = "Top 10% (KPI 1)"
    col10_short = "Bottom 10% (KPI 1)"

    # Create a DataFrame with the data and specified columns
    performer_data = pd.DataFrame({
        'Category': df['Category'],
        col9_short: df[col9],
        col10_short: df[col10]
    })

    # Reshape data for seaborn
    all_performers = pd.melt(performer_data, 
                         id_vars=['Category'], 
                         value_vars=[col9_short, col10_short],
                         var_name='Performance', 
                         value_name='Value')
    
    # Using seaborn barplot with grouped bars with enhanced colors
    bars = sns.barplot(x='Category', y='Value', hue='Performance', 
                      data=all_performers, 
                      palette=['#9467bd', '#d8b2ff'],
                      ax=ax, width=0.7)

    ax.set_title(f'Top vs Bottom Performers by {name}', pad=20)
    ax.set_xlabel(f'{name}', labelpad=15)
    ax.set_ylabel('CAP Value', labelpad=15)
    
    # Adding value labels dynamically inside or outside the bars
    base_fontsize = 16
    threshold = ax.get_ylim()[1] * 0.10

    for container in ax.containers:
        for bar in container:
            height = bar.get_height()
            if height == 0:
                continue
            
            label_text = f'{height:.1f}'
            x_pos = bar.get_x() + bar.get_width() / 2

            if height < threshold:
                y_pos = height
                offset = ax.get_ylim()[1] * 0.02 
                color = 'black'
                ax.text(x_pos, y_pos, label_text,
                       ha='center', va='bottom',
                       color=color,
                       fontsize=base_fontsize,
                       fontweight='bold',)
            else:
                y_pos = height / 2
                color = 'white'
                ax.text(x_pos, y_pos, label_text,
                       ha='center', va='center',
                       color=color,
                       fontsize=base_fontsize,
                       fontweight='bold')

    # Enhance grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust legend with better styling
    legend = ax.legend(title='Performance Group', fontsize=14)
    plt.setp(legend.get_title(), fontsize=16, fontweight='bold')
    
    # Extend y-axis to provide more space for labels
    extend_y_limits(ax, 0.2)
    
    # Rotate x-axis labels for Education dashboard
    if name == "Education":
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
    # Add more padding around figure
    plt.tight_layout(pad=3.0)
    
    # Use full width in Streamlit
    st.pyplot(fig, use_container_width=True)
    
    # Generate a unique key for this chart's recommendation
    rec_key = get_recommendation_key(name, "Top vs Bottom Performers")
    
    # Get any existing recommendation
    existing_rec = st.session_state.custom_recommendations.get(rec_key, "")
    
    # Create text area for custom recommendation
    custom_rec = st.text_area(
        "Recommendation:",
        value=existing_rec,
        height=150,
        key=f"text_{rec_key}"
    )
    
    # Add a save button
    if st.button("Save", key=f"save_{rec_key}"):
        save_recommendation(rec_key, custom_rec)

def create_time_to_first_sale_chart(df, name):
    """Create the time to first sale chart."""
    fig, ax = setup_chart_style()
    
    # Get the 12th column (index 11)
    col12 = df.columns[11]

    # Create a DataFrame for the chart
    first_sale_data = pd.DataFrame({
        'Category': df['Category'],
        'Time to First Sale': df[col12]
    })

    # Using seaborn barplot (fixed deprecation warning)    
    # Generate a palette with enough colors for all categories
    category_count = len(first_sale_data['Category'].unique())
    palette = sns.color_palette("Blues_d", category_count)
    
    bars = sns.barplot(x='Category', y='Time to First Sale', data=first_sale_data, 
                     hue='Category', palette=palette, legend=False, ax=ax, width=0.7)

    ax.set_title(f'Time to Make First Sale by {name}', pad=20)
    ax.set_xlabel(f'{name}', labelpad=15)
    ax.set_ylabel('Time (months)', labelpad=15)    # Adding value labels dynamically inside or outside the bars
    base_fontsize = 16
    threshold = ax.get_ylim()[1] * 0.10

    for i, container in enumerate(ax.containers):
        for j, bar in enumerate(container):
            height = bar.get_height()
            if height == 0:
                continue
            
            label_text = f'{height:.2f}'
            x_pos = bar.get_x() + bar.get_width() / 2

            if height < threshold:
                y_pos = height
                color = 'black'
                ax.text(x_pos, y_pos, label_text,
                        ha='center', va='bottom',
                        color=color,
                        fontweight='bold',
                        fontsize=base_fontsize)
            else:
                y_pos = height / 2
                color = 'white'
                ax.text(x_pos, y_pos, label_text,
                        ha='center', va='center',
                        color=color,
                        fontweight='bold',
                        fontsize=base_fontsize)    # Add a horizontal line for the average with larger font size
    avg_time = df[col12].mean()
    ax.axhline(y=avg_time, color='red', linestyle='--', alpha=0.7)
      # Find appropriate empty space for the average label
    # Move text to upper right corner of the chart instead of directly on the line
    right_edge = ax.get_xlim()[1]
    y_min, y_max = ax.get_ylim()
    y_position = y_max * 0.9  # Position at 90% of chart height
    
    ax.text(right_edge * 0.8, y_position, f'Average: {avg_time:.2f} months', 
            color='red', ha='right', va='center', fontsize=14, fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='red', alpha=0.7, pad=5, boxstyle='round'))
    
    # Enhance grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    # Extend y-axis to provide more space for labels
    extend_y_limits(ax, 0.2)
    # Extend y-axis to provide more space for labels
    extend_y_limits(ax, 0.2)
    
    # Rotate x-axis labels for Education dashboard
    if name == "Education":
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
    # Add more padding around figure
    plt.tight_layout(pad=3.0)
    
    # Use full width in Streamlit
    st.pyplot(fig, use_container_width=True)
    
    # Generate a unique key for this chart's recommendation
    rec_key = get_recommendation_key(name, "Time to First Sale")
    
    # Get any existing recommendation
    existing_rec = st.session_state.custom_recommendations.get(rec_key, "")
    
    # Create text area for custom recommendation
    custom_rec = st.text_area(
        "Recommendation:",
        value=existing_rec,
        height=150,
        key=f"text_{rec_key}"
    )
    
    # Add a save button
    if st.button("Save", key=f"save_{rec_key}"):
        save_recommendation(rec_key, custom_rec)

def create_car2catpo_ratio_chart(df, name):
    """Create the CAR2CATPO ratio chart."""
    fig, ax = setup_chart_style()
    
    # Get the 13th column (index 12)
    col13 = df.columns[12]

    # Create a DataFrame for the chart
    ratio_data = pd.DataFrame({
        'Category': df['Category'],
        'CAR2CATPO Ratio': df[col13]
    })

    # Using seaborn barplot (fixed deprecation warning)
    # Generate a palette with enough colors for all categories
    category_count = len(ratio_data['Category'].unique())
    palette = sns.color_palette("Greens_d", category_count)    
    bars = sns.barplot(x='Category', y='CAR2CATPO Ratio', data=ratio_data, 
                      hue='Category', palette=palette, legend=False, ax=ax, width=0.7)

    ax.set_title(f'CAR2CATPO Ratio by {name}', pad=20)
    ax.set_xlabel(f'{name}', labelpad=15)
    ax.set_ylabel('Ratio Value', labelpad=15)    # Adding value labels dynamically inside or outside the bars
    base_fontsize = 16
    threshold = ax.get_ylim()[1] * 0.10

    for i, container in enumerate(ax.containers):
        for j, bar in enumerate(container):
            height = bar.get_height()
            if height == 0:
                continue
            
            label_text = f'{height:.2f}'
            x_pos = bar.get_x() + bar.get_width() / 2

            if height < threshold:
                y_pos = height
                color = 'black'
                ax.text(x_pos, y_pos, label_text,
                        ha='center', va='bottom',
                        color=color,
                        fontweight='bold',
                        fontsize=base_fontsize,
                        xytext=(0, 5),
                        textcoords='offset points')
            else:
                y_pos = height / 2
                color = 'white'
                ax.text(x_pos, y_pos, label_text,
                        ha='center', va='center',
                        color=color,
                        fontweight='bold',
                        fontsize=base_fontsize)    # Add a horizontal line for the average with larger font size
    avg_ratio = df[col13].mean()
    ax.axhline(y=avg_ratio, color='red', linestyle='--', alpha=0.7)
      # Find appropriate empty space for the average label
    # Move text to upper right corner of the chart instead of directly on the line
    right_edge = ax.get_xlim()[1]
    y_min, y_max = ax.get_ylim()
    y_position = y_max * 0.9  # Position at 90% of chart height
    
    ax.text(right_edge * 0.8, y_position, f'Average: {avg_ratio:.2f}', 
            color='red', ha='right', va='center', fontsize=14, fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='red', alpha=0.7, pad=5, boxstyle='round'))
    
    # Enhance grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    # Extend y-axis to provide more space for labels
    extend_y_limits(ax, 0.2)
    # Extend y-axis to provide more space for labels
    extend_y_limits(ax, 0.2)
    
    # Rotate x-axis labels for Education dashboard
    if name == "Education":
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
    # Add more padding around figure
    plt.tight_layout(pad=3.0)
    
    # Use full width in Streamlit
    st.pyplot(fig, use_container_width=True)
    
    # Generate a unique key for this chart's recommendation
    rec_key = get_recommendation_key(name, "CAR2CATPO Ratio")
    
    # Get any existing recommendation
    existing_rec = st.session_state.custom_recommendations.get(rec_key, "")
    
    # Create text area for custom recommendation
    custom_rec = st.text_area(
        "Recommendation:",
        value=existing_rec,
        height=150,
        key=f"text_{rec_key}"
    )
    
    # Add a save button
    if st.button("Save", key=f"save_{rec_key}"):
        save_recommendation(rec_key, custom_rec)

def create_attrition_count_chart(df, name):
    """Create the attrition count chart."""
    fig, ax = setup_chart_style()
    
    # Get the 14th column (index 13)
    col14 = df.columns[13]

    # Create a DataFrame for the chart
    attrition_data = pd.DataFrame({
        'Category': df['Category'],
        'Attrited Employees': df[col14]
    })

    # Using seaborn barplot (fixed deprecation warning)
    # Generate a palette with enough colors for all categories
    category_count = len(attrition_data['Category'].unique())
    palette = sns.color_palette("Reds_d", category_count)
    
    bars = sns.barplot(x='Category', y='Attrited Employees', data=attrition_data, 
                     hue='Category', palette=palette, legend=False, ax=ax, width=0.7)

    ax.set_title(f'Employee Attrition by {name}', pad=20)
    ax.set_xlabel(f'{name}', labelpad=15)
    ax.set_ylabel('Number of Attrited Employees', labelpad=15)    # Calculate and display attrition percentages
    total_per_category = df['CAP LRM cohort'].values
    attrition_per_category = df[col14].values
    attrition_rates = attrition_per_category / total_per_category * 100

    # Increased font size by 20%
    base_fontsize = 16  # Increased from 13
      # Add value count and percentage annotations dynamically
    base_fontsize = 16
    threshold = ax.get_ylim()[1] * 0.10

    for i, container in enumerate(ax.containers):
        for j, bar in enumerate(container):
            count = bar.get_height()
            if count == 0:
                continue
            
            rate = attrition_rates[j]
            x_pos = bar.get_x() + bar.get_width() / 2

            if count < threshold:
                y_pos = count
                color = 'black'
                ax.text(x_pos, y_pos, f'{int(count)} ({rate:.1f}%)',
                        ha='center', va='bottom',
                        color=color,
                        fontweight='bold',
                        fontsize=base_fontsize,
                        xytext=(0, 5),
                        textcoords='offset points')
            else:
                # Position for the value at top third of the bar
                ax.text(x_pos, count * 0.7,
                        f'{int(count)}',
                        ha='center', va='center', color='white',
                        fontweight='bold', fontsize=base_fontsize)
                
                # Position for the percentage at bottom third of the bar
                ax.text(x_pos, count * 0.3,
                        f'({rate:.1f}%)',
                        ha='center', va='center', color='white',
                        fontweight='bold', fontsize=base_fontsize - 2)
                    
    # Enhance grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    # Extend y-axis to provide more space for labels
    extend_y_limits(ax, 0.2)
    # Extend y-axis to provide more space for labels
    extend_y_limits(ax, 0.2)

    # Rotate x-axis labels for Education dashboard
    if name == "Education":
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
    # Add more padding around figure
    plt.tight_layout(pad=3.0)
    
    # Use full width in Streamlit
    st.pyplot(fig, use_container_width=True)
    
    # Generate a unique key for this chart's recommendation
    rec_key = get_recommendation_key(name, "Attrition Count")
    
    # Get any existing recommendation
    existing_rec = st.session_state.custom_recommendations.get(rec_key, "")
    
    # Create text area for custom recommendation
    custom_rec = st.text_area(
        "Recommendation:",
        value=existing_rec,
        height=150,
        key=f"text_{rec_key}"
    )
    
    # Add a save button
    if st.button("Save", key=f"save_{rec_key}"):
        save_recommendation(rec_key, custom_rec)

def create_average_residency_chart(df, name):
    """Create the average residency chart."""
    fig, ax = setup_chart_style()
    
    # Get the 15th column (index 14)
    col15 = df.columns[14]  # Average Residency of all employees
    col16 = df.columns[15]  # Average Residency of TOP 100 employees

    # Create shorter column names for display
    col15_short = "All Employees"
    col16_short = "Top 100 Performers"

    # Create a DataFrame for the chart
    residency_data = pd.DataFrame({
        'Category': df['Category'],
        col15_short: df[col15],
        col16_short: df[col16]
    })    # Calculate the percentage differences between Top 100 and All employees
    for idx, row in residency_data.iterrows():
        if row[col15_short] > 0:  # Avoid division by zero
            residency_data.loc[idx, 'Percentage Diff'] = ((row[col16_short] - row[col15_short]) / row[col15_short]) * 100
        else:
            residency_data.loc[idx, 'Percentage Diff'] = 0

    # Reshape data for seaborn
    residency_melted = pd.melt(residency_data, 
                             id_vars=['Category'], 
                             value_vars=[col15_short, col16_short],
                             var_name='Employee Group', 
                             value_name='Average Residency')

    # Using seaborn barplot with grouped bars - improved color palette
    bars = sns.barplot(x='Category', y='Average Residency', hue='Employee Group', 
                      data=residency_melted, 
                      palette=['#4472C4', '#8FAADC'], 
                      ax=ax, width=0.7)

    ax.set_title(f'Employment Tenure by {name}', pad=20)
    ax.set_xlabel(f'{name}', labelpad=15)
    ax.set_ylabel('Average Tenure (months)', labelpad=15)    # Adding value labels dynamically inside or outside the bars
    base_fontsize = 16
    threshold = ax.get_ylim()[1] * 0.10

    for container_idx, container in enumerate(ax.containers):
        for bar_idx, bar in enumerate(container):
            height = bar.get_height()
            if height == 0:
                continue
            
            label_text = f'{height:.2f}'
            x_pos = bar.get_x() + bar.get_width() / 2

            if height < threshold:
                y_pos = height
                color = 'black'
                ax.text(x_pos, y_pos, label_text,
                        ha='center', va='bottom',
                        color=color,
                        fontweight='bold',
                        fontsize=base_fontsize,
                        xytext=(0, 5),
                        textcoords='offset points')
            else:
                y_pos = height / 2
                color = 'white'
                ax.text(x_pos, y_pos, label_text,
                        ha='center', va='center',
                        color=color,
                        fontweight='bold',
                        fontsize=base_fontsize)    # Add horizontal line for overall average tenure for all employees with larger font size
    overall_avg = df[col15].mean()
    ax.axhline(y=overall_avg, color='red', linestyle='--', alpha=0.7)
    
    # Find appropriate empty space for the average label
    right_edge = ax.get_xlim()[1]
    y_min, y_max = ax.get_ylim()
    y_position = y_max * 0.9  # Position at 90% of chart height
    
    ax.text(right_edge * 0.8, y_position, f'Average: {overall_avg:.2f}', 
            color='red', ha='right', va='center', fontsize=14, fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='red', alpha=0.7, pad=5, boxstyle='round'))

    # Enhance grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust legend with better positioning
    ax.legend(title='Employee Group', loc='upper right')
    # Extend y-axis to provide more space for labels
    extend_y_limits(ax, 0.2)
    # Extend y-axis to provide more space for labels
    extend_y_limits(ax, 0.2)
    
    # Rotate x-axis labels for Education dashboard
    if name == "Education":
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
    # Add more padding around figure
    plt.tight_layout(pad=3.0)
    
    # Use full width in Streamlit
    st.pyplot(fig, use_container_width=True)
    
    # Generate a unique key for this chart's recommendation
    rec_key = get_recommendation_key(name, "Average Residency")
    
    # Get any existing recommendation
    existing_rec = st.session_state.custom_recommendations.get(rec_key, "")
    
    # Create text area for custom recommendation
    custom_rec = st.text_area(
        "Recommendation:",
        value=existing_rec,
        height=150,
        key=f"text_{rec_key}"
    )
    
    # Add a save button
    if st.button("Save", key=f"save_{rec_key}"):
        save_recommendation(rec_key, custom_rec)

def create_infant_attrition_chart(df, name):
    """Create the infant attrition chart."""
    fig, ax = setup_chart_style()
    
    # Get the 16th column (index 15)
    col16 = df.columns[15]

    # Create a DataFrame for the chart with a shorter column name for display
    infant_attrition_data = pd.DataFrame({
        'Category': df['Category'],
        'Infant Attrition': df[col16] * 100  # Convert to percentage
    })

    # Using seaborn barplot (fixed deprecation warning)
    # Generate a palette with enough colors for all categories
    category_count = len(infant_attrition_data['Category'].unique())
    palette = sns.color_palette("Blues_d", category_count)    
    bars = sns.barplot(x='Category', y='Infant Attrition', data=infant_attrition_data, 
                      hue='Category', palette=palette, legend=False, ax=ax, width=0.7)

    ax.set_title(f'Infant Attrition Rate by {name}', pad=20)
    ax.set_xlabel(f'{name}', labelpad=15)
    ax.set_ylabel('Attrition Rate (%)', labelpad=15)    # Adding value labels dynamically inside or outside the bars
    base_fontsize = 16
    threshold = ax.get_ylim()[1] * 0.10

    for i, container in enumerate(ax.containers):
        for j, bar in enumerate(container):
            height = bar.get_height()
            if height == 0:
                continue
            
            label_text = f'{height:.1f}%'
            x_pos = bar.get_x() + bar.get_width() / 2

            if height < threshold:
                y_pos = height
                color = 'black'
                ax.text(x_pos, y_pos, label_text,
                        ha='center', va='bottom',
                        color=color,
                        fontweight='bold',
                        fontsize=base_fontsize,
                        xytext=(0, 5),
                        textcoords='offset points')
            else:
                y_pos = height / 2
                color = 'white'
                ax.text(x_pos, y_pos, label_text,
                        ha='center', va='center',
                        color=color,
                        fontweight='bold',
                        fontsize=base_fontsize)    # Add a horizontal line for the average with larger font size
    avg_attrition = infant_attrition_data['Infant Attrition'].mean()
    ax.axhline(y=avg_attrition, color='red', linestyle='--', alpha=0.7)
      # Find appropriate empty space for the average label
    # Move text to upper right corner of the chart instead of directly on the line
    right_edge = ax.get_xlim()[1]
    y_min, y_max = ax.get_ylim()
    y_position = y_max * 0.9  # Position at 90% of chart height
    
    ax.text(right_edge * 0.8, y_position, f'Average: {avg_attrition:.1f}%', 
            color='red', ha='right', va='center', fontsize=14, fontweight='bold',
            bbox=dict(facecolor='white', edgecolor='red', alpha=0.7, pad=5, boxstyle='round'))
    
    # Enhance grid for better readability
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    # Extend y-axis to provide more space for labels
    extend_y_limits(ax, 0.2)
    # Extend y-axis to provide more space for labels
    extend_y_limits(ax, 0.2)
    
    # Rotate x-axis labels for Education dashboard
    if name == "Education":
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
    # Add more padding around figure
    plt.tight_layout(pad=3.0)
    
    # Use full width in Streamlit
    st.pyplot(fig, use_container_width=True)
    
    # Generate a unique key for this chart's recommendation
    rec_key = get_recommendation_key(name, "Infant Attrition")
    
    # Get any existing recommendation
    existing_rec = st.session_state.custom_recommendations.get(rec_key, "")
    
    # Create text area for custom recommendation
    custom_rec = st.text_area(
        "Recommendation:",
        value=existing_rec,
        height=150,
        key=f"text_{rec_key}"
    )
    
    # Add a save button
    if st.button("Save", key=f"save_{rec_key}"):
        save_recommendation(rec_key, custom_rec)

if __name__ == "__main__":
    main()
