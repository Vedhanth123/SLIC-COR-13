"""
Script to add chart observations to streamlit_dashboard_simple.py
"""
import re

def add_observations():
    # Read the current file
    with open('streamlit_dashboard_simple.py', 'r') as file:
        content = file.read()

    # Define the observations for remaining charts
    attrition_observation = """    # Use full width in Streamlit
    st.pyplot(fig, use_container_width=True)
    
    # Add observations below the chart
    st.markdown(f\"\"\"
    <div style="background-color: #f5f7ff; border-left: 4px solid #0A2472; padding: 10px; border-radius: 5px; margin: 10px 0;">
        <h4 style="color: #0A2472; margin-top: 0;">Attrition Count Insights:</h4>
        <ul style="margin-bottom: 0;">
            <li>This chart shows the number of employees who have left the company across {name} categories.</li>
            <li>Percentages inside bars represent attrition rate relative to the total headcount in each category.</li>
            <li>Higher attrition in certain segments may indicate areas requiring attention for retention strategies.</li>
            <li>Understanding attrition patterns by {name} helps prioritize targeted retention initiatives.</li>
        </ul>
    </div>
    \"\"\", unsafe_allow_html=True)"""

    residency_observation = """    # Use full width in Streamlit
    st.pyplot(fig, use_container_width=True)
    
    # Add observations below the chart
    st.markdown(f\"\"\"
    <div style="background-color: #f5f7ff; border-left: 4px solid #0A2472; padding: 10px; border-radius: 5px; margin: 10px 0;">
        <h4 style="color: #0A2472; margin-top: 0;">Average Residency Insights:</h4>
        <ul style="margin-bottom: 0;">
            <li>This chart compares employment tenure (in months) across {name} categories.</li>
            <li>The comparison between all employees and top performers reveals retention patterns of high achievers.</li>
            <li>Longer tenure among top performers indicates a positive correlation between experience and performance.</li>
            <li>Categories with significant differences between groups may benefit from targeted development programs.</li>
        </ul>
    </div>
    \"\"\", unsafe_allow_html=True)"""

    infant_attrition_observation = """    # Use full width in Streamlit
    st.pyplot(fig, use_container_width=True)
    
    # Add observations below the chart
    st.markdown(f\"\"\"
    <div style="background-color: #f5f7ff; border-left: 4px solid #0A2472; padding: 10px; border-radius: 5px; margin: 10px 0;">
        <h4 style="color: #0A2472; margin-top: 0;">Infant Attrition Insights:</h4>
        <ul style="margin-bottom: 0;">
            <li>This chart displays early-stage employee attrition rates across {name} categories.</li>
            <li>Higher percentages indicate greater challenges with onboarding and early engagement.</li>
            <li>The red line shows the organization-wide average for context and comparison.</li>
            <li>Categories with higher infant attrition rates may need improved orientation and early support programs.</li>
        </ul>
    </div>
    \"\"\", unsafe_allow_html=True)"""

    # Define patterns to locate the exact insertion points
    attrition_pattern = r"def create_average_residency_chart\(df, name\):"
    residency_pattern = r"def create_infant_attrition_chart\(df, name\):"
    infant_pattern = r"if __name__ == \"__main__\":"

    # Replace specific patterns with the observations
    content = re.sub(r"# Use full width in Streamlit\n    st\.pyplot\(fig, use_container_width=True\)\n\ndef create_average_residency_chart", 
                    attrition_observation + "\n\ndef create_average_residency_chart", content)
    
    content = re.sub(r"# Use full width in Streamlit\n    st\.pyplot\(fig, use_container_width=True\)\n\ndef create_infant_attrition_chart", 
                    residency_observation + "\n\ndef create_infant_attrition_chart", content)
    
    content = re.sub(r"# Use full width in Streamlit\n    st\.pyplot\(fig, use_container_width=True\)\n\nif __name__ == \"__main__\":", 
                    infant_attrition_observation + "\n\nif __name__ == \"__main__\":", content)

    # Write the updated content back to the file
    with open('streamlit_dashboard_simple.py', 'w') as file:
        file.write(content)
    
    print("Added observations to all charts successfully!")

if __name__ == "__main__":
    add_observations()
