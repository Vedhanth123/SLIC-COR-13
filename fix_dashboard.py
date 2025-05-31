import re

# Read the file
with open('streamlit_dashboard_simple.py', 'r') as file:
    content = file.read()

# Fix indentation in barplots with 2 spaces at the beginning
content = content.replace('      bars', '    bars')

# Fix CAR2CATPO color palette line
content = content.replace('category_count = len(ratio_data[\'Category\'].unique())      palette', 'category_count = len(ratio_data[\'Category\'].unique())\n    palette')

# Fix infant attrition color palette line
content = content.replace('category_count = len(infant_attrition_data[\'Category\'].unique())      palette', 'category_count = len(infant_attrition_data[\'Category\'].unique())\n    palette')

# Add y-axis extension to all chart functions
chart_funcs = ['create_distribution_chart', 'create_kpi_performance_chart', 'create_performance_multiple_chart', 
               'create_top_bottom_performers_chart', 'create_time_to_first_sale_chart', 
               'create_car2catpo_ratio_chart', 'create_attrition_count_chart', 
               'create_average_residency_chart', 'create_infant_attrition_chart']

for func in chart_funcs:
    # Add extend_y_limits call before the tight_layout
    pattern = fr'(def {func}.*?# Enhance grid for better readability.*?)(\s+# Rotate x-axis labels for Education dashboard)'
    replacement = r'\1\n    # Extend y-axis to provide more space for labels\n    extend_y_limits(ax, 0.2)\2'
    content = re.sub(pattern, replacement, content, flags=re.DOTALL)

# Write the fixed content back
with open('streamlit_dashboard_simple.py', 'w') as file:
    file.write(content)

print('Fixed indentation and palette issues in the file and added y-axis extension.')
