#!/usr/bin/env python3
"""
Export Custom Recommendations to a Formatted Report

This script reads the custom recommendations saved in custom_recommendations.json
and generates a formatted report in HTML format. The report can be opened in any
web browser and is also printer-friendly.

Usage:
    python export_recommendations.py [output_file]

Arguments:
    output_file: Optional. Path to the output HTML file.
                 Default: 'hdfc_recommendations_report.html'
"""

import json
import os
import sys
import datetime
import pandas as pd

# Define chart titles and categories for better formatting
CHART_TITLES = {
    "Distribution": "Distribution by Cohort",
    "KPI Performance": "KPI Performance CAP LRM",
    "Performance Multiple": "Performance Multiple",
    "Top vs Bottom Performers": "Top vs Bottom Performers",
    "Time to First Sale": "Time to Make First Sale",
    "CAR2CATPO Ratio": "CAR2CATPO Ratio",
    "Attrition Count": "Employee Attrition",
    "Average Residency": "Employment Tenure",
    "Infant Attrition": "Infant Attrition Rate"
}

CATEGORIES = ["Gender", "Education", "Experience", "Age"]

def load_recommendations():
    """Load recommendations from the JSON file."""
    try:
        with open('custom_recommendations.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print("No recommendations file found. Please create some recommendations first.")
        return {}
    except json.JSONDecodeError:
        print("Error reading recommendations file. The file may be corrupted.")
        return {}

def get_excel_data():
    """Load data from the Excel file for additional insights."""
    try:
        data = {}
        for category in CATEGORIES:
            data[category] = pd.read_excel('HDFC_modified.xlsx', sheet_name=category)
        return data
    except Exception as e:
        print(f"Error loading Excel data: {e}")
        return {}

def generate_report(recommendations, output_file="hdfc_recommendations_report.html"):
    """Generate an HTML report from recommendations."""
    if not recommendations:
        print("No recommendations to export.")
        return False

    # Get current date and time for the report
    now = datetime.datetime.now()
    date_str = now.strftime("%B %d, %Y")
    time_str = now.strftime("%I:%M %p")

    # Start building HTML content
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HDFC Analysis Recommendations Report</title>
    <style>
        @page {{ size: letter; margin: 2cm; }}
        body {{
            font-family: 'Segoe UI', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            color: #333;
            line-height: 1.6;
        }}
        .header {{
            text-align: center;
            margin-bottom: 40px;
            border-bottom: 2px solid #0047AB;
            padding-bottom: 20px;
        }}
        .header h1 {{
            color: #0047AB;
            margin-bottom: 5px;
        }}
        .header p {{
            color: #666;
            margin: 5px 0;
        }}
        .section {{
            margin-bottom: 30px;
            break-inside: avoid;
        }}
        .section h2 {{
            color: #0A2472;
            border-bottom: 1px solid #ddd;
            padding-bottom: 5px;
        }}
        .chart-section {{
            margin-bottom: 40px;
            padding: 15px;
            background-color: #f9f9fa;
            border-radius: 5px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            break-inside: avoid;
        }}
        .chart-section h3 {{
            color: #0A2472;
            margin-top: 10px;
        }}        .recommendation {{
            background-color: #f5f7ff;
            border-left: 4px solid #0A2472;
            padding: 15px;
            margin-top: 10px;
            white-space: pre-wrap;
        }}
        .meta-info {{
            font-size: 12px;
            color: #666;
            margin-top: 10px;
            text-align: right;
            font-style: italic;
        }}
        .chart-count {{
            font-weight: normal;
            font-size: 14px;
            color: #666;
        }}
        .no-recommendation {{
            font-style: italic;
            color: #999;
        }}
        .toc {{
            margin-bottom: 40px;
        }}
        .toc a {{
            text-decoration: none;
            color: #0A2472;
        }}
        .toc a:hover {{
            text-decoration: underline;
        }}
        .toc-list {{
            list-style-type: none;
            padding-left: 20px;
        }}
        .page-break {{
            page-break-after: always;
        }}
        @media print {{
            .no-print {{
                display: none;
            }}
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>HDFC Analysis Recommendations</h1>
        <p>Executive Summary Report</p>
        <p>Generated on {date_str} at {time_str}</p>
    </div>
    
    <div class="toc">
        <h2>Table of Contents</h2>
        <ul>
"""

    # Add table of contents
    for category in CATEGORIES:
        html_content += f'            <li><a href="#{category.lower()}">{category}</a>\n'
        html_content += '                <ul class="toc-list">\n'
        
        for chart in CHART_TITLES.keys():
            chart_key = f"{category}_{chart}"
            if chart_key in recommendations:
                html_content += f'                    <li><a href="#{chart_key.replace(" ", "_")}">{CHART_TITLES[chart]}</a></li>\n'
        
        html_content += '                </ul>\n'
        html_content += '            </li>\n'

    html_content += """        </ul>
    </div>
    
    <div class="page-break"></div>
"""

    # Add recommendations organized by category and chart
    for category in CATEGORIES:
        html_content += f'''
    <div class="section" id="{category.lower()}">
        <h2>{category} Analysis</h2>
'''
        
        # Count recommendations for this category
        category_recs = [r for r in recommendations.keys() if r.startswith(f"{category}_")]
        if not category_recs:
            html_content += f'        <p class="no-recommendation">No recommendations have been created for {category} charts.</p>\n'
            continue

        for chart in CHART_TITLES.keys():
            chart_key = f"{category}_{chart}"
            chart_id = chart_key.replace(" ", "_")
            
            if chart_key in recommendations:
                html_content += f'''        <div class="chart-section" id="{chart_id}">
            <h3>{CHART_TITLES[chart]}</h3>
            <div class="recommendation">
{recommendations[chart_key]}
            </div>
            <div class="meta-info">
                Category: {category}
            </div>
        </div>
'''
        
        html_content += '    </div>\n'
        
        # Add page break after each category except the last one
        if category != CATEGORIES[-1]:
            html_content += '    <div class="page-break"></div>\n'

    # Close the HTML document
    html_content += '''
    <div class="meta-info">
        <p>End of report. Generated from HDFC Analysis Dashboard.</p>
    </div>

    <script class="no-print">
        // Add click handlers for easy navigation
        document.addEventListener('DOMContentLoaded', function() {
            const tocLinks = document.querySelectorAll('.toc a');
            tocLinks.forEach(link => {
                link.addEventListener('click', function(e) {
                    e.preventDefault();
                    const targetId = this.getAttribute('href').substring(1);
                    const targetElement = document.getElementById(targetId);
                    if (targetElement) {
                        targetElement.scrollIntoView({ behavior: 'smooth' });
                    }
                });
            });
        });
    </script>
</body>
</html>
'''

    # Write the HTML content to the output file
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        print(f"Report generated successfully: {os.path.abspath(output_file)}")
        return True
    except Exception as e:
        print(f"Error generating report: {e}")
        return False

def main():
    # Get output filename from command line or use default
    output_file = sys.argv[1] if len(sys.argv) > 1 else "hdfc_recommendations_report.html"
    
    print("HDFC Recommendations Export Tool")
    print("--------------------------------")
    print("Loading recommendations...")
    recommendations = load_recommendations()
    
    if not recommendations:
        print("\nNo recommendations found. Please create some recommendations first.")
        print("To create recommendations:")
        print("1. Run the dashboard using 'run_dashboard.bat'")
        print("2. Select option 2 for Simple Dashboard with Custom Recommendations")
        print("3. Add recommendations for each chart")
        return
    
    print(f"Found {len(recommendations)} recommendation(s)")
    print("\nGenerating report...")
    if generate_report(recommendations, output_file):
        print("\nReport generated successfully!")
        print(f"File saved to: {os.path.abspath(output_file)}")
        print("\nYou can open this file in any web browser to view and print the report.")
    else:
        print("\nFailed to generate report. Please check permissions and try again.")

if __name__ == "__main__":
    main()
