# HDFC Analysis Dashboard

## Overview
This repository contains a scrollable interactive dashboard for HDFC data visualization across multiple categories:
- Gender
- Education
- Experience
- Age

## Features
- Interactive tabs for each category
- Scrollable interface
- 9 different visualizations per category
- Dynamic charts with tooltips
- Rotated labels for better readability (especially for Education)
- Responsive layout
- Custom recommendations for each chart (NEW!)

## Prerequisites
- Python 3.9+ with pip
- Required packages: pandas, matplotlib, seaborn, streamlit

## How to Run

### Option 1: Using the batch file
1. Simply double-click on `run_dashboard.bat`
2. The dashboard will open in your default web browser

### Option 2: Using Command Line
1. Open a command prompt
2. Navigate to the project directory:
```
cd path\to\Ver6-COR-13-HDFC
```
3. Activate the virtual environment:
```
.\env\Scripts\activate
```
4. Run the dashboard:
```
streamlit run streamlit_dashboard.py
```

## Navigation
- Use the tabs at the top to switch between different categories
- Scroll down to see all visualizations
- Hover over data points to see more details
- Each visualization can be expanded to full screen by clicking the "⋮" menu in the top-right corner of each chart
- Download data or images using the export functionality in each chart's menu

## Custom Recommendations Feature

The Simple Dashboard now includes a custom recommendation system that allows you to:

1. Write your own analysis and insights for each chart
2. Save recommendations per chart and category (e.g., separate recommendations for Gender-Distribution vs Age-Distribution)
3. View auto-generated insights in collapsible sections
4. Export all recommendations to a comprehensive report (using `export_recommendations.py`)

### Using Custom Recommendations

1. Select a category (Gender, Education, Experience, Age)
2. For each chart, you'll find a "Your Custom Recommendation" section with a text area
3. Enter your analysis and click "Save Recommendation"
4. Your recommendations are automatically saved to `custom_recommendations.json`
5. Auto-generated insights are still available under the "View Auto-Generated Insights" dropdown

### Advantages of Custom Recommendations

- Personalize analysis based on domain knowledge
- Create presentation-ready explanations
- Build institutional knowledge over time
- Maintain consistency across reports
- Focus on insights that matter most to your stakeholders

## Advantages Over Static Images
- Interactive data exploration
- Ability to view specific data points
- Responsive layout that works on any screen size
- No image clutter
- Easier to share and distribute