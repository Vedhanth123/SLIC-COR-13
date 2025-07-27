import streamlit as st
import pandas as pd
import plotly.express as px

class ChartGenerator:
    """A class responsible for creating all dashboard visualizations using Plotly."""

    def _get_base_fig(self, df, x, y, color=None, barmode='group', text_auto=True, **kwargs):
        """Creates a base Plotly bar chart with common styling."""
        fig = px.bar(df, x=x, y=y, color=color, barmode=barmode, text_auto=text_auto, **kwargs)
        
        fig.update_layout(
            title_x=0.05,
            title_font=dict(size=24),         
            xaxis_title_font=dict(size=18), 
            yaxis_title_font=dict(size=18),   
            legend_font=dict(size=16),  
            font=dict(family="Arial Black", size=14, color="black"),
            plot_bgcolor='rgba(249,249,249,1)',
            paper_bgcolor='rgba(255,255,255,1)',
            yaxis=dict(gridcolor='rgba(220,220,220,0.5)'),
            legend_title_text='',
            bargap=0.1,
            xaxis_tickangle=-45,
        )
        return fig

    def _add_average_line(self, fig, value, label_text):
        """Adds a styled average line and annotation to a Plotly figure."""
        fig.add_hline(
            y=value, line_dash="dash", line_color="red",
            annotation_text=label_text,
            annotation_position="bottom right",
            annotation_font=dict(size=14, color="red")
        )

    def create_distribution_chart(self, df, name):
        cols = df.columns[:3]
        df_melted = pd.melt(df, id_vars=[cols[0]], value_vars=[cols[1], cols[2]], var_name='Metric', value_name='Count')
        
        fig = self._get_base_fig(df_melted, x='Category', y='Count', color='Metric', title=f'{name} Distribution by Cohort')
        fig.update_traces(texttemplate='%{y:.0f}', textposition='inside')
        st.plotly_chart(fig, use_container_width=True)

    def create_kpi_performance_chart(self, df, name):
        kpi_data = pd.DataFrame({'Category': df['Category'], 'Achievement %': df[df.columns[3]] * 100})
        
        fig = self._get_base_fig(kpi_data, x='Category', y='Achievement %', title=f'KPI 1 Performance by {name} CAP LRM', color_discrete_sequence=['#ff7f0e'])
        fig.update_traces(texttemplate='%{y:.0f}%', textposition='auto')
        st.plotly_chart(fig, use_container_width=True)

    def create_performance_multiple_chart(self, df, name):
        perf_data = pd.DataFrame({'Category': df['Category'], 'Multiple': df[df.columns[6]]})
        
        fig = self._get_base_fig(perf_data, x='Category', y='Multiple', title=f'Performance Multiple by {name}', color_discrete_sequence=['#2ca02c'])
        fig.update_traces(texttemplate='%{y:.1f}x', textposition='auto')
        st.plotly_chart(fig, use_container_width=True)

    def create_top_bottom_performers_chart(self, df, name):
        performer_data = pd.DataFrame({'Category': df['Category'], 'Top 10% (KPI 1)': df[df.columns[4]], 'Bottom 10% (KPI 1)': df[df.columns[5]]})
        all_performers = pd.melt(performer_data, id_vars=['Category'], var_name='Performance', value_name='Value')
        
        fig = self._get_base_fig(all_performers, x='Category', y='Value', color='Performance', title=f'Top vs Bottom Performers by {name}')
        fig.update_traces(texttemplate='%{y:.1f}', textposition='auto')
        st.plotly_chart(fig, use_container_width=True)

    def create_time_to_first_sale_chart(self, df, name):
        col = df.columns[7]
        first_sale_data = pd.DataFrame({'Category': df['Category'], 'Time to First Sale': df[col]})
        
        fig = self._get_base_fig(first_sale_data, x='Category', y='Time to First Sale (in months)', title=f'Time to Make First Sale (in months) by {name}', text_auto='.2f')
        avg_time = df[col].mean()
        # self._add_average_line(fig, avg_time, f'Average: {avg_time:.2f} months')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    def create_car2catpo_ratio_chart(self, df, name):
        col = df.columns[8]
        ratio_data = pd.DataFrame({'Category': df['Category'], 'CAR2CATPO Ratio': df[col]})

        fig = self._get_base_fig(ratio_data, x='Category', y='CAR2CATPO Ratio', title=f'CAR2CATPO Ratio by {name}', text_auto='.2f')
        avg_ratio = df[col].mean()
        # self._add_average_line(fig, avg_ratio, f'Average: {avg_ratio:.2f}')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    def create_attrition_count_chart(self, df, name):
        col14 = df.columns[9]
        attrition_data = pd.DataFrame({'Category': df['Category'], 'Attrited Employees': df[col14]})
        total_per_category = df['CAP LRM cohort'].values
        attrition_data['rate'] = (attrition_data['Attrited Employees'] / total_per_category) * 100
        # Create a custom text string for the label, using <br> for a new line in HTML
        attrition_data['text_label'] = [f"{count}<br>({rate:.1f}%)" for count, rate in zip(attrition_data['Attrited Employees'], attrition_data['rate'])]
        
        fig = self._get_base_fig(attrition_data, x='Category', y='Attrited Employees', title=f'Employee Attrition by {name}', text_auto=False)
        fig.update_traces(text=attrition_data['text_label'], textposition='inside')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    def create_average_residency_chart(self, df, name):
        col15, col16 = df.columns[10], df.columns[11]
        residency_data = pd.DataFrame({'Category': df['Category'], "All Employees": df[col15], "Top 100 Performers": df[col16]})
        residency_melted = pd.melt(residency_data, id_vars=['Category'], var_name='Employee Group', value_name='Average Residency')

        fig = self._get_base_fig(residency_melted, x='Category', y='Average Residency', color='Employee Group', title=f'Employment Tenure by {name}', text_auto='.2f')
        overall_avg = df[col15].mean()
        # self._add_average_line(fig, overall_avg, f'Average: {overall_avg:.2f}')
        st.plotly_chart(fig, use_container_width=True)

    def create_infant_attrition_chart(self, df, name):
        col = df.columns[12]
        infant_attrition_data = pd.DataFrame({'Category': df['Category'], 'Infant Attrition': df[col] * 100})

        fig = self._get_base_fig(infant_attrition_data, x='Category', y='Infant Attrition', title=f'Infant Attrition Rate by {name}', text_auto='.1f')
        fig.update_traces(texttemplate='%{y:.1f}%')
        avg_attrition = infant_attrition_data['Infant Attrition'].mean()
        # self._add_average_line(fig, avg_attrition, f'Average: {avg_attrition:.1f}%')
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    def create_retention_chart(self, df, name):
        cols = { "Retention at CAP 3": df.columns[18], "Retention at CAP 6": df.columns[19], "Retention at CAP 9": df.columns[20], "Retention at CAP 12": df.columns[21]}
        data = pd.DataFrame({'Category': df['Category']})
        for short_name, col_name in cols.items():
            data[short_name] = df[col_name] * 100
        melted = pd.melt(data, id_vars=['Category'], var_name='Retention Period', value_name='Retention Rate (%)')
        
        fig = self._get_base_fig(melted, x='Category', y='Retention Rate (%)', color='Retention Period', title=f'Employee Retention Rate by {name}', text_auto='.0f')
        fig.update_traces(texttemplate='%{y:.0f}%')
        st.plotly_chart(fig, use_container_width=True)

    def create_cost_of_hire_chart(self, df, name):
        cols = {"Cost of Wrong Hire": df.columns[16], "Cost of Back Fill": df.columns[17]}
        data = pd.DataFrame({'Category': df['Category']})
        for short_name, col_name in cols.items():
            data[short_name] = df[col_name]
        melted = pd.melt(data, id_vars=['Category'], var_name='Cost Type', value_name='Cost of Hire')
        
        fig = self._get_base_fig(melted, x='Category', y='Cost of Hire', color='Cost Type', title=f'Cost of Hire by {name}', text_auto='.1f')
        st.plotly_chart(fig, use_container_width=True)