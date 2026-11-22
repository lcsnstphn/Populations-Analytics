import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

def plot_population_distribution(df):
    fig = px.histogram(
        df, x='age', y='population', color='gender',
        title='Population Distribution by Age and Gender',
        marginal='box', barmode='group',
        labels={'age': 'Age', 'population': 'Total Population'}
    )
    fig.update_layout(template='plotly_white', legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1))
    return fig

def plot_gender_analysis(df):
    gender_df = df.groupby('gender')['population'].sum().reset_index()
    fig = px.pie(
        gender_df, values='population', names='gender',
        title='Gender Breakdown of Overall Population',
        hole=0.4, color_discrete_sequence=px.colors.qualitative.Pastel
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def plot_youth_analysis(df):
    df_youth = df.copy()
    df_youth['age_group'] = pd.cut(
        df_youth['age'], 
        bins=[0, 14, 24, 64, 100], 
        labels=['Children (0-14)', 'Youth (15-24)', 'Adults (25-64)', 'Seniors (65+)']
    )
    youth_summary = df_youth.groupby('age_group')['population'].sum().reset_index()
    fig = px.bar(
        youth_summary, x='age_group', y='population',
        color='age_group', title='Population Distribution by Life Stage (Youth Highlighted)',
        labels={'population': 'Total Population', 'age_group': 'Age Group'}
    )
    fig.update_layout(showlegend=False, template='plotly_white')
    return fig
