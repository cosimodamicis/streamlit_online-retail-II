"""
Funzioni di visualizzazione per l'analisi retail luxury.
Usa Plotly per grafici interattivi ottimizzati per Streamlit.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List

# Configurazione tema colori
COLORS = {
    'VIP': '#FFD700',          # Gold
    'Loyal': '#C0C0C0',        # Silver
    'Regular': '#CD7F32',      # Bronze
    'Occasional': '#B87333',   # Copper
    'At Risk': '#808080',      # Gray
    'primary': '#1f77b4',      # Blue
    'secondary': '#2ca02c',    # Green
    'accent': '#d62728'        # Red
}

def create_customer_segment_pie(customer_stats: pd.DataFrame) -> go.Figure:
    """
    Crea pie chart della segmentazione clienti.
    
    Args:
        customer_stats (pd.DataFrame): DataFrame con statistiche clienti
        
    Returns:
        go.Figure: Plotly figure
    """
    segment_data = customer_stats['customer_segment'].value_counts()
    
    fig = px.pie(
        values=segment_data.values,
        names=segment_data.index,
        title="Segmentazione Cliente",
        color_discrete_map=COLORS,
    )
    
    fig.update_traces(
        textinfo='percent+label',
        hole=0.4,
    )
    
    fig.update_layout(
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_customer_value_distribution(customer_stats: pd.DataFrame) -> go.Figure:
    """
    Crea istogramma della distribuzione del customer value.
    
    Args:
        customer_stats (pd.DataFrame): DataFrame con statistiche clienti
        
    Returns:
        go.Figure: Plotly figure
    """
    fig = go.Figure()
    
    for segment in customer_stats['customer_segment'].unique():
        segment_data = customer_stats[
            customer_stats['customer_segment'] == segment
        ]['total_spend']
        
        fig.add_trace(go.Histogram(
            x=segment_data,
            name=segment,
            marker_color=COLORS.get(segment, COLORS['primary']),
            opacity=0.7,
        ))
    
    fig.update_layout(
        title="Distribuzione Customer Lifetime Value per Segmento",
        xaxis_title="Lifetime Value (€)",
        yaxis_title="Numero Clienti",
        barmode='overlay',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_seasonal_analysis(df: pd.DataFrame) -> go.Figure:
    """
    Crea radar chart dell'analisi stagionale.
    
    Args:
        df (pd.DataFrame): DataFrame con dati vendite
        
    Returns:
        go.Figure: Plotly figure
    """
    seasonal_data = df.groupby('season')['Total_Value'].agg(['sum', 'mean'])
    
    fig = go.Figure()
    
    # Revenue totale
    fig.add_trace(go.Scatterpolar(
        r=seasonal_data['sum'] / 1000,  # Converti in migliaia
        theta=seasonal_data.index,
        name='Revenue Totale (k€)',
        fill='toself',
        line_color=COLORS['primary']
    ))
    
    # Valore medio ordine
    fig.add_trace(go.Scatterpolar(
        r=seasonal_data['mean'],
        theta=seasonal_data.index,
        name='Valore Medio Ordine (€)',
        fill='toself',
        line_color=COLORS['secondary']
    ))
    
    fig.update_layout(
        title="Performance Stagionale",
        polar=dict(radialaxis=dict(showticklabels=True, ticks='')),
        showlegend=True
    )
    
    return fig

def create_product_performance(df: pd.DataFrame) -> go.Figure:
    """
    Crea grafico della performance prodotti.
    
    Args:
        df (pd.DataFrame): DataFrame con dati vendite
        
    Returns:
        go.Figure: Plotly figure
    """
    # Analisi per segmento prezzo
    price_segment_perf = df.groupby('price_segment').agg({
        'Total_Value': 'sum',
        'Quantity': 'sum',
        'Customer ID': 'nunique'
    }).round(2)
    
    # Normalizza i valori per il confronto
    for col in price_segment_perf.columns:
        price_segment_perf[f'{col}_norm'] = (
            price_segment_perf[col] / price_segment_perf[col].max()
        )
    
    fig = go.Figure()
    
    # Revenue
    fig.add_trace(go.Bar(
        name='Revenue',
        x=price_segment_perf.index,
        y=price_segment_perf['Total_Value'],
        marker_color=COLORS['primary']
    ))
    
    # Quantity
    fig.add_trace(go.Bar(
        name='Volume',
        x=price_segment_perf.index,
        y=price_segment_perf['Quantity'],
        marker_color=COLORS['secondary']
    ))
    
    fig.update_layout(
        title="Performance per Segmento Prezzo",
        xaxis_title="Segmento",
        yaxis_title="Valore",
        barmode='group',
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_ml_results_viz(ml_results: Dict) -> List[go.Figure]:
    """
    Crea visualizzazioni dei risultati ML.
    
    Args:
        ml_results (Dict): Dizionario con risultati ML
        
    Returns:
        List[go.Figure]: Lista di Plotly figures
    """
    figures = []
    
    # Performance modelli
    model_scores = {
        model: results['cv_scores_mean'] 
        for model, results in ml_results.items()
    }
    
    fig_performance = go.Figure(data=[
        go.Bar(
            x=list(model_scores.keys()),
            y=list(model_scores.values()),
            marker_color=COLORS['primary']
        )
    ])
    
    fig_performance.update_layout(
        title="Performance Modelli ML",
        xaxis_title="Modello",
        yaxis_title="Accuracy Score",
        showlegend=False
    )
    
    figures.append(fig_performance)
    
    # Feature importance
    if 'random_forest' in ml_results:
        feature_imp = ml_results['random_forest']['feature_importance']
        
        fig_importance = go.Figure(data=[
            go.Bar(
                x=list(feature_imp.values()),
                y=list(feature_imp.keys()),
                orientation='h',
                marker_color=COLORS['secondary']
            )
        ])
        
        fig_importance.update_layout(
            title="Importanza Features",
            xaxis_title="Importance Score",
            yaxis_title="Feature",
            showlegend=False
        )
        
        figures.append(fig_importance)
    
    return figures

def create_cohort_analysis(df: pd.DataFrame) -> go.Figure:
    """
    Crea heatmap dell'analisi per coorte.
    
    Args:
        df (pd.DataFrame): DataFrame con dati vendite
        
    Returns:
        go.Figure: Plotly figure
    """
    # Prepara dati coorte
    df['cohort_month'] = df['InvoiceDate'].dt.to_period('M')
    df['months_since_first'] = (
        df.groupby('Customer ID')['InvoiceDate']
        .transform('min')
        .dt.to_period('M')
    )
    
    cohort_data = df.groupby(['cohort_month', 'months_since_first'])[
        'Customer ID'
    ].nunique().reset_index()
    
    # Pivot per heatmap
    cohort_pivot = cohort_data.pivot(
        index='cohort_month',
        columns='months_since_first',
        values='Customer ID'
    )
    
    # Calcola retention rate
    cohort_sizes = cohort_pivot.iloc[:, 0]
    retention_pivot = cohort_pivot.div(cohort_sizes, axis=0) * 100
    
    fig = go.Figure(data=go.Heatmap(
        z=retention_pivot.values,
        x=['Month ' + str(i) for i in range(retention_pivot.shape[1])],
        y=[str(period) for period in retention_pivot.index],
        colorscale='RdYlBu_r',
        text=retention_pivot.values.round(1),
        texttemplate='%{text}%',
        textfont={"size": 10},
        colorbar=dict(title='Retention Rate %')
    ))
    
    fig.update_layout(
        title='Analisi Retention per Coorte',
        xaxis_title='Mesi dalla Prima Transazione',
        yaxis_title='Coorte',
        height=400
    )
    
    return fig