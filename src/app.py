"""
Streamlit dashboard per analisi luxury retail.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from analyzer import LuxuryRetailAnalyzer
import numpy as np
from plotly.subplots import make_subplots #added for advanced product insights
import os
from scipy import stats
from scipy.stats import pearsonr
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import PartialDependenceDisplay
from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt

from utils.preprocessing import consolidate_excel_sheets
from spark_utils import load_data_with_spark, perform_spark_analysis, render_spark_analysis



class LuxuryRetailDashboard:
    def __init__(self):
        """Inizializza la dashboard"""
        st.set_page_config(
            page_title="Luxury Retail Analytics",
            page_icon="💎",
            layout="wide"
        )
        
        # Inizializza session state
        if 'df' not in st.session_state:
            st.session_state.df = None
        if 'customer_stats' not in st.session_state:
            st.session_state.customer_stats = None
        if 'data_loaded' not in st.session_state:
            st.session_state.data_loaded = False

        # Inizializzazione parte di Spark
        if 'spark_df' not in st.session_state:
            st.session_state.spark_df = None
        if 'spark_results' not in st.session_state:
            st.session_state.spark_results = None
    def render_header(self):
        """Render dell'header"""
        st.title("💎 Luxury Retail Analytics")
        st.markdown("Dashboard per analisi dati retail nel settore lusso")
            
    def load_default_dataset(self):
        """Carica il dataset di default dalla cartella data"""
        try:
            default_path = os.path.join('data', 'online_retail_II.xlsx')

            # Usa la funzione per consolidare i dati da tutti i fogli
            df = consolidate_excel_sheets(default_path)

            # Passa il DataFrame consolidato all'analizzatore
            analyzer = LuxuryRetailAnalyzer(df)
            df, customer_stats = analyzer.process_data()

            # Salva in session state
            st.session_state.df = df
            st.session_state.customer_stats = customer_stats
            st.session_state.data_loaded = True

            st.success("Dataset di default consolidato e caricato con successo!")
        except Exception as e:
            st.error(f"Errore nel caricamento del dataset di default: {str(e)}")

            
    def load_custom_dataset(self, uploaded_file):
        """Carica il dataset da file caricato"""
        try:
            df = pd.read_excel(uploaded_file)
            analyzer = LuxuryRetailAnalyzer(df)
            df, customer_stats = analyzer.process_data()
            
            # Salva in session state
            st.session_state.df = df
            st.session_state.customer_stats = customer_stats
            st.session_state.data_loaded = True
            
            st.success("File caricato con successo!")
        except Exception as e:
            st.error(f"Errore nel caricamento del file: {str(e)}")
            
    def render_kpis(self, df, customer_stats):
        """Render dei KPI principali"""
        st.header("📊 Statistiche Base")
        
        total_revenue = df['Total_Value'].sum()
        avg_order = df.groupby('Invoice')['Total_Value'].sum().mean()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Revenue Totale",
                f"€{total_revenue:,.2f}"
            )
        
        with col2:
            st.metric(
                "Valore Medio Ordine",
                f"€{avg_order:,.2f}"
            )
            
        with col3:
            st.metric(
                "Clienti Totali",
                f"{len(customer_stats):,}"
            )
            
        with col4:
            vip_perc = (customer_stats['customer_segment'] == 'VIP').mean() * 100
            st.metric(
                "% Clienti VIP",
                f"{vip_perc:.1f}%"
            )
            
    def render_customer_analysis(self, customer_stats, df):
        """Render dell'analisi clienti"""
        st.header("👥 Analisi Cliente")
        st.write("Colonne presenti nel DataFrame:", df.columns)
        # Segmentazione clienti
        segment_dist = customer_stats['customer_segment'].value_counts().reset_index()
        segment_dist.columns = ['Segmento', 'Numero Clienti']
        total_customers = segment_dist['Numero Clienti'].sum()
        segment_dist['Percentuale'] = (segment_dist['Numero Clienti'] / total_customers * 100)

        fig_segments = go.Figure()
        fig_segments.add_trace(go.Bar(
            x=segment_dist['Segmento'],
            y=segment_dist['Numero Clienti'],
            text=[f"n: {n:,.0f}<br>({p:.1f}%)" for n, p in zip(
                segment_dist['Numero Clienti'], 
                segment_dist['Percentuale']
            )],
            textposition='auto',
        ))

        fig_segments.update_layout(
            title="Distribuzione Segmenti Cliente",
            xaxis_title="Segmento",
            yaxis_title="Numero Clienti",
            showlegend=False,
            height=400,
            xaxis={'categoryorder':'total descending'}  # Ordina le barre per valore decrescente
        )

        st.plotly_chart(fig_segments, use_container_width=True)
        
        # Customer Value Distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig_value = px.histogram(
                customer_stats,
                x='total_spend',
                nbins=50,
                title="Distribuzione Customer Value",
                labels={'total_spend': 'Valore Totale (€)', 'count': 'Numero Clienti (log)'}
            )
            # Aggiungi scala logaritmica per l'asse Y
            fig_value.update_layout(
                yaxis_type="log",  # Scala logaritmica sull'asse Y
                #xaxis_type="log"   # Scala logaritmica sull'asse X
            )
            st.plotly_chart(fig_value, use_container_width=True)
            
        with col2:
            # Average Order Value by Segment
            avg_by_segment = customer_stats.groupby('customer_segment')['avg_order_value'].mean()
            fig_avg = px.bar(
                x=avg_by_segment.index,
                y=avg_by_segment.values,
                title="Valore Medio Ordine per Segmento",
                labels={'x': 'Segmento', 'y': 'Valore Medio Ordine (€)'}
            )
            
            # Aggiungi annotazioni sopra le barre
            for i, value in enumerate(avg_by_segment.values):
                fig_avg.add_annotation(
                    x=avg_by_segment.index[i],
                    y=value,
                    text=f"€{value:,.2f}",
                    showarrow=False,
                    font=dict(size=12),
                    yshift=10  # Sposta leggermente il testo verso l'alto
                )
            
            st.plotly_chart(fig_avg, use_container_width=True)

        
        """Render dell'analisi segmenti"""
        # Prima tabella - sempre visibile
        st.subheader("Statistiche Descrittive Valore Speso per Segmento (per Cliente)")
        monetary_stats = customer_stats.groupby('customer_segment')['total_spend'].describe()
        segment_totals = customer_stats.groupby('customer_segment')['total_spend'].sum()
        monetary_stats.insert(1, 'Total', segment_totals)
        monetary_stats = monetary_stats.sort_values('Total', ascending=False).reset_index()
        
        for col in ['Total', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
            monetary_stats[col] = monetary_stats[col].apply(lambda x: f"€{x:,.2f}")
        
        st.dataframe(monetary_stats, use_container_width=True, hide_index=True)
        
        # Seconda tabella - sempre visibile
        st.subheader("Statistiche Descrittive Valore Ordini per Segmento (per Ordine)")
        orders_by_segment = df.merge(
            customer_stats[['customer_segment']], 
            left_on='Customer ID', 
            right_index=True
        )
        order_stats = orders_by_segment.groupby('customer_segment')['Total_Value'].describe()
        segment_order_totals = orders_by_segment.groupby('customer_segment')['Total_Value'].sum()
        order_stats.insert(1, 'Total', segment_order_totals)
        order_stats = order_stats.sort_values('mean', ascending=False).reset_index()
        
        for col in ['Total', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
            order_stats[col] = order_stats[col].apply(lambda x: f"€{x:,.2f}")
        
        st.dataframe(order_stats, use_container_width=True, hide_index=True)
        
        # Insight base - sempre visibile
        st.markdown("""
        **Confronto tra le tabelle:**
        - La prima tabella mostra il comportamento lifetime dei clienti (quanto spende mediamente un cliente di ogni segmento)
        - La seconda tabella mostra il comportamento per singolo ordine (quanto vale mediamente un ordine per ogni segmento)
        
        Questo confronto evidenzia che mentre i clienti Loyal potrebbero spendere di più nel loro lifetime totale,
        i clienti VIP tendono ad effettuare ordini di valore superiore ma potenzialmente meno frequenti.
        """)
    def render_rfm_analysis(self, customer_stats):
        
        # RFM Analysis
        st.subheader("📊 Analisi RFM")
        # 1. Radar Chart originale
        rfm_cols = ['r_score', 'f_score', 'm_score']
        rfm_means = customer_stats.groupby('customer_segment')[rfm_cols].mean()
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_rfm = go.Figure(data=[
                go.Scatterpolar(
                    r=row,
                    theta=['Recency', 'Frequency', 'Monetary'],
                    name=segment
                    # rimosso fill='toself'
                ) for segment, row in rfm_means.iterrows()
            ])
            
            fig_rfm.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[1, 4])),
                showlegend=True,
                title="RFM Profile per Segmento"
            )
            st.plotly_chart(fig_rfm, use_container_width=True)
        
        with col2:
            # Heatmap delle medie RFM per segmento
            fig_heatmap = px.imshow(
                rfm_means,
                labels=dict(x="Metrica RFM", y="Segmento", color="Score"),
                aspect="auto",
                title="RFM Heatmap per Segmento"
            )
            st.plotly_chart(fig_heatmap, use_container_width=True)
        
        # 2. Distribuzione componenti RFM per segmento
        st.subheader("Distribuzione Componenti RFM")
        
        metrics = {
            'recency': 'Recency (giorni)',
            'frequency': 'Frequency (ordini)',
            'total_spend': 'Monetary (€)'
        }
        
        tabs = st.tabs(list(metrics.values()))
        
        for tab, (metric, label) in zip(tabs, metrics.items()):
            with tab:
                fig_dist = px.box(
                    customer_stats,
                    x='customer_segment',
                    y=metric,
                    color='customer_segment',
                    title=f"Distribuzione {label} per Segmento",
                    points="all"  # mostra tutti i punti oltre al box plot
                )
                
                # Applica la scala logaritmica solo se il metric è 'total_spend'
                if metric == 'total_spend':
                    fig_dist.update_layout(
                        yaxis_type="log"
                    )
                
                fig_dist.update_layout(showlegend=False)
                st.plotly_chart(fig_dist, use_container_width=True)
        
        # 3. Tabella riassuntiva RFM
        st.subheader("Metriche RFM per Segmento")
        summary_stats = pd.DataFrame({
            'Segmento': rfm_means.index,
            'R Score': rfm_means['r_score'].round(2),
            'F Score': rfm_means['f_score'].round(2),
            'M Score': rfm_means['m_score'].round(2),
            'Score Medio': rfm_means.mean(axis=1).round(2)
        }).sort_values('Score Medio', ascending=False)
        
        # Versione semplice senza gradient
        st.dataframe(summary_stats, use_container_width=True)
    
    def render_segment_analysis(self, customer_stats, df):
        """Render dell'analisi segmenti"""
        # Prima tabella - sempre visibile
        st.subheader("Statistiche Descrittive Valore Speso per Segmento (per Cliente)")
        monetary_stats = customer_stats.groupby('customer_segment')['total_spend'].describe()
        segment_totals = customer_stats.groupby('customer_segment')['total_spend'].sum()
        monetary_stats.insert(1, 'Total', segment_totals)
        monetary_stats = monetary_stats.sort_values('Total', ascending=False).reset_index()
        
        for col in ['Total', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
            monetary_stats[col] = monetary_stats[col].apply(lambda x: f"€{x:,.2f}")
        
        st.dataframe(monetary_stats, use_container_width=True, hide_index=True)
        
        # Seconda tabella - sempre visibile
        st.subheader("Statistiche Descrittive Valore Ordini per Segmento (per Ordine)")
        orders_by_segment = df.merge(
            customer_stats[['customer_segment']], 
            left_on='Customer ID', 
            right_index=True
        )
        order_stats = orders_by_segment.groupby('customer_segment')['Total_Value'].describe()
        segment_order_totals = orders_by_segment.groupby('customer_segment')['Total_Value'].sum()
        order_stats.insert(1, 'Total', segment_order_totals)
        order_stats = order_stats.sort_values('mean', ascending=False).reset_index()
        
        for col in ['Total', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
            order_stats[col] = order_stats[col].apply(lambda x: f"€{x:,.2f}")
        
        st.dataframe(order_stats, use_container_width=True, hide_index=True)
        
        # Insight base - sempre visibile
        st.markdown("""
        **Confronto tra le tabelle:**
        - La prima tabella mostra il comportamento lifetime dei clienti (quanto spende mediamente un cliente di ogni segmento)
        - La seconda tabella mostra il comportamento per singolo ordine (quanto vale mediamente un ordine per ogni segmento)
        
        Questo confronto evidenzia che mentre i clienti Loyal potrebbero spendere di più nel loro lifetime totale,
        i clienti VIP tendono ad effettuare ordini di valore superiore ma potenzialmente meno frequenti.
        """)
        
        # Pulsante per analisi statistica
        if st.button("Mostra Analisi Statistica"):
            st.markdown("""
            ### Test Statistico delle Differenze tra Segmenti
            Utilizziamo il test di Kruskal-Wallis (ANOVA non parametrico) per validare le differenze osservate tra i segmenti, 
            in quanto i dati di spesa tipicamente violano l'assunzione di omoschedasticità richiesta dall'ANOVA parametrica.
            """)
            
            # Test statistici
            h_stat_lifetime, p_val_lifetime = stats.kruskal(
                *[group['total_spend'].values for name, group in customer_stats.groupby('customer_segment')]
            )
            h_stat_order, p_val_order = stats.kruskal(
                *[group['Total_Value'].values for name, group in orders_by_segment.groupby('customer_segment')]
            )
            
            results_df = pd.DataFrame({
                'Metrica': ['Lifetime Value', 'Order Value'],
                'H-statistic': [h_stat_lifetime, h_stat_order],
                'p-value': [p_val_lifetime, p_val_order],
                'Significativo': ['Sì' if p < 0.05 else 'No' for p in [p_val_lifetime, p_val_order]]
            })
            
            st.table(results_df.style.format({
                'H-statistic': '{:.2f}',
                'p-value': '{:.4f}'
            }))
            
            st.markdown("""
            **Interpretazione dei Risultati:**
            I test confermano che esistono differenze statisticamente significative (p < 0.05) sia nel lifetime value che nel valore degli ordini 
            tra i diversi segmenti di clienti. Questo supporta scientificamente la nostra segmentazione e conferma che i pattern di spesa 
            osservati non sono casuali ma riflettono reali differenze nel comportamento d'acquisto.
            """)
        
        # Selectbox per visualizzazioni
        plot_type = st.selectbox(
            "Scegli il tipo di visualizzazione",
            ["Box Plot", "Violin Plot", "Nessuna Visualizzazione"],
            index=2  # Default a "Nessuna Visualizzazione"
        )
        
        if plot_type == "Box Plot":
            st.subheader("Visualizzazione delle Differenze tra Segmenti")
            col1, col2 = st.columns(2)
            
            with col1:
                fig_lifetime = go.Figure()
                for segment in customer_stats['customer_segment'].unique():
                    fig_lifetime.add_trace(go.Box(
                        y=customer_stats[customer_stats['customer_segment'] == segment]['total_spend'],
                        name=segment,
                        boxpoints='outliers'
                    ))
                fig_lifetime.update_layout(
                    title="Distribuzione Lifetime Value per Segmento",
                    yaxis_title="Lifetime Value (€)",
                    showlegend=False,
                    height=500,
                    yaxis_type="log"
                )
                st.plotly_chart(fig_lifetime, use_container_width=True)
            
            with col2:
                fig_order = go.Figure()
                for segment in orders_by_segment['customer_segment'].unique():
                    fig_order.add_trace(go.Box(
                        y=orders_by_segment[orders_by_segment['customer_segment'] == segment]['Total_Value'],
                        name=segment,
                        boxpoints='outliers'
                    ))
                fig_order.update_layout(
                    title="Distribuzione Order Value per Segmento",
                    yaxis_title="Order Value (€)",
                    showlegend=False,
                    height=500,
                    yaxis_type="log"
                )
                st.plotly_chart(fig_order, use_container_width=True)
                
        elif plot_type == "Violin Plot":
            st.subheader("Distribuzione Dettagliata dei Valori")
            fig_violin = go.Figure()
            
            for segment in customer_stats['customer_segment'].unique():
                fig_violin.add_trace(go.Violin(
                    x=[f"{segment} - Lifetime" for _ in range(len(customer_stats[customer_stats['customer_segment'] == segment]))],
                    y=customer_stats[customer_stats['customer_segment'] == segment]['total_spend'],
                    name=f"{segment} - Lifetime",
                    side='positive',
                    meanline_visible=True,
                    box_visible=True
                ))
                fig_violin.add_trace(go.Violin(
                    x=[f"{segment} - Order" for _ in range(len(orders_by_segment[orders_by_segment['customer_segment'] == segment]))],
                    y=orders_by_segment[orders_by_segment['customer_segment'] == segment]['Total_Value'],
                    name=f"{segment} - Order",
                    side='negative',
                    meanline_visible=True,
                    box_visible=True
                ))
            
            fig_violin.update_layout(
                title="Confronto Distribuzioni: Lifetime Value vs Order Value",
                yaxis_title="Valore (€)",
                violinmode='overlay',
                height=600,
                yaxis_type="log"
            )
            st.plotly_chart(fig_violin, use_container_width=True)
    
    def render_advanced_product_analysis(self, df, customer_stats):
        """
        Render dell'analisi prodotti avanzata con approfondimenti dettagliati.
        
        Args:
            df (pd.DataFrame): DataFrame con dati vendite
            customer_stats (pd.DataFrame): DataFrame con statistiche clienti
        """
        st.header("🔍 Analisi Prodotti Avanzata")
        
        # 1. Stagionalità per segmento di prodotto
        st.subheader("1. Stagionalità per Segmento di Prezzo")

        # Calcolo delle metriche stagionali per segmento
        seasonal_segment_perf = df.groupby(['price_segment', 'season'])['Total_Value'].agg([
            'sum',  # Revenue totale
            'mean',  # Revenue medio
            'count'  # Numero di transazioni
        ]).reset_index()

        # Pivot per la heatmap (Revenue totale)
        pivot_seasonal = seasonal_segment_perf.pivot(
            index='price_segment', 
            columns='season', 
            values='sum'
        )

        # Creazione della heatmap con annotazioni numeriche
        fig_seasonal = px.imshow(
            pivot_seasonal, 
            labels=dict(x="Stagione", y="Segmento Prezzo", color="Revenue (€)"),
            title="Revenue per Segmento e Stagione",
            text_auto='.2s'  # Aggiunge annotazioni automatiche con formato ridotto
        )

        fig_seasonal.update_layout(
            coloraxis_colorbar=dict(
                title="Revenue (€)",
                tickformat=".2s"  # Formatta i numeri con suffissi (K, M, etc.)
            ),
            font=dict(size=12),
        )

        st.plotly_chart(fig_seasonal, use_container_width=True)

        # Aggiunta di una tabella riepilogativa con valori di Revenue
        st.subheader("Tabella Riepilogativa: Revenue per Segmento e Stagione")

        # Tabella pivot con valori riepilogativi
        pivot_table = seasonal_segment_perf.pivot_table(
            index='price_segment', 
            columns='season', 
            values='sum', 
            aggfunc='sum'
        ).fillna(0)  # Sostituisci eventuali valori NaN con 0

        # Formattazione dei valori in euro con separatore delle migliaia
        pivot_table_display = pivot_table.applymap(lambda x: f"€{x:,.2f}")

        # Mostra la tabella riepilogativa
        st.dataframe(pivot_table_display, use_container_width=True, hide_index=False)

        # Spiegazione aggiuntiva per interpretare il grafico e la tabella
        st.markdown("""
        **Interpretazione del Grafico e della Tabella**
        - La heatmap mostra il revenue totale per ciascun segmento di prodotto e stagione.
        - Le annotazioni numeriche sul grafico rappresentano i valori esatti di revenue.
        - La tabella riepilogativa sottostante fornisce una visione dettagliata e formattata dei dati.
        - Utilizza queste informazioni per identificare i segmenti e le stagioni che generano il maggiore impatto sul revenue complessivo.
        """)
        
        # Analisi statistica delle differenze stagionali
        st.markdown("**Test Statistico Stagionalità**")
        seasonal_tests = []
        for segment in df['price_segment'].unique():
            segment_df = df[df['price_segment'] == segment]
            seasonal_groups = [
                group['Total_Value'].values 
                for name, group in segment_df.groupby('season')
            ]
            
            if len(seasonal_groups) > 1:
                h_stat, p_val = stats.kruskal(*seasonal_groups)
                seasonal_tests.append({
                    'Segmento': segment,
                    'H-statistic': h_stat,
                    'p-value': p_val,
                    'Significativo': 'Sì' if p_val < 0.05 else 'No'
                })
        
        st.dataframe(pd.DataFrame(seasonal_tests), use_container_width=True, hide_index=True)
        

        # Calcolo del coefficiente di correlazione di Pearson
        pearson_correlations = []

        # Calcolo delle serie delle vendite per ciascun segmento e per le stagioni
        seasonal_totals = seasonal_segment_perf.groupby('season')['sum'].sum()

        for segment in df['price_segment'].unique():
            # Serie delle vendite per il segmento corrente
            segment_sales = seasonal_segment_perf.loc[
                seasonal_segment_perf['price_segment'] == segment
            ].set_index('season')['sum']

            # Controlla che le serie abbiano esattamente quattro valori
            if len(segment_sales) == 4 and len(seasonal_totals) == 4:
                # Calcola la correlazione
                rho, p_value = pearsonr(segment_sales.values, seasonal_totals.values)
                pearson_correlations.append({
                    'Segmento': segment,
                    'Rho': rho,
                    'p-value': p_value
                })
            else:
                pearson_correlations.append({
                    'Segmento': segment,
                    'Rho': None,
                    'p-value': None
                })

        # Mostra i risultati in una tabella
        pearson_df = pd.DataFrame(pearson_correlations)
        st.subheader("Correlazione Segmenti-Stagionalità")
        st.dataframe(pearson_df, use_container_width=True, hide_index=True)

        st.markdown("""
            ### Interpretazione Statistica e Business dei Pattern Stagionali

            #### 1. Significatività Stagionale
            - Tutti i segmenti mostrano differenze stagionali statisticamente significative (p-value = 0), indicando che la stagionalità è un fattore determinante per tutti i livelli di prezzo.

            #### 2. Pattern di Stagionalità
            - **Dominanza dell'Autunno**: Tutti i segmenti mostrano il picco di revenue in autunno, rappresentando circa il 35% del fatturato annuale per ogni segmento  
            - **Estate vs Inverno**:  
            - Luxury e Regular mantengono performance simili in estate e inverno
            - Premium mostra una forte debolezza estiva (-33% rispetto all'inverno)
            - Budget ha la sua seconda migliore performance in primavera, contrariamente agli altri segmenti

            #### 3. Intensità delle Variazioni
            - Premium mostra le oscillazioni più marcate (102% tra picco e minimo)  
            - Budget e Luxury mostrano variazioni intermedie (73% e 64% rispettivamente)  
            - Regular ha le variazioni più contenute (61%)

            #### 4. Correlazioni con il Trend Generale (Rho)
            - Tutti i segmenti mostrano correlazioni molto alte (>0.94) con l'andamento generale  
            - La correlazione decresce leggermente man mano che si scende di fascia prezzo:  
            - Luxury: 0.9976 
            - Premium: 0.9556
            - Regular: 0.9832
            - Budget: 0.9432

            #### Suggerimenti per Approfondimenti

            1. **Analisi Mix Prodotto**  
            Esaminare se le variazioni stagionali sono dovute a:
            - Cambiamenti nel mix di prodotti venduti
            - Variazioni di prezzo/promozioni 
            - Variazioni nei volumi degli stessi prodotti

            2. **Analisi Clientela**
            - Verificare se cambiano i pattern di acquisto degli stessi clienti
            - O se cambiano i segmenti di clientela attivi nelle diverse stagioni

            3. **Analisi Marginalità**
            - Studiare come variano i margini nelle diverse stagioni
            - Valutare l'impatto delle promozioni stagionali

            4. **Confronto con il Mercato**
            - Confrontare questi pattern con i trend di mercato del settore luxury
            - Identificare eventuali specificità del brand

            *Queste analisi aggiuntive potrebbero fornire insight più actionable per ottimizzare la strategia stagionale per ciascun segmento.*
            """)

        

        # Calcolo della concentrazione globale
        st.subheader("Concentrazione delle Vendite (Globale)")

        # Raggruppa per prodotto e calcola le vendite totali per ciascun prodotto
        product_sales_global = df.groupby('StockCode')['Total_Value'].sum().sort_values(ascending=False)

        # Funzione per calcolare la curva di Lorenz
        def calculate_lorenz_curve(values):
            """Calcola i punti per la curva di Lorenz"""
            sorted_values = np.sort(values)
            cumx = np.cumsum(sorted_values)
            sumy = cumx / cumx[-1]  # Normalizzazione
            sumx = np.arange(1, len(sumy) + 1) / len(sumy)
            return sumx, sumy

        # Calcola la curva di Lorenz per tutte le vendite
        lorenz_x, lorenz_y = calculate_lorenz_curve(product_sales_global.values)

        # Calcolo del top 5% (globale)
        total_sales_global = product_sales_global.sum()
        num_top_5_products_global = max(1, int(len(product_sales_global) * 0.05))
        top_5_products_global = product_sales_global.head(num_top_5_products_global).sum()
        top_5_pct_global = (top_5_products_global / total_sales_global) * 100

        # Genera il grafico della curva di Lorenz
        fig_lorenz = go.Figure()

        # Curva di Lorenz
        fig_lorenz.add_trace(go.Scatter(
            x=lorenz_x, 
            y=lorenz_y, 
            mode='lines', 
            name='Curva di Lorenz'
        ))

        # Linea di uniformità (diagonale)
        fig_lorenz.add_trace(go.Scatter(
            x=[0, 1], 
            y=[0, 1], 
            mode='lines', 
            line=dict(color='red', dash='dash'), 
            name='Uniformità'
        ))

        # Configurazione del layout
        fig_lorenz.update_layout(
            title="Curva di Concentrazione Globale",
            xaxis_title="Percentuale cumulativa dei Prodotti",
            yaxis_title="Percentuale cumulativa delle Vendite",
            showlegend=True,
            height=500
        )

        # Mostra il grafico
        st.plotly_chart(fig_lorenz, use_container_width=True)

        # Tabella Riepilogativa
        st.subheader("Tabella Riepilogativa della Concentrazione Globale")
        concentration_global_data = {
            'Metrica': ['Top 5% Prodotti Generano', 'N. Prodotti Totali', 'N. Prodotti Top 5%'],
            'Valore': [f'{top_5_pct_global:.1f}%', len(product_sales_global), num_top_5_products_global]
        }

        concentration_global_df = pd.DataFrame(concentration_global_data)
        st.dataframe(concentration_global_df, use_container_width=True)

        st.markdown("""
        ### Interpretazione della Concentrazione delle Vendite (Globale)

        #### **Curva di Lorenz**
        - La curva di Lorenz indica una **forte concentrazione delle vendite**, con una piccola frazione di prodotti che genera la maggior parte dei ricavi.
        - Il **44.4% delle vendite totali** è generato dal **top 5% dei prodotti**, evidenziando che circa 200 prodotti dominano il mercato.
        - Una curva più vicina all'angolo inferiore sinistro rispetto alla diagonale rossa tratteggiata riflette questa concentrazione.

        #### **Tabella Riepilogativa**
        - **N. Prodotti Totali**: 4017 prodotti analizzati.
        - **Top 5% Prodotti Generano**: 44.4% del revenue totale.
        - **N. Prodotti Top 5%**: 200 prodotti.

        #### **Conclusioni e Implicazioni**
        1. **Concentrazione significativa**: Una piccola frazione di prodotti (5%) genera quasi la metà delle vendite.
        2. **Focalizzazione strategica**:
        - **Best-seller**: Dare priorità al mantenimento delle scorte, pricing ottimizzato e promozioni mirate.
        - **Prodotti a bassa performance**: Valutare opportunità di riposizionamento o eliminazione.
        3. **Monitoraggio continuo**: Tenere sotto controllo la concentrazione per identificare cambiamenti strategici o potenziali rischi.
        """)


        

        # 3. Cross-Selling tra Segmenti

        
        # Ordinare gli acquisti nel tempo
        df = df.sort_values(by=['Customer ID', 'InvoiceDate'])

        # Identificare il segmento iniziale per ogni cliente
        df['initial_segment'] = df.groupby('Customer ID')['price_segment'].transform('first')

        # Identificare tutti i segmenti successivi per ogni cliente
        df['next_segment'] = df['price_segment']

        # Creare una matrice di transizione
        cross_selling_matrix = df.groupby(['initial_segment', 'next_segment'])['Customer ID'].nunique()
        cross_selling_matrix = cross_selling_matrix.unstack(fill_value=0)

        # Calcolare le percentuali per ogni segmento iniziale
        cross_selling_percentages = (cross_selling_matrix.T / cross_selling_matrix.sum(axis=1)).T * 100

        


        st.subheader("3. Cross-Selling tra Segmenti")

        # Calcolo della matrice di transizione cross-selling
        cross_selling_matrix = df.groupby(['initial_segment', 'next_segment'])['Customer ID'].nunique()
        cross_selling_matrix = cross_selling_matrix.unstack(fill_value=0)

        # Calcolo delle percentuali
        cross_selling_percentages = (cross_selling_matrix.T / cross_selling_matrix.sum(axis=1)).T * 100

        # Creazione del grafico a heatmap
        fig_cross_selling = px.imshow(
            cross_selling_percentages,
            labels=dict(x="Segmento Successivo", y="Segmento Iniziale", color="% Clienti"),
            text_auto='.1f',  # Aggiunge annotazioni con formato a una cifra decimale
            title="Matrice Cross-Selling tra Segmenti",
            color_continuous_scale="Blues",
        )

        # Configurazione del layout
        fig_cross_selling.update_layout(
            xaxis_title="Segmento Successivo",
            yaxis_title="Segmento Iniziale",
            font=dict(size=12),
            coloraxis_colorbar=dict(title="% Clienti")
        )

        # Mostra il grafico
        st.plotly_chart(fig_cross_selling, use_container_width=True)

        # Spiegazione aggiuntiva
        st.markdown("""
        ### Cosa significa il Cross-Selling in questo contesto?
        - Questa matrice rappresenta il comportamento dei clienti che acquistano prodotti appartenenti a segmenti diversi.
        - Ogni cella mostra la percentuale di clienti che, partendo da un segmento iniziale, hanno acquistato prodotti di un altro segmento.
        - Un valore più alto indica una maggiore propensione al cross-selling tra quei due segmenti.

        ### Interpretazione del Grafico
        - Le righe rappresentano il segmento da cui i clienti hanno iniziato i loro acquisti.
        - Le colonne mostrano i segmenti successivi in cui questi clienti hanno acquistato.
        - Le celle scure indicano percentuali più alte, cioè una maggiore incidenza di cross-selling.
        """)
        
        # 4. Affinità tra Segmenti Clienti e Prodotti
        st.subheader("4. Affinità tra Segmenti Clienti e Prodotti")

        # Verifica o calcolo della colonna 'Total_Spend'
        if 'Total_Spend' not in customer_stats.columns:
            st.warning("La colonna 'Total_Spend' non è presente. La stiamo calcolando...")

            # Calcola il valore totale speso da ciascun cliente
            customer_total_spend = df.groupby('Customer ID')['Total_Value'].sum().reset_index()
            customer_total_spend.columns = ['Customer ID', 'Total_Spend']

            # Unisci il calcolo al dataset customer_stats
            customer_stats = customer_stats.merge(customer_total_spend, on='Customer ID', how='left')

        # Verifica o calcolo della colonna 'customer_segment'
        if 'customer_segment' not in customer_stats.columns:
            st.warning("La colonna 'customer_segment' non è presente. La stiamo ricreando...")

            # Crea i segmenti cliente basati su 'Total_Spend'
            customer_stats['customer_segment'] = pd.cut(
                customer_stats['Total_Spend'],
                bins=[0, 500, 1000, 5000, np.inf],
                labels=['At Risk', 'Regular', 'Loyal', 'VIP']
            )

        # Aggiungi 'customer_segment' a df tramite unione
        if 'customer_segment' not in df.columns:
            df = df.merge(customer_stats[['Customer ID', 'customer_segment']], on='Customer ID', how='left')

        # Verifica la presenza di 'customer_segment'
        if 'customer_segment' not in df.columns:
            st.error("La colonna 'customer_segment' non è stata aggiunta correttamente a df.")
        else:
            # Calcolo della matrice di affinità
            affinity_matrix = df.groupby(['customer_segment', 'price_segment'])['Total_Value'].sum()
            affinity_matrix = affinity_matrix.unstack(fill_value=0)

            # Calcolo delle percentuali
            affinity_percentages = (affinity_matrix.T / affinity_matrix.sum(axis=1)).T * 100

            # Creazione del grafico a heatmap
            fig_affinity = px.imshow(
                affinity_percentages,
                labels=dict(x="Segmento Prodotto", y="Segmento Cliente", color="% Vendite"),
                text_auto='.1f',
                title="Affinità tra Segmenti Clienti e Prodotti",
                color_continuous_scale="Blues"
            )

            fig_affinity.update_layout(
                xaxis_title="Segmento Prodotto",
                yaxis_title="Segmento Cliente",
                font=dict(size=12),
                coloraxis_colorbar=dict(title="% Vendite")
            )

            # Mostra il grafico
            st.plotly_chart(fig_affinity, use_container_width=True)

            # Tabella riepilogativa
            st.subheader("Tabella Riepilogativa dell'Affinità")
            st.dataframe(affinity_percentages)

            st.markdown("""
            ### Interpretazione del Grafico e della Tabella

            #### **Grafico Heatmap**
            1. **Segmento Cliente "At Risk"**:
            - Affinità distribuita tra i segmenti **Luxury** (31.5%) e **Regular** (25.9%).
            - La distribuzione è equilibrata, senza una forte dominanza di un segmento specifico.

            2. **Segmento Cliente "Loyal"**:
            - Forte affinità verso prodotti **Luxury** (33.9%), seguiti da **Regular** (28.5%).
            - I clienti fedeli mostrano un interesse sia per prodotti di fascia alta sia per quelli regolari.

            3. **Segmento Cliente "Regular"**:
            - Leggera preferenza per prodotti **Luxury** (30.9%) e **Regular** (28.1%).
            - Questo gruppo evidenzia una moderata propensione verso prodotti di fascia più alta.

            4. **Segmento Cliente "VIP"**:
            - Netta affinità verso prodotti **Luxury** (40.7%).
            - I VIP sono altamente focalizzati sui prodotti di lusso, con una minore propensione verso **Budget** e **Regular**.

            #### **Tabella Riepilogativa**
            1. **Segmento Luxury**:
            - È dominante tra i clienti **VIP** (40.7%) e ha una forte affinità con i clienti **Loyal** (33.9%).
            2. **Segmento Budget**:
            - Attira maggiormente i clienti **At Risk** (16.8%).
            3. **Segmento Premium**:
            - Riceve vendite principalmente dai clienti **Loyal** (23.6%) e **Regular** (24.6%).
            4. **Segmento Regular**:
            - Presenta una distribuzione uniforme tra i clienti **At Risk**, **Loyal**, e **Regular** (25-28%).

            #### **Conclusioni**
            1. **VIP**:
            - Concentrarsi sui prodotti **Luxury** con strategie di marketing personalizzate.
            2. **At Risk**:
            - Incentivare il segmento **Budget** con promozioni per aumentare la penetrazione.
            3. **Loyal e Regular**:
            - Investire in prodotti **Luxury** e **Premium** per massimizzare le vendite.
            """)



        
    def render_product_analysis(self, df):
        """Render dell'analisi prodotti"""
        st.header("🛍️ Analisi Prodotti")
        
        # Definiamo l'ordine dei segmenti una volta sola
        segment_order = ['Budget', 'Regular', 'Premium', 'Luxury']
        
        # Creiamo una copia del dataframe e settiamo categorical
        df = df.copy()
        df['StockCode'] = df['StockCode'].astype(str)
        df['price_segment'] = pd.Categorical(
            df['price_segment'],
            categories=segment_order,
            ordered=True
        )
        
        # Spiegazione della segmentazione
        st.markdown("""
        ### Segmentazione dei Prodotti

        I prodotti sono stati categorizzati in quattro segmenti di prezzo utilizzando i quartili della distribuzione dei prezzi medi:
        - **Budget**: prodotti con prezzo ≤ 25° percentile
        - **Regular**: prodotti con prezzo tra 25° e 50° percentile
        - **Premium**: prodotti con prezzo tra 50° e 75° percentile
        - **Luxury**: prodotti con prezzo > 75° percentile

        Questa segmentazione è relativa alla distribuzione dei prezzi nel dataset, garantendo una suddivisione bilanciata 
        dei prodotti tra i segmenti (~25% in ogni segmento) e adattandosi automaticamente al range di prezzi presente nei dati.
        """)
        
        try:
            # Performance per categoria
            col1, col2 = st.columns(2)
            
            with col1:
                # Calcoliamo revenue e percentuali per categoria
                category_perf = df.groupby('price_segment')['Total_Value'].sum().reset_index()
                category_perf.columns = ['Segmento', 'Revenue']
                total_revenue = category_perf['Revenue'].sum()
                category_perf['Percentuale'] = (category_perf['Revenue'] / total_revenue * 100)

                fig_cat = go.Figure()
                fig_cat.add_trace(go.Bar(
                    x=category_perf['Segmento'],
                    y=category_perf['Revenue'],
                    text=[f"€{r:,.0f}<br>({p:.1f}%)" for r, p in zip(
                        category_perf['Revenue'],
                        category_perf['Percentuale']
                    )],
                    textposition='auto',
                ))

                fig_cat.update_layout(
                    title="Revenue per Categoria",
                    xaxis_title="Segmento",
                    yaxis_title="Revenue (€)",
                    showlegend=False,
                    height=400
                )
                st.plotly_chart(fig_cat, use_container_width=True)
                    
            with col2:
                # Trend stagionale
                seasonal = df.groupby('season')['Total_Value'].sum()
                fig_seasonal = px.line_polar(
                    r=seasonal.values,
                    theta=seasonal.index,
                    line_close=True,
                    title="Trend Stagionale"
                )
                st.plotly_chart(fig_seasonal, use_container_width=True)

            # Interpretazione del trend stagionale
            st.markdown("""
            ### Interpretazione del Trend Stagionale

            Il grafico radar mostra la distribuzione stagionale delle vendite, evidenziando che:
            - L'Autunno è il periodo di picco per le vendite
            - Primavera e Inverno mantengono livelli di vendita simili, leggermente inferiori all'Autunno
            - L'Estate registra le performance più basse

            Questo pattern suggerisce:
            - L'opportunità di rafforzare le strategie commerciali durante il periodo estivo
            - La possibilità di capitalizzare ulteriormente sul picco autunnale
            - La presenza di una base stabile di vendite durante Inverno e Primavera
            """)
                
            # Top prodotti
            st.subheader("📈 Top Prodotti")
            top_products = (df.groupby('StockCode').agg({
                'Description': 'first',
                'price_segment': 'first',
                'Total_Value': 'sum',
                'Quantity': 'sum'
            })
            .sort_values('Total_Value', ascending=False)
            .head(10)
            .reset_index())
            
            formatted_products = pd.DataFrame({
                'Codice': top_products['StockCode'],
                'Descrizione': top_products['Description'],
                'Categoria': top_products['price_segment'],
                'Revenue Totale': top_products['Total_Value'].map('€{:,.2f}'.format),
                'Quantità': top_products['Quantity'].map('{:,}'.format)
            })
            
            st.dataframe(formatted_products, use_container_width=True, hide_index=True)
            
            # Caratteristiche dei Segmenti di Prezzo
            st.header("Analisi Dettagliata dei Segmenti di Prezzo")
            st.subheader("Caratteristiche dei Segmenti di Prezzo")

            # Calcoliamo prima i quartili che definiscono i confini dei segmenti
            product_avg_prices = df.groupby('StockCode')['Price'].mean()
            price_percentiles = product_avg_prices.quantile([0.25, 0.5, 0.75])

            # Creiamo prima la tabella con i range corretti
            price_ranges = pd.DataFrame({
                'Range Prezzo': [
                    f"€0.00 - €{price_percentiles[0.25]:.2f}",
                    f"€{price_percentiles[0.25]:.2f} - €{price_percentiles[0.50]:.2f}",
                    f"€{price_percentiles[0.50]:.2f} - €{price_percentiles[0.75]:.2f}",
                    f"€{price_percentiles[0.75]:.2f} - €{df['Price'].max():.2f}"
                ]
            }, index=['Budget', 'Regular', 'Premium', 'Luxury'])

            st.dataframe(price_ranges, use_container_width=True , hide_index=True)

            st.markdown("""
            **Nota sulla segmentazione:** 
            I segmenti sono definiti sui prezzi medi dei prodotti usando i quartili della distribuzione:
            - Budget: prodotti con prezzo medio fino al 25° percentile
            - Regular: prodotti con prezzo medio tra il 25° e il 50° percentile
            - Premium: prodotti con prezzo medio tra il 50° e il 75° percentile
            - Luxury: prodotti con prezzo medio sopra il 75° percentile
            """)

            # Poi continuiamo con le altre statistiche
            price_stats = df.groupby('price_segment').agg({
                'Price': ['min', 'max', 'mean', 'median', 'std'],
                'StockCode': 'nunique'
            }).round(2)
            
            total_sku = price_stats[('StockCode', 'nunique')].sum()
            price_stats[('StockCode', '%_SKU')] = (price_stats[('StockCode', 'nunique')] / total_sku * 100).round(1)
            
            price_stats_display = pd.DataFrame({
                
                'Prezzo Medio': [f"€{x:,.2f}" for x in price_stats[('Price', 'mean')]],
                'Prezzo Mediano': [f"€{x:,.2f}" for x in price_stats[('Price', 'median')]],
                'Dev. Standard': [f"€{x:,.2f}" for x in price_stats[('Price', 'std')]],
                'Num. Prodotti': price_stats[('StockCode', 'nunique')],
                '% sul Totale Prodotti': [f"{x}%" for x in price_stats[('StockCode', '%_SKU')]]
            }, index=price_stats.index)
            
            st.dataframe(price_stats_display, use_container_width=True)
            
            # Performance di Vendita per Segmento
            st.subheader("Performance di Vendita per Segmento")
            
            segment_perf = df.groupby('price_segment').agg({
                'Invoice': 'count',
                'Quantity': 'sum',
                'Total_Value': 'sum'
            })
            
            segment_perf['% Volume'] = (segment_perf['Quantity'] / segment_perf['Quantity'].sum() * 100).round(1)
            segment_perf['% Valore'] = (segment_perf['Total_Value'] / segment_perf['Total_Value'].sum() * 100).round(1)
            segment_perf['AOV'] = (segment_perf['Total_Value'] / segment_perf['Invoice']).round(2)
            
            perf_display = pd.DataFrame({
                'N. Transazioni': segment_perf['Invoice'].map('{:,.0f}'.format),
                'Volume Totale': segment_perf['Quantity'].map('{:,.0f}'.format),
                'Valore Totale': segment_perf['Total_Value'].map('€{:,.2f}'.format),
                '% Volume': segment_perf['% Volume'].map('{}%'.format),
                '% Valore': segment_perf['% Valore'].map('{}%'.format),
                'AOV': segment_perf['AOV'].map('€{:,.2f}'.format)
            }, index=segment_perf.index)
            
            st.dataframe(perf_display, use_container_width=True)
            
            # Analisi Statistica
            st.subheader("Analisi Statistica delle Differenze tra Segmenti")
            
            # Test per valore
            st.markdown("**Test sulle differenze in termini di valore**")
            h_stat_value, p_val_value = stats.kruskal(
                *[group['Total_Value'].values for name, group in df.groupby('price_segment')]
            )
            
            value_test = pd.DataFrame({
                'Metrica': ['H-statistic', 'p-value', 'Esito'],
                'Valore': [
                    f"{h_stat_value:.2f}",
                    f"{p_val_value:.4f}",
                    'Differenze Significative' if p_val_value < 0.05 else 'Differenze Non Significative'
                ]
            })
            
            st.table(value_test)
            
            # Test per volume
            st.markdown("**Test sulle differenze in termini di volume**")
            h_stat_vol, p_val_vol = stats.kruskal(
                *[group['Quantity'].values for name, group in df.groupby('price_segment')]
            )
            
            volume_test = pd.DataFrame({
                'Metrica': ['H-statistic', 'p-value', 'Esito'],
                'Valore': [
                    f"{h_stat_vol:.2f}",
                    f"{p_val_vol:.4f}",
                    'Differenze Significative' if p_val_vol < 0.05 else 'Differenze Non Significative'
                ]
            })
            
            st.table(volume_test)
            
            st.markdown("""
            **Interpretazione dei Test Statistici:**
            - Il test di Kruskal-Wallis verifica se esistono differenze significative tra i segmenti
            - Un p-value < 0.05 indica differenze statisticamente significative
            - I test sono stati eseguiti sia sul valore delle transazioni che sul volume
            """)
                
        except Exception as e:
            st.error(f"Errore nell'analisi prodotti: {str(e)}")      
    
    def render_retention_analysis(self, customer_stats):
        st.header("🔄 Analisi Retention")

        # Opzione per includere o escludere nuovi clienti del 2011
        include_new_customers = st.radio(
            "Includere i clienti che hanno fatto il primo acquisto nel 2011?",
            ("Includi", "Escludi")
        )

        if include_new_customers == "Includi":
            # Tutti i clienti attivi nel 2011
            customer_stats['is_retained'] = customer_stats['last_purchase'].dt.year == 2011
            st.write("Distribuzione con tutti i clienti attivi nel 2011:")
            st.bar_chart(customer_stats['is_retained'].value_counts())
        else:
            # Escludere i nuovi clienti del 2011
            retained_excluding_new = (
                (customer_stats['last_purchase'].dt.year == 2011) &
                (customer_stats['first_purchase'].dt.year < 2011)
            )
            customer_stats['is_retained'] = retained_excluding_new.astype(int)
            st.write("Distribuzione escludendo i nuovi clienti nel 2011:")
            st.bar_chart(customer_stats['is_retained'].value_counts())

        # Selezione delle feature e del target
        features = customer_stats[['recency', 'frequency', 'total_spend', 'avg_order_value']]
        target = customer_stats['is_retained']

        # Verifica statistiche delle feature
        st.write("Statistiche delle Variabili di Input:")
        st.dataframe(features.describe())

        # Divisione train/test
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # -------------------------
        # Decision Tree Model
        # -------------------------
        st.subheader("📉 Decision Tree")
        dt_model = DecisionTreeClassifier(max_depth=3, random_state=42)
        dt_model.fit(X_train, y_train)
        dt_preds = dt_model.predict(X_test)

        st.text("Classification Report (Decision Tree):")
        st.text(classification_report(y_test, dt_preds))

        dt_accuracy = accuracy_score(y_test, dt_preds)
        st.metric("Decision Tree Accuracy", f"{dt_accuracy * 100:.2f}%")

        # Visualizzazione dell'albero decisionale
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_tree(dt_model, feature_names=features.columns, class_names=['Non Retained', 'Retained'], filled=True, ax=ax)
        st.pyplot(fig)

        # -------------------------
        # Confusion Matrix
        # -------------------------
        st.subheader("🔢 Matrice di Confusione")
        fig, ax = plt.subplots(figsize=(6, 6))
        ConfusionMatrixDisplay.from_estimator(dt_model, X_test, y_test, display_labels=['Non Retained', 'Retained'], cmap='Blues', ax=ax)
        plt.title('Matrice di Confusione (Decision Tree)')
        st.pyplot(fig)

        # -------------------------
        # Partial Dependence Plots (PDP)
        # -------------------------
        st.subheader("📊 Partial Dependence Plots (PDP)")
        pdp_features = ['recency', 'frequency', 'total_spend']
        # Partial Dependence Plots
        fig, ax = plt.subplots(figsize=(10, 8))
        PartialDependenceDisplay.from_estimator(
            dt_model, X_train, ['recency', 'frequency', 'total_spend'], ax=ax
        )
        plt.title('Partial Dependence Plots (Decision Tree)')
        st.pyplot(fig)

        # -------------------------
        # Feature Importance
        # -------------------------
        st.subheader("Importanza delle Variabili")
        feature_importance = pd.DataFrame({
            'Feature': features.columns,
            'Importance': dt_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        st.bar_chart(feature_importance.set_index('Feature'))

        # -------------------------
        # Random Forest Model
        # -------------------------
        st.subheader("🌲 Random Forest")
        rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
        rf_model.fit(X_train, y_train)
        rf_preds = rf_model.predict(X_test)

        st.text("Classification Report (Random Forest):")
        st.text(classification_report(y_test, rf_preds))

        rf_accuracy = accuracy_score(y_test, rf_preds)
        st.metric("Random Forest Accuracy", f"{rf_accuracy * 100:.2f}%")

        # Feature Importance (Random Forest)
        rf_feature_importance = pd.DataFrame({
            'Feature': features.columns,
            'Importance': rf_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        st.subheader("Importanza delle Variabili (Random Forest)")
        st.bar_chart(rf_feature_importance.set_index('Feature'))

        # PDP per recency e frequency
        st.subheader("📊 Partial Dependence Plots (Random Forest)")
        pdp_features = ['recency', 'frequency', 'total_spend']
        
        # Plot PDP
        fig, ax = plt.subplots(figsize=(10, 6))
        PartialDependenceDisplay.from_estimator(
            rf_model, X_train, pdp_features, ax=ax
        )
        plt.title('Partial Dependence Plots (Random Forest)')
        st.pyplot(fig)

        st.markdown("""
        # Relazione sull'Analisi della Retention

        ## Introduzione
        L'analisi della retention ha avuto l'obiettivo di comprendere i fattori che influenzano il comportamento dei clienti in termini di fedeltà e frequenza di acquisto, utilizzando modelli di Machine Learning (“Decision Tree” e “Random Forest”). I risultati sono stati approfonditi tramite tecniche di interpretazione come i Partial Dependence Plots (PDP) e le metriche di performance.

        ---

        ## Risultati Principali

        ### 1. **Performance dei Modelli**
        #### Decision Tree
        - **Accuracy**: 75%
        - **Precision (classe Retained)**: 84%
        - **Recall (classe Retained)**: 81%
        - **Variabile più importante**: `recency`

        #### Random Forest
        - **Accuracy**: 93.79%
        - **Precision (classe Retained)**: 95%
        - **Recall (classe Retained)**: 91%
        - **Variabile più importante**: `recency`

        ### 2. **Analisi delle Variabili Chiave tramite PDP**

        #### **Recency** (tempo dall'ultimo acquisto)
        - La probabilità di retention diminuisce drasticamente con l'aumento di `recency`.
        - **Punto critico**: Oltre i 400 giorni, la probabilità di retention si stabilizza a valori bassi (<20%).
        - **Interpretazione**: Clienti che non effettuano acquisti da molto tempo sono a rischio di abbandono.

        #### **Frequency** (frequenza media degli acquisti)
        - La probabilità di retention è più alta per valori di `frequency` molto bassi (<0.1), ma diminuisce rapidamente e si stabilizza dopo 0.25 (1 acquisto ogni anno).
        - **Punto critico**: Valori di `frequency` inferiori a 0.25 indicano clienti meno regolari ma ancora recuperabili.
        - **Interpretazione**: Alcuni clienti con bassa frequenza potrebbero essere occasionali ma fedeli.

        #### **Total Spend** (spesa totale)
        - La curva è piatta, indicando un impatto minimo della variabile sulla probabilità di retention.
        - **Interpretazione**: Il totale speso non è un indicatore significativo per la retention.

        ---

        ## Evidenze Specifiche

        1. **Distribuzione dei Clienti per Retention**:
        - Escludendo i nuovi clienti del 2011, i modelli migliorano in precisione e recall.
        - La classe "Retained" è influenzata principalmente da `recency` e `frequency`.

        2. **Confusion Matrix**:
        - I modelli performano meglio nel predire i clienti retained rispetto ai non-retained.
        - La classe "Non Retained" potrebbe beneficiare di un miglior bilanciamento dei dati.

        3. **Feature Importance**:
        - `recency` è la variabile più influente sia per il Decision Tree che per la Random Forest.
        - `frequency` gioca un ruolo significativo ma secondario.
        - `total_spend` ha un impatto trascurabile.

        ---

        ## Suggerimenti e Iniziative

        ### **Strategie Basate su Recency**
        1. **Targeting clienti con alta recency (300-400 giorni):**
        - Offrire promozioni di ritorno (es. sconti personalizzati o offerte a tempo).
        - Inviare comunicazioni mirate tramite email o SMS.
        2. **Programmi di riattivazione per recency > 400 giorni:**
        - Implementare campagne "win-back" per clienti inattivi.
        - Offrire incentivi significativi per incoraggiarli a tornare.

        ### **Strategie Basate su Frequency**
        1. **Incentivare acquisti regolari:**
        - Programmi fedeltà con premi per acquisti ricorrenti (es. cashback).
        - Creare campagne stagionali o mensili per mantenere alta la frequenza.
        2. **Analizzare clienti a bassa frequency (<0.25):**
        - Identificare segmenti specifici (es. clienti con acquisti occasionali ma significativi).
        - Offrire pacchetti o abbonamenti per fidelizzarli.

        ### **Strategie Generali**
        1. **Segmentazione Avanzata:**
        - Segmentare ulteriormente i clienti in base a combinazioni di `recency` e `frequency` per personalizzare le strategie.
        2. **Integrare Nuove Variabili:**
        - Esplorare variabili aggiuntive come il tipo di prodotto acquistato o la geolocalizzazione per migliorare il modello.
        3. **Monitoraggio Continuo:**
        - Monitorare costantemente i clienti retained e non-retained per valutare l'efficacia delle iniziative.

        ---

        ## Conclusioni
        L'analisi ha evidenziato che la **recency** è il principale driver di retention, seguito dalla **frequency**. Le iniziative suggerite mirano a ridurre il rischio di abbandono e a incrementare la fedeltà dei clienti attraverso strategie mirate e programmi personalizzati.

        Ulteriori approfondimenti potrebbero includere modelli di clustering per identificare pattern nascosti nei dati e migliorare ulteriormente l'efficacia delle azioni.
        """)

        # -------------------------
        # Ottimizzazione della Random Forest con GridSearchCV
        # -------------------------
        st.subheader("🌲 Ottimizzazione Random Forest")
        if st.button("Esegui Ottimizzazione Random Forest"):
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }

            st.write("Esecuzione GridSearchCV per ottimizzare gli iperparametri...")
            grid_search = GridSearchCV(
                estimator=RandomForestClassifier(random_state=42),
                param_grid=param_grid,
                scoring='accuracy',
                cv=3,
                verbose=1,
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)

            # Migliori parametri trovati
            st.write("Migliori parametri trovati:", grid_search.best_params_)

            # Modello ottimizzato
            best_model = grid_search.best_estimator_

        # -------------------------
        # Random Forest con i migliori parametri
        # -------------------------
        if st.button("Esegui Analisi Random Forest Ottimizzata"):
            st.subheader("🌲 Random Forest Ottimizzato")
            rf_preds = best_model.predict(X_test)

            st.text("Classification Report (Random Forest Ottimizzato):")
            st.text(classification_report(y_test, rf_preds))

            rf_accuracy = accuracy_score(y_test, rf_preds)
            st.metric("Random Forest Accuracy (Ottimizzato)", f"{rf_accuracy * 100:.2f}%")

            # Feature Importance
            rf_feature_importance = pd.DataFrame({
                'Feature': features.columns,
                'Importance': best_model.feature_importances_
            }).sort_values(by='Importance', ascending=False)

            st.subheader("Importanza delle Variabili (Random Forest Ottimizzato)")
            st.bar_chart(rf_feature_importance.set_index('Feature'))


    def render_yoy_analysis(self, df: pd.DataFrame):
        """Render dell'analisi Year-over-Year completa"""
        if df is None:
            st.warning("Nessun dato disponibile per l'analisi YoY.")
            return
            
        st.header("📈 Analisi Year-over-Year (2010 vs 2011)")
        
        # Preparazione dati
        df = df.copy()
        df['year'] = df['InvoiceDate'].dt.year
        df['month'] = df['InvoiceDate'].dt.month
        df['year_month'] = df['InvoiceDate'].dt.to_period('M')
        
        # Filtriamo solo per 2010 e 2011
        df_yoy = df[df['year'].isin([2010, 2011])]
        
        if len(df_yoy) == 0:
            st.warning("Non ci sono dati sufficienti per un'analisi YoY tra 2010 e 2011.")
            return
            
        try:
            # 1. METRICHE PRINCIPALI YOY
            metrics_by_year = df_yoy.groupby('year').agg({
                'Total_Value': 'sum',
                'Invoice': 'nunique',
                'Customer ID': 'nunique',
                'Quantity': 'sum'
            }).round(2)

            # Calcolo variazioni YoY
            yoy_changes = {
                'Revenue': ((metrics_by_year.loc[2011, 'Total_Value'] / 
                            metrics_by_year.loc[2010, 'Total_Value'] - 1) * 100).round(1),
                'Ordini': ((metrics_by_year.loc[2011, 'Invoice'] / 
                        metrics_by_year.loc[2010, 'Invoice'] - 1) * 100).round(1),
                'Clienti': ((metrics_by_year.loc[2011, 'Customer ID'] / 
                            metrics_by_year.loc[2010, 'Customer ID'] - 1) * 100).round(1),
                'Volume': ((metrics_by_year.loc[2011, 'Quantity'] / 
                            metrics_by_year.loc[2010, 'Quantity'] - 1) * 100).round(1)
            }

            # Calcolo delta assoluti
            absolute_changes = {
                'Revenue': metrics_by_year.loc[2011, 'Total_Value'] - metrics_by_year.loc[2010, 'Total_Value'],
                'Ordini': metrics_by_year.loc[2011, 'Invoice'] - metrics_by_year.loc[2010, 'Invoice'],
                'Clienti': metrics_by_year.loc[2011, 'Customer ID'] - metrics_by_year.loc[2010, 'Customer ID'],
                'Volume': metrics_by_year.loc[2011, 'Quantity'] - metrics_by_year.loc[2010, 'Quantity']
            }

            st.subheader("Metriche Principali YoY")
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Revenue YoY", 
                    f"{yoy_changes['Revenue']}%",
                    delta=f"-€{abs(absolute_changes['Revenue']):,.0f}",  # Aggiungo il segno - 
                    delta_color="normal"
                )
            with col2:
                st.metric(
                    "Ordini YoY", 
                    f"{yoy_changes['Ordini']}%",
                    delta=f"-{abs(absolute_changes['Ordini']):,.0f}",  # Aggiungo il segno -
                    delta_color="normal"
                )
            with col3:
                st.metric(
                    "Clienti YoY", 
                    f"{yoy_changes['Clienti']}%",
                    delta=f"-{abs(absolute_changes['Clienti']):,.0f}",  # Aggiungo il segno -
                    delta_color="normal"
                )
            with col4:
                st.metric(
                    "Volume YoY", 
                    f"{yoy_changes['Volume']}%",
                    delta=f"-{abs(absolute_changes['Volume']):,.0f}",  # Aggiungo il segno -
                    delta_color="normal"
                )


            st.markdown("""
                    ### 📊 Interpretazione Metriche Principali YoY

                    #### Performance Generale
                    - Il business ha registrato un **calo generale** nel 2011 rispetto al 2010, con riduzioni in tutti i KPI principali
                    - Il **volume** ha subito la contrazione maggiore (-10.9%), suggerendo una significativa riduzione nelle quantità vendute
                    - Il **revenue** è calato del 4.4% (€-379,351), una diminuzione meno marcata rispetto al volume, suggerendo possibili aumenti di prezzo o shift verso prodotti a maggior valore

                    #### Dettaglio KPI
                    1. **Revenue** (-4.4%)  
                    - *Calcolo*: Somma totale di (Quantità × Prezzo) per anno
                    - *Variazione*: ((Revenue 2011 - Revenue 2010) / Revenue 2010) × 100

                    2. **Ordini** (-6.5%)
                    - *Calcolo*: Numero univoco di ordini (Invoice) per anno
                    - *Variazione*: ((N° Ordini 2011 - N° Ordini 2010) / N° Ordini 2010) × 100

                    3. **Clienti** (-0.3%)
                    - *Calcolo*: Numero univoco di Customer ID per anno
                    - *Variazione*: ((N° Clienti 2011 - N° Clienti 2010) / N° Clienti 2010) × 100

                    4. **Volume** (-10.9%)
                    - *Calcolo*: Somma totale delle quantità vendute per anno
                    - *Variazione*: ((Quantità 2011 - Quantità 2010) / Quantità 2010) × 100

                    #### Insights Chiave
                    1. La minima perdita di clienti (-0.3%) rispetto al calo più marcato di ordini (-6.5%) suggerisce che i clienti esistenti hanno **ridotto la frequenza di acquisto**

                    2. Il divario tra calo del revenue (-4.4%) e volume (-10.9%) indica un possibile **aumento del valore medio per unità venduta**, che potrebbe derivare da:
                    - Aumenti di prezzo
                    - Mix di vendita spostato verso prodotti premium
                    - Minori promozioni/sconti

                    3. Le metriche suggeriscono la necessità di:
                    - Analizzare le cause della riduzione della frequenza d'acquisto
                    - Verificare l'impatto dei possibili aumenti di prezzo sulla domanda
                    - Investigare se il calo dei volumi è concentrato in specifici segmenti di prodotto
                    """)
            # 2. TREND MENSILE YOY
            st.markdown("---")
            st.subheader("Trend Mensile YoY")
            
            monthly_revenue = df_yoy.groupby(['year', 'month'])['Total_Value'].sum().unstack(0)
            monthly_growth = ((monthly_revenue[2011] / monthly_revenue[2010] - 1) * 100).round(1)
            
            # Grafico trend mensile
            fig_monthly = go.Figure()
            fig_monthly.add_trace(go.Scatter(
                x=list(range(1,13)), 
                y=monthly_revenue[2010],
                name='2010',
                mode='lines+markers'
            ))
            fig_monthly.add_trace(go.Scatter(
                x=list(range(1,13)), 
                y=monthly_revenue[2011],
                name='2011',
                mode='lines+markers'
            ))
            
            fig_monthly.update_layout(
                title='Confronto Revenue Mensile 2010 vs 2011',
                xaxis_title='Mese',
                yaxis_title='Revenue (€)',
                hovermode='x unified'
            )
            st.plotly_chart(fig_monthly, use_container_width=True)
            
            # Grafico crescita mensile
            fig_growth = go.Figure(go.Bar(
                x=list(range(1,13)),
                y=monthly_growth,
                text=[f"{v:,.1f}%" for v in monthly_growth],
                textposition='auto',
            ))
            fig_growth.update_layout(
                title='Crescita Mensile YoY (%)',
                xaxis_title='Mese',
                yaxis_title='Crescita %',
                showlegend=False
            )
            st.plotly_chart(fig_growth, use_container_width=True)

            # 3. ANALISI SEGMENTI YOY
            st.markdown("---")
            st.subheader("Performance Segmenti YoY")
            
            # Crescita per segmento di prezzo
            segment_yoy = df_yoy.groupby(['year', 'price_segment'])['Total_Value'].sum().unstack(0)
            segment_growth = ((segment_yoy[2011] / segment_yoy[2010] - 1) * 100).round(1)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Revenue per segmento
                fig_segments = go.Figure()
                for year in [2010, 2011]:
                    fig_segments.add_trace(go.Bar(
                        name=str(year),
                        x=segment_yoy.index,
                        y=segment_yoy[year],
                        text=[f"€{v:,.0f}" for v in segment_yoy[year]],
                        textposition='auto',
                    ))
                
                fig_segments.update_layout(
                    title='Revenue per Segmento',
                    xaxis_title='Segmento',
                    yaxis_title='Revenue (€)',
                    barmode='group'
                )
                st.plotly_chart(fig_segments, use_container_width=True)
                
            with col2:
                # Crescita per segmento
                fig_segment_growth = go.Figure(go.Bar(
                    x=segment_growth.index,
                    y=segment_growth,
                    text=[f"{v:,.1f}%" for v in segment_growth],
                    textposition='auto',
                ))
                
                fig_segment_growth.update_layout(
                    title='Crescita YoY per Segmento (%)',
                    xaxis_title='Segmento',
                    yaxis_title='Crescita %',
                    showlegend=False
                )
                st.plotly_chart(fig_segment_growth, use_container_width=True)

            # 4. ANALISI CLIENTE YOY
            st.markdown("---")
            st.subheader("Evoluzione Comportamento Cliente")
            
            # Metriche per cliente
            customer_metrics = df_yoy.groupby(['year', 'Customer ID']).agg({
                'Total_Value': ['sum', 'mean'],
                'Invoice': 'nunique',
            }).round(2)
            
            # Reset index per accesso più facile
            customer_metrics.columns = ['total_spend', 'avg_order_value', 'num_orders']
            customer_metrics = customer_metrics.reset_index()
            
            # Calcolo metriche medie per cliente
            avg_metrics = customer_metrics.groupby('year').agg({
                'total_spend': 'mean',
                'avg_order_value': 'mean',
                'num_orders': 'mean'
            }).round(2)
            
            # Calcolo variazioni
            customer_yoy = {
                'Spesa Media': ((avg_metrics.loc[2011, 'total_spend'] / 
                            avg_metrics.loc[2010, 'total_spend'] - 1) * 100).round(1),
                'Valore Medio Ordine': ((avg_metrics.loc[2011, 'avg_order_value'] / 
                                    avg_metrics.loc[2010, 'avg_order_value'] - 1) * 100).round(1),
                'Frequenza Ordini': ((avg_metrics.loc[2011, 'num_orders'] / 
                                    avg_metrics.loc[2010, 'num_orders'] - 1) * 100).round(1)
            }
            
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Δ Spesa Media per Cliente", 
                    f"{customer_yoy['Spesa Media']}%",
                    delta=f"-€{abs(avg_metrics.loc[2011, 'total_spend'] - avg_metrics.loc[2010, 'total_spend']):,.2f}",
                    delta_color="normal"
                )
            with col2:
                st.metric(
                    "Δ Valore Medio Ordine", 
                    f"{customer_yoy['Valore Medio Ordine']}%",
                    delta=f"€{abs(avg_metrics.loc[2011, 'avg_order_value'] - avg_metrics.loc[2010, 'avg_order_value']):,.2f}",
                    delta_color="normal"
                )
            with col3:
                st.metric(
                    "Δ Frequenza Ordini", 
                    f"{customer_yoy['Frequenza Ordini']}%",
                    delta=f"-{abs(avg_metrics.loc[2011, 'num_orders'] - avg_metrics.loc[2010, 'num_orders']):,.1f}",
                    delta_color="normal"
                )

            st.markdown("""
                    ### 📊 Interpretazione Evoluzione Comportamento Cliente

                    #### Performance Generale
                    - La **spesa media per cliente** è diminuita del 4.1%, indicando una riduzione nel valore lifetime dei clienti
                    - Il **valore medio degli ordini** è aumentato significativamente (+83.8%), suggerendo un cambio nelle abitudini di acquisto
                    - La **frequenza degli ordini** è calata del 6.2%, mostrando una minore attività dei clienti

                    #### Analisi Dettagliata
                    1. **Spesa Media per Cliente** (-4.1%)
                    - *Calcolo*: Revenue totale / Numero di clienti unici per anno
                    - *Variazione*: ((Spesa Media 2011 - Spesa Media 2010) / Spesa Media 2010) × 100
                    - Il calo suggerisce una diminuzione generale nella propensione alla spesa

                    2. **Valore Medio Ordine** (+83.8%)
                    - *Calcolo*: Revenue totale / Numero di ordini per anno
                    - *Variazione*: ((Valore Medio 2011 - Valore Medio 2010) / Valore Medio 2010) × 100
                    - L'aumento significativo indica un consolidamento degli ordini: meno ordini ma di valore maggiore

                    3. **Frequenza Ordini** (-6.2%)
                    - *Calcolo*: Numero ordini / Numero clienti unici per anno
                    - *Variazione*: ((Frequenza 2011 - Frequenza 2010) / Frequenza 2010) × 100
                    - Il calo nella frequenza suggerisce una minore fidelizzazione

                    #### Insights Chiave
                    1. Il forte aumento del valore medio ordine (+83.8%) combinato con il calo della frequenza (-6.2%) suggerisce un cambio significativo nel comportamento d'acquisto:
                    - I clienti preferiscono fare ordini più sostanziosi ma meno frequenti
                    - Possibile ottimizzazione dei costi di spedizione da parte dei clienti
                    - Potenziale opportunità per strategie di up-selling

                    2. Il calo della spesa media (-4.1%) nonostante l'aumento del valore ordine indica:
                    - Una possibile perdita di opportunità di vendita
                    - Necessità di lavorare sulla frequenza di acquisto
                    - Potenziale spazio per programmi di fidelizzazione

                    #### Suggerimenti Operativi
                    1. Implementare strategie per aumentare la frequenza di acquisto:
                    - Programmi di fedeltà con incentivi sulla frequenza
                    - Comunicazioni marketing più regolari
                    - Offerte speciali per riattivare clienti dormienti

                    2. Capitalizzare sul trend degli ordini di maggior valore:
                    - Bundle di prodotti
                    - Sconti progressivi su ordini più grandi
                    - Servizi premium per ordini di alto valore
                    """)

            # 5. ANALISI RETENTION
            st.markdown("---")
            st.subheader("Analisi Retention")

            # Calcolo retention
            # Prima calcolare tutti i valori necessari
            customers_2010 = set(df_yoy[df_yoy['year']==2010]['Customer ID'].unique())
            customers_2011 = set(df_yoy[df_yoy['year']==2011]['Customer ID'].unique())
            retained = customers_2010.intersection(customers_2011)
            new_2011 = customers_2011 - customers_2010

            # Calcolo retention e acquisition rate
            retention_rate = round((len(retained) / len(customers_2010) * 100), 1)
            acquisition_rate = round((len(new_2011) / len(customers_2010) * 100), 1)

            # Calcolo valori per segmenti di clienti
            retained_value_2010 = df_yoy[
                (df_yoy['year'] == 2010) & 
                (df_yoy['Customer ID'].isin(retained))
            ]['Total_Value'].sum()

            retained_value_2011 = df_yoy[
                (df_yoy['year'] == 2011) & 
                (df_yoy['Customer ID'].isin(retained))
            ]['Total_Value'].sum()

            # Visualizzazione metriche
            col1, col2 = st.columns(2)
            with col1:
                retained_delta = f"-{len(customers_2010) - len(retained):,}"
                st.metric(
                    "Retention Rate", 
                    f"{retention_rate}%",
                    delta=retained_delta,
                    delta_color="normal",
                    help=f"Clienti mantenuti: {len(retained):,}"
                )
            with col2:
                acquisition_delta = f"+{len(new_2011):,}"
                st.metric(
                    "Acquisition Rate", 
                    f"{acquisition_rate}%",
                    delta=acquisition_delta,
                    delta_color="normal",
                    help=f"Nuovi clienti 2011: {len(new_2011):,}"
                )

            # Ora possiamo usare tutti i valori nel markdown
            st.markdown(f"""
            ### 📊 Interpretazione Metriche Retention

            #### Performance Generale
            - **Retention Rate del {retention_rate:.1f}%**: Più della metà dei clienti 2010 ha continuato ad acquistare nel 2011
            - **Acquisition Rate del {acquisition_rate:.1f}%**: Significativa acquisizione di nuovi clienti, ma non sufficiente a compensare la perdita

            #### Dettaglio Metriche
            1. **Retention Rate** ({retention_rate:.1f}%)
            - *Calcolo*: (Clienti attivi in entrambi gli anni / Clienti 2010) × 100
            - Su {len(customers_2010):,} clienti del 2010, {len(retained):,} sono rimasti attivi nel 2011
            - Perdita netta di {len(customers_2010) - len(retained):,} clienti

            2. **Acquisition Rate** ({acquisition_rate:.1f}%)
            - *Calcolo*: (Nuovi clienti 2011 / Clienti 2010) × 100
            - {len(new_2011):,} nuovi clienti acquisiti nel 2011
            - Rappresenta {(len(new_2011)/len(customers_2010)*100):.1f}% della base clienti 2010

            #### Insights Chiave
            1. **Gap di Sostituzione**
            - La perdita di clienti ({(100-retention_rate):.1f}%) non è completamente compensata dai nuovi acquisiti ({acquisition_rate:.1f}%)
            - Risultato: variazione netta della base clienti del {(len(customers_2011) - len(customers_2010))/len(customers_2010)*100:.1f}%

            2. **Opportunità e Rischi**
            - Capacità di mantenere {retention_rate:.1f}% dei clienti
            - Base clienti finale 2011: {len(customers_2011):,} ({(len(customers_2011)/len(customers_2010) - 1)*100:.1f}% vs 2010)
            - Valore medio cliente retained: €{retained_value_2011/len(retained):,.2f} vs €{retained_value_2010/len(retained):,.2f} nel 2010 ({((retained_value_2011/len(retained))/(retained_value_2010/len(retained)) - 1)*100:.1f}% YoY)

            #### Suggerimenti Operativi
            1. **Programma di Retention**:
            - Target: recuperare almeno {int(len(customers_2010) * 0.1):,} clienti persi (10% della base 2010)
            - Potenziale revenue recuperabile: €{(retained_value_2010/len(retained)) * (len(customers_2010) * 0.1):,.2f}
            - Focus sui {len(retained):,} clienti fedeli per programmi di referral

            2. **Strategia di Acquisizione**:
            - Obiettivo acquisizione: {int(len(customers_2010) * (1-retention_rate)):,} nuovi clienti per compensare il churn
            - Target valore primi 12 mesi: €{retained_value_2011/len(retained):,.2f} per cliente
            - Monitoraggio attivo dei {len(new_2011):,} clienti acquisiti nel 2011
            """)
            # 6. ANALISI STAGIONALE YOY
            st.markdown("---")
            st.subheader("Performance Stagionale YoY")
            
            seasonal_yoy = df_yoy.groupby(['year', 'season'])['Total_Value'].sum().unstack(0)
            seasonal_growth = ((seasonal_yoy[2011] / seasonal_yoy[2010] - 1) * 100).round(1)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Revenue per stagione
                fig_seasonal = go.Figure()
                for year in [2010, 2011]:
                    fig_seasonal.add_trace(go.Bar(
                        name=str(year),
                        x=seasonal_yoy.index,
                        y=seasonal_yoy[year],
                        text=[f"€{v:,.0f}" for v in seasonal_yoy[year]],
                        textposition='auto',
                    ))
                
                fig_seasonal.update_layout(
                    title='Revenue per Stagione',
                    xaxis_title='Stagione',
                    yaxis_title='Revenue (€)',
                    barmode='group'
                )
                st.plotly_chart(fig_seasonal, use_container_width=True)
                
            with col2:
                # Crescita stagionale
                fig_seasonal_growth = go.Figure(go.Bar(
                    x=seasonal_growth.index,
                    y=seasonal_growth,
                    text=[f"{v:,.1f}%" for v in seasonal_growth],
                    textposition='auto',
                ))
                
                fig_seasonal_growth.update_layout(
                    title='Crescita YoY per Stagione (%)',
                    xaxis_title='Stagione',
                    yaxis_title='Crescita %',
                    showlegend=False
                )
                st.plotly_chart(fig_seasonal_growth, use_container_width=True)

            # 7. INSIGHTS TESTUALI
            st.markdown("---")
            
        except Exception as e:
            st.error(f"Errore durante l'analisi YoY: {str(e)}")
    
    def render_insights(self, analyzer):
        """Render degli insights di business"""
        st.header("💡 Business Insights")
        
        insights = analyzer.generate_business_insights()
        
        for insight in insights:
            with st.expander(f"📌 {insight['title']}", expanded=True):
                cols = st.columns([2,1,1])
                
                with cols[0]:
                    st.markdown(f"**Finding:** {insight['finding']}")
                    st.markdown(f"**Azione Raccomandata:** {insight['action']}")
                    
                with cols[1]:
                    st.metric("Impatto Stimato", insight['impact'])
                    
                with cols[2]:
                    st.metric("Priorità", insight['priority'])
                    
    def render_sidebar(self):
        """Render della sidebar"""
        with st.sidebar:
            st.header("Info")
            st.markdown("""
            **Progetto**: Luxury Retail Analytics
            
            **Dataset**: Online Retail II
            
            Scaricato da UCI ML Repository                       
            """)
            st.markdown("""
            <a href="https://archive.ics.uci.edu/dataset/502/online+retail+ii" target="_blank">Link</a>
            """, unsafe_allow_html=True)
            st.markdown("""
            **Funzionalità**:
            - Analisi cliente
            - Segmentazione RFM
            - Analisi prodotti
            - Insight di business
            """)
            
    def run(self):
        """Main function della dashboard"""
        self.render_header()
        
        # Opzioni per il caricamento dei dati
        data_option = st.radio(
            "Scegli la fonte dei dati:",
            ["Usa dataset di default", "Carica file personalizzato"]
        )
        
        # Gestione caricamento dati
        if data_option == "Usa dataset di default":
            if not st.session_state.data_loaded and st.button("Carica dataset di default"):
                self.load_default_dataset()
        else:
            uploaded_file = st.file_uploader(
                "Carica il file Excel (Online Retail II dataset)",
                type=['xlsx']
            )
            if uploaded_file is not None and not st.session_state.data_loaded:
                self.load_custom_dataset(uploaded_file)
                
        # Clear data button
        if st.session_state.data_loaded:
            if st.button("Cancella dati caricati"):
                st.session_state.df = None
                st.session_state.customer_stats = None
                st.session_state.data_loaded = False
                st.experimental_rerun()
        
        # Procedi con l'analisi se abbiamo i dati
        if st.session_state.data_loaded:
            try:
                # Render components
                self.render_kpis(st.session_state.df, st.session_state.customer_stats)
                
                tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9 = st.tabs([
                    "👥 Analisi Cliente",
                    "📊 Analisi RFM",
                    "🎯 Analisi Segmenti",
                    "🛍️ Analisi Prodotti",
                    "🔍 Analisi Prodotti Avanzata",
                    "💡 Business Insights",
                    "🔄 Analisi Retention",
                    "📈 Analisi YoY",
                    "🚀 Analisi Spark"
                ])
            
                with tab1:
                    # Mantenere solo la parte iniziale dell'analisi cliente
                    # (fino a prima della sezione RFM)
                    self.render_customer_analysis(st.session_state.customer_stats, st.session_state.df)
                    
                with tab2:
                    self.render_rfm_analysis(st.session_state.customer_stats)
                    
                with tab3:
                    self.render_segment_analysis(st.session_state.customer_stats, st.session_state.df)
                    
                with tab4:
                    self.render_product_analysis(st.session_state.df)
                    
                with tab5:
                    self.render_advanced_product_analysis(st.session_state.df, st.session_state.customer_stats)
                    
                with tab6:
                    analyzer = LuxuryRetailAnalyzer(st.session_state.df)
                    analyzer.df = st.session_state.df
                    analyzer.customer_stats = st.session_state.customer_stats
                    self.render_insights(analyzer)

                with tab7:  # Codice per Analisi Retention
                    self.render_retention_analysis(st.session_state.customer_stats)

                with tab8:
                    self.render_yoy_analysis(st.session_state.df)

                with tab9:
                    if st.session_state.spark_df is None:
                        try:
                            st.session_state.spark_df = load_data_with_spark(st.session_state.df)
                            st.session_state.spark_results = perform_spark_analysis(st.session_state.spark_df)
                        except Exception as e:
                            st.error(f"Errore nel caricamento dei dati in Spark: {str(e)}")
                            return
                    
                    render_spark_analysis(st.session_state.spark_results)
                    
            except Exception as e:
                st.error(f"Si è verificato un errore nell'analisi: {str(e)}")
        
        # Render sidebar
        self.render_sidebar()

if __name__ == "__main__":

    dashboard = LuxuryRetailDashboard()
    dashboard.run()