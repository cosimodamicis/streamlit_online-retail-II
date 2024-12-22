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
import matplotlib.pyplot as plt



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
            
    def render_header(self):
        """Render dell'header"""
        st.title("💎 Luxury Retail Analytics")
        st.markdown("Dashboard per analisi dati retail nel settore lusso")
            
    def load_default_dataset(self):
        """Carica il dataset di default dalla cartella data"""
        try:
            default_path = os.path.join('data', 'online_retail_II.xlsx')
            df = pd.read_excel(default_path)
            analyzer = LuxuryRetailAnalyzer(df)
            df, customer_stats = analyzer.process_data()
            
            # Salva in session state
            st.session_state.df = df
            st.session_state.customer_stats = customer_stats
            st.session_state.data_loaded = True
            
            st.success("Dataset di default caricato con successo!")
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
        st.dataframe(pivot_table_display, use_container_width=True, hide_index=True)

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
            ### Interpretazione del Grafico: Revenue per Segmento e Stagione

            - **Luxury**:
            - Il revenue è distribuito uniformemente tra le stagioni, senza picchi evidenti.
            - Questo conferma che non ci sono differenze significative tra stagioni (**p = 0.1147**, Test Statistico).
            - Tuttavia, il segmento segue strettamente il trend complessivo delle vendite stagionali (**Rho = 0.9918**, Correlazione Lineare).

            - **Premium**:
            - Il revenue varia significativamente tra le stagioni, con il picco in **Fall** e il valore minimo in **Summer**.
            - Questo è confermato dalle differenze significative tra stagioni rilevate dal Test Statistico (**p = 0**).
            - La forte correlazione lineare (**Rho = 0.981**) indica che le variazioni interne seguono comunque il trend generale.

            - **Budget**:
            - Il revenue è più alto in **Fall** rispetto a **Summer**, mostrando variazioni stagionali pronunciate.
            - Differenze significative tra stagioni sono confermate dal Test Statistico (**p = 0**).
            - La correlazione lineare alta (**Rho = 0.9551**) suggerisce che, nonostante le variazioni interne, il segmento segue il trend complessivo.

            - **Regular**:
            - Le differenze stagionali sono evidenti, con il revenue massimo in **Fall** e minimo in **Summer**.
            - Le differenze sono significative secondo il Test Statistico (**p = 0**).
            - La correlazione lineare (**Rho = 0.9771**) mostra che il segmento è ben allineato con le vendite complessive.

            ### Conclusione Generale
            Il segmento **Luxury** si distingue per la sua uniformità stagionale, mentre gli altri segmenti presentano variazioni significative tra stagioni. Tuttavia, tutti i segmenti seguono strettamente le tendenze generali delle vendite stagionali.
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

        # Creazione del target `is_retained` con criterio migliorato
        customer_stats['is_retained'] = (
            (customer_stats['last_purchase'] > pd.Timestamp.now() - pd.Timedelta(days=180)) &
            (customer_stats['num_orders'] > 1)
        ).astype(int)

        # Controlla la distribuzione della variabile target
        st.write("Distribuzione Target:")
        st.bar_chart(customer_stats['is_retained'].value_counts())

        # Selezione delle feature e del target
        features = customer_stats[['recency', 'frequency', 'total_spend', 'avg_order_value']]
        target = customer_stats['is_retained']

        # Verifica statistiche delle feature
        st.write("Statistiche delle Variabili di Input:")
        st.dataframe(features.describe())

        # Divisione train/test
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

        # Decision Tree Model con profondità maggiore
        dt_model = DecisionTreeClassifier(max_depth=10, random_state=42)
        dt_model.fit(X_train, y_train)

        # Predizioni
        y_pred = dt_model.predict(X_test)

        # Visualizzazione dei risultati
        st.subheader("Performance del Modello")
        st.text("Classification Report:")
        st.text(classification_report(y_test, y_pred))

        # Accuracy
        accuracy = accuracy_score(y_test, y_pred)
        st.metric("Accuracy", f"{accuracy * 100:.2f}%")

        # Visualizzazione dell'albero decisionale
        st.subheader("Albero Decisionale")
        fig, ax = plt.subplots(figsize=(12, 8))
        plot_tree(dt_model, feature_names=features.columns, class_names=['Non Retained', 'Retained'], filled=True, ax=ax)
        st.pyplot(fig)

        # Insight sulle feature importance
        st.subheader("Importanza delle Variabili")
        feature_importance = pd.DataFrame({
            'Feature': features.columns,
            'Importance': dt_model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        st.bar_chart(feature_importance.set_index('Feature'))

    
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
            
            **Funzionalità**:
            - Analisi cliente
            - Segmentazione RFM
            - Analisi prodotti
            - Insights di business
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
                
                            # Tabs per le analisi
                tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
                    "👥 Analisi Cliente",
                    "📊 Analisi RFM",
                    "🎯 Analisi Segmenti",
                    "🛍️ Analisi Prodotti",
                    "🔍 Analisi Prodotti Avanzata",
                    "💡 Business Insights",
                    "🔄 Analisi Retention"  # Nuova tab per Analisi Retentio
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
            except Exception as e:
                st.error(f"Si è verificato un errore nell'analisi: {str(e)}")
        
        # Render sidebar
        self.render_sidebar()

if __name__ == "__main__":

    dashboard = LuxuryRetailDashboard()
    dashboard.run()