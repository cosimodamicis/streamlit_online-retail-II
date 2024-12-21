"""
Streamlit dashboard per analisi luxury retail.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from analyzer import LuxuryRetailAnalyzer
import os
from scipy import stats

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
        
        # Segmentazione clienti
        segment_dist = customer_stats['customer_segment'].value_counts()
        fig_segments = px.pie(
            values=segment_dist.values,
            names=segment_dist.index,
            title="Distribuzione Segmenti Cliente"
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
            fig_value.update_layout(yaxis_type="log")
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
            st.plotly_chart(fig_avg, use_container_width=True)
    
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
        # Tabella riepilogativa con statistiche descrittive Monetary per segmento (per Cliente)
        st.subheader("Statistiche Descrittive Valore Speso per Segmento (per Cliente)")

        # Calcolo statistiche descrittive
        monetary_stats = customer_stats.groupby('customer_segment')['total_spend'].describe()

        # Calcolo il totale e lo inserisco dopo count
        segment_totals = customer_stats.groupby('customer_segment')['total_spend'].sum()
        monetary_stats.insert(1, 'Total', segment_totals)

        # Ordino per Total decrescente e resetto l'indice
        monetary_stats = monetary_stats.sort_values('Total', ascending=False).reset_index()

        # Formatto i valori monetari
        for col in ['Total', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
            monetary_stats[col] = monetary_stats[col].apply(lambda x: f"€{x:,.2f}")

        # Visualizzo la tabella
        st.dataframe(monetary_stats, use_container_width=True, hide_index=True)

        # Seconda tabella (nuova, basata sugli ordini)
        st.subheader("Statistiche Descrittive Valore Ordini per Segmento (per Ordine)")

        # Uniamo i dati dei segmenti con i dati degli ordini
        orders_by_segment = df.merge(
            customer_stats[['customer_segment']], 
            left_on='Customer ID', 
            right_index=True
        )

        # Calcoliamo le statistiche per ordine
        order_stats = orders_by_segment.groupby('customer_segment')['Total_Value'].describe()

        # Calcolo il totale e lo inserisco dopo count
        segment_order_totals = orders_by_segment.groupby('customer_segment')['Total_Value'].sum()
        order_stats.insert(1, 'Total', segment_order_totals)

        # Ordino per Total decrescente e resetto l'indice
        order_stats = order_stats.sort_values('mean', ascending=False).reset_index()

        # Formatto i valori monetari
        for col in ['Total', 'mean', 'std', 'min', '25%', '50%', '75%', 'max']:
            order_stats[col] = order_stats[col].apply(lambda x: f"€{x:,.2f}")

        # Visualizzo la tabella
        st.dataframe(order_stats, use_container_width=True, hide_index=True)

        # Aggiungiamo un commento esplicativo
        st.markdown("""
        **Confronto tra le tabelle:**
        - La prima tabella mostra il comportamento lifetime dei clienti (quanto spende mediamente un cliente di ogni segmento)
        - La seconda tabella mostra il comportamento per singolo ordine (quanto vale mediamente un ordine per ogni segmento)

        Questo confronto evidenzia che mentre i clienti Loyal potrebbero spendere di più nel loro lifetime totale,
        i clienti VIP tendono ad effettuare ordini di valore superiore ma potenzialmente meno frequenti.
        """)

        # Spiegazione scelta del test
        st.markdown("""
        ### Test Statistico delle Differenze tra Segmenti
        Utilizziamo il test di Kruskal-Wallis (ANOVA non parametrico) per validare le differenze osservate tra i segmenti, 
        in quanto i dati di spesa tipicamente violano l'assunzione di omoschedasticità richiesta dall'ANOVA parametrica.
        """)

        # Eseguiamo il test di Kruskal-Wallis sia per lifetime value che per order value
        
        # Test per lifetime value
        h_stat_lifetime, p_val_lifetime = stats.kruskal(
            *[group['total_spend'].values for name, group in customer_stats.groupby('customer_segment')]
        )

        # Test per order value
        h_stat_order, p_val_order = stats.kruskal(
            *[group['Total_Value'].values for name, group in orders_by_segment.groupby('customer_segment')]
        )

        # Creiamo una tabella con i risultati
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

        # Insight sui risultati
        st.markdown("""
        **Interpretazione dei Risultati:**
        I test confermano che esistono differenze statisticamente significative (p < 0.05) sia nel lifetime value che nel valore degli ordini 
        tra i diversi segmenti di clienti. Questo supporta scientificamente la nostra segmentazione e conferma che i pattern di spesa 
        osservati non sono casuali ma riflettono reali differenze nel comportamento d'acquisto.
        """)

        # Test statistici (codice precedente invariato)
        st.markdown("""
        ### Test Statistico delle Differenze tra Segmenti
        Utilizziamo il test di Kruskal-Wallis (ANOVA non parametrico) per validare le differenze osservate tra i segmenti, 
        in quanto i dati di spesa tipicamente violano l'assunzione di omoschedasticità richiesta dall'ANOVA parametrica.
        """)

        # Test e tabella risultati (come prima)
        # ...

        # Ora aggiungiamo le visualizzazioni con Plotly
        st.subheader("Visualizzazione delle Differenze tra Segmenti")

        # Creiamo due box plot affiancati
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
                height=500
            )
            fig_lifetime.update_layout(yaxis_type="log")
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
                height=500
            )
            fig_order.update_layout(yaxis_type="log")
            st.plotly_chart(fig_order, use_container_width=True)

        # Aggiungiamo un violin plot per una visualizzazione più dettagliata
        st.subheader("Distribuzione Dettagliata dei Valori")

        fig_violin = go.Figure()

        # Violin plot per lifetime value
        for segment in customer_stats['customer_segment'].unique():
            fig_violin.add_trace(go.Violin(
                x=[f"{segment} - Lifetime" for _ in range(len(customer_stats[customer_stats['customer_segment'] == segment]))],
                y=customer_stats[customer_stats['customer_segment'] == segment]['total_spend'],
                name=f"{segment} - Lifetime",
                side='positive',
                meanline_visible=True,
                box_visible=True
            ))
            
            # Violin plot per order value
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
            height=600
        )
        fig_violin.update_layout(yaxis_type="log")
        st.plotly_chart(fig_violin, use_container_width=True)

        # Aggiungiamo una spiegazione delle visualizzazioni
        st.markdown("""
        **Interpretazione delle Visualizzazioni:**
        - I box plot mostrano chiaramente la differenza nella distribuzione dei valori tra i segmenti
        - Il violin plot permette di vedere la forma completa della distribuzione, evidenziando come:
        - I clienti VIP hanno ordini di valore più alto (Order Value)
        - I clienti Loyal hanno un lifetime value complessivo maggiore
        - La dispersione dei valori è diversa tra i segmenti
        
        Queste visualizzazioni supportano i risultati del test Kruskal-Wallis, mostrando non solo 
        che le differenze sono statisticamente significative, ma anche come queste differenze 
        si manifestano nella distribuzione dei valori.
        """)
            
    def render_product_analysis(self, df):
        """Render dell'analisi prodotti"""
        st.header("🛍️ Analisi Prodotti")
        
        try:
            # Assicuriamoci che StockCode sia una stringa
            df = df.copy()
            df['StockCode'] = df['StockCode'].astype(str)
            
            # Performance per categoria
            col1, col2 = st.columns(2)
            
            with col1:
                category_perf = df.groupby('price_segment')['Total_Value'].sum()
                fig_cat = px.pie(
                    values=category_perf.values,
                    names=category_perf.index,
                    title="Revenue per Categoria"
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
                
            # Top prodotti
            st.subheader("📈 Top Prodotti")
            
            # Aggreghiamo prima i dati
            top_products = (df.groupby('StockCode').agg({
                'Description': 'first',
                'price_segment': 'first',
                'Total_Value': 'sum',
                'Quantity': 'sum'
            })
            .sort_values('Total_Value', ascending=False)
            .head(10)
            .reset_index())
            
            # Formatta la tabella
            formatted_products = pd.DataFrame({
                'Codice': top_products['StockCode'],
                'Descrizione': top_products['Description'],
                'Categoria': top_products['price_segment'],
                'Revenue Totale': top_products['Total_Value'].map('€{:,.2f}'.format),
                'Quantità': top_products['Quantity'].map('{:,}'.format)
            })
            
            st.dataframe(formatted_products, use_container_width=True)
            
        except Exception as e:
            st.error(f"Errore nell'analisi prodotti: {str(e)}")
            
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
                tab1, tab2, tab3, tab4, tab5 = st.tabs([
                    "👥 Analisi Cliente",
                    "📊 Analisi RFM",
                    "🎯 Analisi Segmenti",
                    "🛍️ Analisi Prodotti",
                    "💡 Business Insights"
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
                    analyzer = LuxuryRetailAnalyzer(st.session_state.df)
                    analyzer.df = st.session_state.df
                    analyzer.customer_stats = st.session_state.customer_stats
                    self.render_insights(analyzer)
                    
            except Exception as e:
                st.error(f"Si è verificato un errore nell'analisi: {str(e)}")
        
        # Render sidebar
        self.render_sidebar()

if __name__ == "__main__":
    dashboard = LuxuryRetailDashboard()
    dashboard.run()