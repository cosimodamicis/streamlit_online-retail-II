"""
Streamlit dashboard per analisi luxury retail.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from analyzer import LuxuryRetailAnalyzer
import os

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
            
        # RFM Analysis
        st.subheader("📊 Analisi RFM")
        rfm_cols = ['r_score', 'f_score', 'm_score']
        rfm_means = customer_stats.groupby('customer_segment')[rfm_cols].mean()
        
        fig_rfm = go.Figure(data=[
            go.Scatterpolar(
                r=row,
                theta=['Recency', 'Frequency', 'Monetary'],
                name=segment
            ) for segment, row in rfm_means.iterrows()
        ])
        
        fig_rfm.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[1, 4])),
            showlegend=True,
            title="RFM Profile per Segmento"
        )
        
        st.plotly_chart(fig_rfm, use_container_width=True)
            
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
                tab1, tab2, tab3 = st.tabs([
                    "👥 Analisi Cliente",
                    "🛍️ Analisi Prodotti",
                    "💡 Business Insights"
                ])
                
                with tab1:
                    self.render_customer_analysis(st.session_state.customer_stats, st.session_state.df)
                    
                with tab2:
                    self.render_product_analysis(st.session_state.df)
                    
                with tab3:
                    # Creiamo un analyzer temporaneo per gli insights
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