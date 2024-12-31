"""
Modulo per l'integrazione di Spark nell'app Streamlit.
"""
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType, TimestampType
import pandas as pd
import streamlit as st

def create_spark_session():
    """
    Crea una sessione Spark minima compatibile con Streamlit Cloud.
    """
    spark = (SparkSession.builder
            .appName("LuxuryRetailAnalytics")
            .config("spark.driver.memory", "2g")
            .config("spark.sql.session.timeZone", "UTC")
            # Disabilita funzionalità Hadoop
            .config("spark.hadoop.fs.defaultFS", "file:///")
            .config("spark.driver.host", "localhost")
            .config("spark.driver.bindAddress", "127.0.0.1")
            .master("local[*]")
            .getOrCreate())
    
    return spark

def load_data_with_spark(file_path):
    """
    Carica i dati Excel usando Spark.
    """
    # Schema dei dati
    schema = StructType([
        StructField("Invoice", StringType(), True),
        StructField("StockCode", StringType(), True),
        StructField("Description", StringType(), True),
        StructField("Quantity", IntegerType(), True),
        StructField("InvoiceDate", TimestampType(), True),
        StructField("Price", DoubleType(), True),
        StructField("Customer ID", StringType(), True),
        StructField("Country", StringType(), True)
    ])
    
    # Leggi il file Excel come pandas e converti in Spark DataFrame
    pdf = pd.read_excel(file_path)
    spark = create_spark_session()
    df = spark.createDataFrame(pdf, schema=schema)
    
    return df

def perform_spark_analysis(spark_df):
    """
    Esegue analisi avanzate usando Spark.
    """
    # 1. Calcola metriche aggregate con window functions
    window_monthly = Window.partitionBy(F.month('InvoiceDate')).orderBy('InvoiceDate')
    window_customer = Window.partitionBy('Customer ID').orderBy('InvoiceDate')
    
    enriched_df = spark_df.withColumn(
        "Total_Value", 
        F.col("Quantity") * F.col("Price")
    ).withColumn(
        "Running_Total", 
        F.sum("Total_Value").over(window_monthly)
    ).withColumn(
        "Customer_Running_Total",
        F.sum("Total_Value").over(window_customer)
    )
    
    # 2. Analisi RFM con Spark
    current_date = spark_df.agg(F.max("InvoiceDate")).collect()[0][0]
    
    rfm_analysis = spark_df.groupBy("Customer ID").agg(
        F.datediff(F.lit(current_date), F.max("InvoiceDate")).alias("recency"),
        F.countDistinct("Invoice").alias("frequency"),
        F.sum(F.col("Quantity") * F.col("Price")).alias("monetary")
    )
    
    # 3. Calcolo dei quartili per segmentazione
    quartiles = rfm_analysis.select(
        F.percentile_approx("recency", [0.25, 0.5, 0.75]).alias("r_quartiles"),
        F.percentile_approx("frequency", [0.25, 0.5, 0.75]).alias("f_quartiles"),
        F.percentile_approx("monetary", [0.25, 0.5, 0.75]).alias("m_quartiles")
    ).collect()[0]
    
    # 4. Customer Lifetime Value Analysis
    clv_analysis = enriched_df.groupBy("Customer ID").agg(
        F.sum("Total_Value").alias("total_spend"),
        F.avg("Total_Value").alias("avg_transaction"),
        F.count("Invoice").alias("num_transactions"),
        F.datediff(
            F.max("InvoiceDate"), 
            F.min("InvoiceDate")
        ).alias("customer_lifetime_days")
    ).withColumn(
        "clv_score",
        F.col("total_spend") / F.col("customer_lifetime_days")
    )
    
    return {
        "enriched_df": enriched_df,
        "rfm_analysis": rfm_analysis,
        "clv_analysis": clv_analysis,
        "quartiles": quartiles
    }

def render_spark_analysis(spark_results):
    """
    Renderizza i risultati dell'analisi Spark in Streamlit.
    """
    st.header("🚀 Analisi con Apache Spark")
    
    # 1. Overview delle performance Spark
    st.subheader("Performance Overview")
    col1, col2 = st.columns(2)
    
    with col1:
        # Converti Spark DataFrame in Pandas per la visualizzazione
        perf_metrics = spark_results["enriched_df"].select(
            F.sum("Total_Value").alias("total_revenue"),
            F.countDistinct("Customer ID").alias("unique_customers"),
            F.countDistinct("Invoice").alias("total_transactions")
        ).toPandas()
        
        st.metric(
            "Revenue Totale (Spark)", 
            f"€{perf_metrics['total_revenue'].iloc[0]:,.2f}"
        )
        st.metric(
            "Clienti Unici (Spark)", 
            f"{perf_metrics['unique_customers'].iloc[0]:,}"
        )
    
    with col2:
        st.metric(
            "Transazioni Totali (Spark)", 
            f"{perf_metrics['total_transactions'].iloc[0]:,}"
        )
        st.metric(
            "AOV (Spark)", 
            f"€{perf_metrics['total_revenue'].iloc[0] / perf_metrics['total_transactions'].iloc[0]:,.2f}"
        )
    
    # 2. Customer Lifetime Value Analysis
    st.subheader("Customer Lifetime Value Analysis (Spark)")
    clv_pd = spark_results["clv_analysis"].toPandas()
    
    # Plot CLV distribution
    fig = px.histogram(
        clv_pd,
        x="clv_score",
        nbins=50,
        title="Distribuzione Customer Lifetime Value"
    )
    fig.update_layout(
        xaxis_title="CLV Score (€/giorno)",
        yaxis_title="Numero Clienti"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # 3. RFM Insights
    st.subheader("RFM Insights (Spark)")
    rfm_pd = spark_results["rfm_analysis"].toPandas()
    
    # Plot RFM distributions
    col1, col2 = st.columns(2)
    with col1:
        fig_r = px.box(
            rfm_pd,
            y="recency",
            title="Distribuzione Recency"
        )
        st.plotly_chart(fig_r, use_container_width=True)
    
    with col2:
        fig_m = px.box(
            rfm_pd,
            y="monetary",
            title="Distribuzione Monetary"
        )
        st.plotly_chart(fig_m, use_container_width=True)
    
    st.markdown("""
    ### Vantaggi dell'Analisi con Spark
    
    1. **Scalabilità**
       - Processamento distribuito dei dati
       - Gestione efficiente della memoria
       - Parallelizzazione automatica delle operazioni
    
    2. **Performance**
       - Ottimizzazione delle query con Catalyst Optimizer
       - Lazy evaluation per operazioni efficienti
       - Caching intelligente dei DataFrame
    
    3. **Funzionalità Avanzate**
       - Window Functions per analisi temporali
       - UDF per logiche custom
       - APIs SQL-like per analisi complesse
    """)