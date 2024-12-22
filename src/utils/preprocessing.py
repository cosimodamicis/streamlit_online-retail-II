"""
Funzioni di preprocessing per i dati di retail luxury.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict
from datetime import datetime

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applica preprocessing base al dataset.
    """
    df = df.copy()
    df = df[
        (df['Quantity'] > 0) &
        (df['Price'] > 0) &
        (pd.notnull(df['Customer ID']))
    ]
    df['Total_Value'] = df['Quantity'] * df['Price']
    df['Customer ID'] = df['Customer ID'].astype(str)
    df['StockCode'] = df['StockCode'].astype(str)
    return df

def create_customer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea features aggregate a livello cliente.
    """
    agg_dict = {
        'Invoice': 'nunique',
        'Total_Value': ['sum', 'mean'],
        'InvoiceDate': ['min', 'max'],
        'StockCode': 'nunique',
        'Quantity': 'sum'
    }
    
    customer_stats = df.groupby('Customer ID').agg(agg_dict)
    customer_stats.columns = [
        'num_orders',
        'total_spend',
        'avg_order_value',
        'first_purchase',
        'last_purchase',
        'unique_items',
        'total_items'
    ]
    
    customer_stats['recency'] = (
        customer_stats['last_purchase'] - 
        customer_stats['first_purchase']
    ).dt.days

    customer_stats['frequency'] = customer_stats['num_orders'] / customer_stats['recency'].clip(lower=1)
    
    # RFM Scoring usando metodo alternativo invece di qcut
    def assign_score(series, reverse=False):
        quartiles = series.quantile([0.25, 0.5, 0.75])
        scores = pd.Series(index=series.index, data=1)
        
        if reverse:
            scores[series <= quartiles[0.75]] = 2
            scores[series <= quartiles[0.50]] = 3
            scores[series <= quartiles[0.25]] = 4
        else:
            scores[series >= quartiles[0.25]] = 2
            scores[series >= quartiles[0.50]] = 3
            scores[series >= quartiles[0.75]] = 4
            
        return scores
    
    # Calcola scores
    customer_stats['r_score'] = assign_score(customer_stats['recency'], reverse=True)
    customer_stats['f_score'] = assign_score(customer_stats['frequency'])
    customer_stats['m_score'] = assign_score(customer_stats['total_spend'])
    
    # Segmentazione
    def get_segment(row):
        try:
            score = (row['r_score'] + row['f_score'] + row['m_score']) / 3
            if score >= 3.5:
                return 'VIP'
            elif score >= 2.5:
                return 'Loyal'
            elif score >= 1.5:
                return 'Regular'
            else:
                return 'At Risk'
        except:
            return 'New'
            
    customer_stats['customer_segment'] = customer_stats.apply(get_segment, axis=1)
    return customer_stats

def consolidate_excel_sheets(file_path: str) -> pd.DataFrame:
    """
    Legge tutti i fogli di un file Excel e li consolida in un unico DataFrame.
    
    Args:
        file_path (str): Percorso del file Excel.
    
    Returns:
        pd.DataFrame: DataFrame consolidato con i dati di tutti i fogli.
    """
    try:
        # Leggi tutti i fogli
        sheets = pd.read_excel(file_path, sheet_name=None)  # Legge tutti i fogli come un dizionario {sheet_name: DataFrame}
        
        # Consolida i fogli in un unico DataFrame
        consolidated_df = pd.concat(sheets.values(), ignore_index=True)
        return consolidated_df
    except Exception as e:
        raise ValueError(f"Errore nel consolidamento dei fogli Excel: {str(e)}")
    
def categorize_products(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorizza prodotti in segmenti di prezzo.
    """
    df = df.copy()
    product_stats = df.groupby('StockCode').agg({
        'Description': 'first',
        'Price': 'mean',
        'Quantity': 'sum'
    })
    
    # Usa percentili fissi invece di qcut
    price_percentiles = product_stats['Price'].quantile([0.25, 0.5, 0.75])
    
    def get_price_segment(price):
        if price <= price_percentiles[0.25]:
            return 'Budget'
        elif price > price_percentiles[0.25] and price <= price_percentiles[0.5]:  # modifica qui
            return 'Regular'
        elif price > price_percentiles[0.5] and price <= price_percentiles[0.75]:  # e qui
            return 'Premium'
        else:  # implicitamente price > price_percentiles[0.75]
            return 'Luxury'
            
    product_stats['price_segment'] = product_stats['Price'].apply(get_price_segment)
    
    df = df.merge(
        product_stats[['price_segment']], 
        left_on='StockCode', 
        right_index=True,
        how='left'
    )
    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggiunge features temporali al dataset.
    """
    df = df.copy()
    
    df['month'] = df['InvoiceDate'].dt.month
    df['day_of_week'] = df['InvoiceDate'].dt.dayofweek
    df['hour'] = df['InvoiceDate'].dt.hour
    
    # Definizione diretta delle stagioni
    def get_season(month):
        if month in [12, 1, 2]:
            return 'Winter'
        elif month in [3, 4, 5]:
            return 'Spring'
        elif month in [6, 7, 8]:
            return 'Summer'
        else:  # 9, 10, 11
            return 'Fall'
            
    df['season'] = df['month'].apply(get_season)
    
    # Holiday season
    df['is_holiday_season'] = (
        (df['month'] == 12) | 
        ((df['month'] == 11) & (df['InvoiceDate'].dt.day >= 25))
    )
    
    return df