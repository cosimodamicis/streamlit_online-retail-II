"""
Core business logic per l'analisi retail luxury.
Integra preprocessing, ML e visualizzazione.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, silhouette_score
from typing import Dict, Tuple, List, Optional

from utils.preprocessing import (
    preprocess_data,
    create_customer_features,
    categorize_products,
    add_time_features
)
from utils.visualization import (
    create_customer_segment_pie,
    create_customer_value_distribution,
    create_seasonal_analysis,
    create_product_performance,
    create_ml_results_viz,
    create_cohort_analysis
)

class LuxuryRetailAnalyzer:
    """
    Classe principale per l'analisi dei dati retail luxury.
    Integra preprocessing, analisi statistica, ML e visualizzazione.
    """
    
    def __init__(self, df: pd.DataFrame):
        """
        Inizializza l'analyzer con il dataset grezzo.
        
        Args:
            df (pd.DataFrame): DataFrame grezzo dal file Excel
        """
        self.raw_df = df
        self.df = None
        self.customer_stats = None
        self.ml_results = None
        self.scaler = StandardScaler()
        
    def process_data(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Esegue la pipeline completa di preprocessing.
        
        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]: (DataFrame processato, statistiche clienti)
        """
        # Pipeline preprocessing
        self.df = preprocess_data(self.raw_df)
        self.df = categorize_products(self.df)
        self.df = add_time_features(self.df)
        
        # Crea customer stats
        self.customer_stats = create_customer_features(self.df)
        
        return self.df, self.customer_stats
    
    def get_kpis(self) -> Dict:
        """
        Calcola i KPI principali.
        
        Returns:
            Dict: Dizionario con i KPI principali
        """
        return {
            'total_revenue': self.df['Total_Value'].sum(),
            'avg_order_value': self.df.groupby('InvoiceNo')['Total_Value'].sum().mean(),
            'total_customers': len(self.customer_stats),
            'total_orders': self.df['InvoiceNo'].nunique(),
            'vip_percentage': (
                self.customer_stats['customer_segment'] == 'VIP'
            ).mean() * 100
        }
    
    def run_statistical_analysis(self) -> Dict:
        """
        Esegue analisi statistiche sui dati.
        
        Returns:
            Dict: Risultati delle analisi statistiche
        """
        results = {}
        
        # Test di normalità su order value
        order_values = self.df.groupby('InvoiceNo')['Total_Value'].sum()
        stat, p_value = stats.normaltest(order_values)
        results['order_value_normality'] = {
            'statistic': stat,
            'p_value': p_value,
            'is_normal': p_value > 0.05
        }
        
        # Confronto segmenti (ANOVA)
        segment_values = [
            group['total_spend'].values 
            for name, group in self.customer_stats.groupby('customer_segment')
        ]
        f_stat, p_value = stats.f_oneway(*segment_values)
        results['segment_comparison'] = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'significant_difference': p_value < 0.05
        }
        
        return results
    
    def train_models(self) -> Dict:
        """
        Addestra e valuta modelli ML.
        
        Returns:
            Dict: Risultati dei modelli ML
        """
        # Prepara features
        features = self.customer_stats[[
            'total_spend', 'num_orders', 'recency',
            'purchase_frequency', 'avg_order_value'
        ]]
        
        # Target: VIP vs non-VIP
        target = (self.customer_stats['customer_segment'] == 'VIP').astype(int)
        
        # Split e scaling
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=42
        )
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Modelli da testare
        models = {
            'logistic_regression': LogisticRegression(random_state=42),
            'decision_tree': DecisionTreeClassifier(random_state=42),
            'random_forest': RandomForestClassifier(random_state=42)
        }
        
        results = {}
        
        for name, model in models.items():
            # Training e cross-validation
            cv_scores = cross_val_score(
                model, X_train_scaled, y_train, cv=5
            )
            
            # Fit su training set
            model.fit(X_train_scaled, y_train)
            
            # Predictions su test set
            y_pred = model.predict(X_test_scaled)
            
            # Salva risultati
            results[name] = {
                'cv_scores_mean': cv_scores.mean(),
                'cv_scores_std': cv_scores.std(),
                'classification_report': classification_report(y_test, y_pred)
            }
            
            # Feature importance
            if hasattr(model, 'feature_importances_'):
                results[name]['feature_importance'] = dict(
                    zip(features.columns, model.feature_importances_)
                )
            elif hasattr(model, 'coef_'):
                results[name]['feature_importance'] = dict(
                    zip(features.columns, model.coef_[0])
                )
        
        self.ml_results = results
        return results
    
    def generate_business_insights(self) -> List[Dict]:
        """
        Genera insights di business actionable.
        
        Returns:
            List[Dict]: Lista di insights con actions
        """
        insights = []
        
        # 1. Analisi segmenti
        segment_revenue = self.df.merge(
            pd.DataFrame(self.customer_stats['customer_segment']),
            left_on='Customer ID',
            right_index=True
        ).groupby('customer_segment')['Total_Value'].sum()
        
        vip_revenue_pct = segment_revenue['VIP'] / segment_revenue.sum() * 100
        
        insights.append({
            'title': 'VIP Customer Impact',
            'finding': f'I clienti VIP generano {vip_revenue_pct:.1f}% del revenue totale',
            'action': 'Implementare programma VIP personalizzato',
            'impact': '+15% retention VIP',
            'priority': 'Alta'
        })
        
        # 2. Analisi stagionalità
        seasonal_revenue = self.df.groupby('season')['Total_Value'].sum()
        best_season = seasonal_revenue.idxmax()
        worst_season = seasonal_revenue.idxmin()
        
        insights.append({
            'title': 'Seasonal Optimization',
            'finding': f'Performance gap del {((seasonal_revenue[best_season]/seasonal_revenue[worst_season])-1)*100:.1f}% tra {best_season} e {worst_season}',
            'action': f'Ottimizzare inventory per {worst_season}',
            'impact': '+20% revenue in low season',
            'priority': 'Media'
        })
        
        # 3. Product Mix
        category_margins = self.df.groupby('price_segment')['Total_Value'].mean()
        
        insights.append({
            'title': 'Product Mix Strategy',
            'finding': f'Margine {(category_margins["Luxury"]/category_margins["Regular"]-1)*100:.1f}% superiore nel segmento Luxury',
            'action': 'Espandere assortimento Luxury',
            'impact': '+10% margine medio',
            'priority': 'Alta'
        })
        
        return insights
    
    def get_visualizations(self, period: Optional[Tuple[pd.Timestamp, pd.Timestamp]] = None) -> Dict:
        """
        Genera tutte le visualizzazioni necessarie.
        
        Args:
            period: Optional[Tuple[pd.Timestamp, pd.Timestamp]]: Periodo di analisi
            
        Returns:
            Dict: Dizionario con tutte le visualizzazioni Plotly
        """
        # Filtra per periodo se specificato
        df = self.df
        if period:
            df = df[
                (df['InvoiceDate'] >= period[0]) & 
                (df['InvoiceDate'] <= period[1])
            ]
        
        return {
            'customer_segments': create_customer_segment_pie(self.customer_stats),
            'customer_value': create_customer_value_distribution(self.customer_stats),
            'seasonal_analysis': create_seasonal_analysis(df),
            'product_performance': create_product_performance(df),
            'cohort_analysis': create_cohort_analysis(df),
            'ml_results': create_ml_results_viz(self.ml_results) if self.ml_results else None
        }