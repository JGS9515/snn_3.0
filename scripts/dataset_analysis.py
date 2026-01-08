#!/usr/bin/env python3
"""
Dataset Analysis Script for SNN 3.0 Project
===========================================

Este script analiza los datasets utilizados en el proyecto SNN 3.0 y verifica
la información proporcionada en la documentación.

Datasets analizados:
- IOPS (KPI de servicios)
- CalIt2 (flujos de entrada/salida en edificio CalIt2)

Autor: Javier González Santos
Fecha: Diciembre 2024
"""

import pandas as pd
import numpy as np
import os
import sys
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
# import seaborn as sns  # Comentado porque no está disponible

# Configuración para visualizaciones
plt.style.use('default')
# sns.set_palette("husl")  # Comentado porque seaborn no está disponible

class DatasetAnalyzer:
    """Clase principal para el análisis de datasets."""

    def __init__(self, base_path="Nuevos datasets"):
        """Inicializar el analizador con la ruta base de los datasets."""
        self.base_path = Path(base_path)
        self.results = {}

    def analyze_calit2_dataset(self):
        """Analizar el dataset CalIt2."""
        print("🔍 Analizando dataset CalIt2...")

        # Cargar datos
        calit2_path = self.base_path / "Callt2" / "preliminar" / "train_label_filled.csv"
        df_calit2 = pd.read_csv(calit2_path)

        # Análisis básico
        total_observations = len(df_calit2)
        unique_timestamps = df_calit2['timestamp'].nunique()
        anomaly_percentage = (df_calit2['label'].sum() / total_observations) * 100

        # Análisis temporal
        df_calit2['datetime'] = pd.to_datetime(df_calit2['timestamp'], unit='s')
        time_range = df_calit2['datetime'].max() - df_calit2['datetime'].min()
        weeks = time_range.days / 7

        # Estadísticas de valores
        value_stats = df_calit2['value'].describe()

        # Verificar frecuencia de muestreo (cada 30 minutos)
        time_diffs = df_calit2['timestamp'].diff().dropna()
        sampling_intervals = time_diffs.value_counts()
        main_interval = sampling_intervals.index[0]  # debería ser 1800 segundos (30 min)

        # Análisis de particionado temporal
        mid_timestamp = df_calit2['timestamp'].median()
        train_size = len(df_calit2[df_calit2['timestamp'] <= mid_timestamp])
        test_size = len(df_calit2[df_calit2['timestamp'] > mid_timestamp])

        self.results['CalIt2'] = {
            'total_observations': total_observations,
            'unique_timestamps': unique_timestamps,
            'anomaly_percentage': anomaly_percentage,
            'time_range_weeks': weeks,
            'sampling_interval_seconds': main_interval,
            'value_stats': value_stats.to_dict(),
            'train_observations': train_size,
            'test_observations': test_size,
            'train_test_ratio': train_size / test_size if test_size > 0 else 0,
            'expected_observations': 10080,  # según documentación
            'expected_anomaly_percentage': 24.80,  # según documentación
            'expected_weeks': 15,  # según documentación
            'expected_sampling_minutes': 30  # según documentación
        }

        return self.results['CalIt2']

    def analyze_iops_dataset(self):
        """Analizar el dataset IOPS."""
        print("🔍 Analizando dataset IOPS...")

        # Cargar datos de entrenamiento
        train_path = self.base_path / "iops" / "preliminar" / "train.csv"
        df_train = pd.read_csv(train_path)

        # Cargar datos de prueba
        test_path = self.base_path / "iops" / "preliminar" / "test.csv"
        df_test = pd.read_csv(test_path)

        # Combinar para análisis global
        df_iops = pd.concat([df_train, df_test], ignore_index=True)

        # Análisis básico
        total_observations = len(df_iops)
        unique_kpis = df_iops['KPI ID'].nunique()
        anomaly_percentage = (df_iops['label'].sum() / total_observations) * 100

        # Análisis temporal
        df_iops['datetime'] = pd.to_datetime(df_iops['timestamp'], unit='s')
        time_range = df_iops['datetime'].max() - df_iops['datetime'].min()
        days = time_range.days

        # Verificar frecuencia de muestreo (cada 1 minuto)
        time_diffs = df_iops['timestamp'].diff().dropna()
        sampling_intervals = time_diffs.value_counts()
        main_interval = sampling_intervals.index[0]  # debería ser 60 segundos (1 min)

        # Análisis por KPI
        kpi_stats = df_iops.groupby('KPI ID').agg({
            'value': ['count', 'mean', 'std', 'min', 'max'],
            'label': ['sum', 'mean']
        }).round(4)

        # Análisis de particionado temporal
        mid_timestamp = df_iops['timestamp'].median()
        train_size = len(df_train)
        test_size = len(df_test)

        self.results['IOPS'] = {
            'total_observations': total_observations,
            'unique_kpis': unique_kpis,
            'anomaly_percentage': anomaly_percentage,
            'time_range_days': days,
            'sampling_interval_seconds': main_interval,
            'kpi_stats': kpi_stats.to_dict(),
            'train_observations': train_size,
            'test_observations': test_size,
            'train_test_ratio': train_size / test_size if test_size > 0 else 0,
            'expected_observations': 2788680,  # según documentación
            'expected_kpis': 26,  # según documentación
            'expected_anomaly_percentage': 1.92,  # según documentación
            'expected_sampling_minutes': 1  # según documentación
        }

        return self.results['IOPS']

    def generate_summary_report(self):
        """Generar reporte resumen de la verificación."""
        print("\n" + "="*80)
        print("📊 REPORTE DE VERIFICACIÓN DE DATASETS")
        print("="*80)

        for dataset_name, data in self.results.items():
            print(f"\n🔹 DATASET: {dataset_name}")
            print("-" * 40)

            if dataset_name == 'CalIt2':
                print(f"📈 Observaciones totales: {data['total_observations']:,} (esperado: {data['expected_observations']:,})")
                print(f"✅ Verificación: {'✓' if abs(data['total_observations'] - data['expected_observations']) <= 1 else '✗'}")

                print(".2f")
                print(".2f")

                print(".1f")
                print(f"✅ Verificación: {'✓' if abs(data['time_range_weeks'] - data['expected_weeks']) <= 0.1 else '✗'}")

                sampling_minutes = data['sampling_interval_seconds'] / 60
                print(".0f")
                print(f"✅ Verificación: {'✓' if abs(sampling_minutes - data['expected_sampling_minutes']) <= 1 else '✗'}")

                print("\n📊 Estadísticas de valores:")
                print(f"   • Media: {data['value_stats']['mean']:.2f}")
                print(f"   • Desviación: {data['value_stats']['std']:.2f}")
                print(f"   • Mínimo: {data['value_stats']['min']}")
                print(f"   • Máximo: {data['value_stats']['max']}")

                print("\n📅 Particionado temporal:")
                print(f"   • Entrenamiento: {data['train_observations']:,} observaciones")
                print(f"   • Prueba: {data['test_observations']:,} observaciones")
                print(".2f")
                print(f"   • Ratio train/test: {data['train_test_ratio']:.2f}")

            elif dataset_name == 'IOPS':
                print(f"📈 Observaciones totales: {data['total_observations']:,} (esperado: {data['expected_observations']:,})")
                print(f"✅ Verificación: {'✓' if abs(data['total_observations'] - data['expected_observations']) <= 10000 else '✗'}")

                print(f"🏷️  KPIs únicos: {data['unique_kpis']} (esperado: {data['expected_kpis']})")
                print(f"✅ Verificación: {'✓' if data['unique_kpis'] == data['expected_kpis'] else '✗'}")

                print(".2f")
                print(".2f")

                sampling_minutes = data['sampling_interval_seconds'] / 60
                print(".0f")
                print(f"✅ Verificación: {'✓' if abs(sampling_minutes - data['expected_sampling_minutes']) <= 0.1 else '✗'}")

                print("\n📅 Particionado temporal:")
                print(f"   • Entrenamiento: {data['train_observations']:,} observaciones")
                print(f"   • Prueba: {data['test_observations']:,} observaciones")
                print(".2f")
                print(f"   • Ratio train/test: {data['train_test_ratio']:.2f}")

                print("\n📊 Estadísticas por KPI:")
                kpi_df = pd.DataFrame(data['kpi_stats'])
                print("   • Número de observaciones por KPI:")
                print(f"     - Media: {kpi_df[('value', 'count')].mean():.0f}")
                print(f"     - Mínimo: {kpi_df[('value', 'count')].min()}")
                print(f"     - Máximo: {kpi_df[('value', 'count')].max()}")

    def generate_additional_insights(self):
        """Generar insights adicionales sobre los datasets."""
        print("\n" + "="*80)
        print("🔍 INSIGHTS ADICIONALES")
        print("="*80)

        # Análisis comparativo
        print("\n📊 COMPARACIÓN ENTRE DATASETS:")
        print("-" * 40)

        if 'CalIt2' in self.results and 'IOPS' in self.results:
            calit2_data = self.results['CalIt2']
            iops_data = self.results['IOPS']

            print(f"🔹 Tamaño relativo: CalIt2/IOPS = {calit2_data['total_observations']/iops_data['total_observations']:.3f}")
            print(f"🔹 Ratio anomalías: CalIt2/IOPS = {calit2_data['anomaly_percentage']/iops_data['anomaly_percentage']:.1f}x")
            print(f"🔹 Frecuencia muestreo: CalIt2/IOPS = {(iops_data['sampling_interval_seconds']/calit2_data['sampling_interval_seconds']):.0f}x más frecuente en IOPS")

        # Recomendaciones
        print("\n💡 RECOMENDACIONES:")
        print("-" * 40)

        for dataset_name, data in self.results.items():
            print(f"\n🔹 {dataset_name}:")

            if data['anomaly_percentage'] < 5:
                print("   • Dataset altamente desbalanceado - considerar técnicas de oversampling")
            elif data['anomaly_percentage'] > 20:
                print("   • Dataset moderadamente balanceado - menor necesidad de técnicas de balanceo")

            if data['train_test_ratio'] != 1.0:
                print(f"   • Ratio train/test actual: {data['train_test_ratio']:.2f}")
            else:
                print("   • Particionado 50/50 temporal correcto ✓")

    def create_visualizations(self):
        """Crear visualizaciones básicas de los datasets."""
        print("\n📈 Generando visualizaciones...")

        try:
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Análisis Visual de Datasets SNN 3.0', fontsize=16)

            # Cargar datos para visualizaciones
            calit2_path = self.base_path / "Callt2" / "preliminar" / "train_label_filled.csv"
            df_calit2 = pd.read_csv(calit2_path)
            df_calit2['datetime'] = pd.to_datetime(df_calit2['timestamp'], unit='s')

            iops_path = self.base_path / "iops" / "preliminar" / "train.csv"
            df_iops = pd.read_csv(iops_path)
            df_iops['datetime'] = pd.to_datetime(df_iops['timestamp'], unit='s')

            # Gráfico 1: Distribución temporal CalIt2
            axes[0, 0].plot(df_calit2['datetime'], df_calit2['value'], alpha=0.7)
            axes[0, 0].set_title('Serie Temporal CalIt2')
            axes[0, 0].set_xlabel('Tiempo')
            axes[0, 0].set_ylabel('Valor')
            axes[0, 0].tick_params(axis='x', rotation=45)

            # Gráfico 2: Anomalías CalIt2
            anomaly_indices = df_calit2['label'] == 1
            axes[0, 1].plot(df_calit2['datetime'], df_calit2['value'], alpha=0.7, label='Normal')
            axes[0, 1].scatter(df_calit2['datetime'][anomaly_indices],
                              df_calit2['value'][anomaly_indices],
                              color='red', alpha=0.7, s=10, label='Anomalía')
            axes[0, 1].set_title('Anomalías en CalIt2')
            axes[0, 1].set_xlabel('Tiempo')
            axes[0, 1].set_ylabel('Valor')
            axes[0, 1].legend()
            axes[0, 1].tick_params(axis='x', rotation=45)

            # Gráfico 3: Distribución valores IOPS (muestra)
            sample_iops = df_iops.sample(min(10000, len(df_iops)))
            axes[1, 0].plot(sample_iops['datetime'], sample_iops['value'], alpha=0.7)
            axes[1, 0].set_title('Serie Temporal IOPS (muestra)')
            axes[1, 0].set_xlabel('Tiempo')
            axes[1, 0].set_ylabel('Valor')
            axes[1, 0].tick_params(axis='x', rotation=45)

            # Gráfico 4: Distribución de anomalías por KPI
            kpi_anomaly_rates = df_iops.groupby('KPI ID')['label'].mean().sort_values(ascending=False)
            top_10_kpis = kpi_anomaly_rates.head(10)
            axes[1, 1].bar(range(len(top_10_kpis)), top_10_kpis.values)
            axes[1, 1].set_title('Top 10 KPIs por Tasa de Anomalías')
            axes[1, 1].set_xlabel('KPI ID')
            axes[1, 1].set_ylabel('Tasa de Anomalías')
            axes[1, 1].set_xticks(range(len(top_10_kpis)))
            axes[1, 1].set_xticklabels([str(i+1) for i in range(len(top_10_kpis))])

            plt.tight_layout()
            plt.savefig('dataset_analysis_visualization.png', dpi=300, bbox_inches='tight')
            print("✅ Visualizaciones guardadas en 'dataset_analysis_visualization.png'")

        except Exception as e:
            print(f"⚠️  Error al generar visualizaciones: {e}")

def main():
    """Función principal del programa."""
    print("🚀 Iniciando análisis de datasets SNN 3.0")
    print("=" * 50)

    # Inicializar analizador
    analyzer = DatasetAnalyzer()

    try:
        # Analizar datasets
        analyzer.analyze_calit2_dataset()
        analyzer.analyze_iops_dataset()

        # Generar reportes
        analyzer.generate_summary_report()
        analyzer.generate_additional_insights()
        analyzer.create_visualizations()

        print("\n" + "="*80)
        print("✅ ANÁLISIS COMPLETADO EXITOSAMENTE")
        print("="*80)

    except Exception as e:
        print(f"❌ Error durante el análisis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
