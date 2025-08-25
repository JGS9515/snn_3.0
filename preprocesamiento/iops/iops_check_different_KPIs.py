import pandas as pd

def iops_check_different_KPIs():
    # Leer el archivo Excel
    # Nuevos datasets/iops/preliminar/test.csv
    df = pd.read_csv('/home/javier/Practicas/Nuevos datasets/iops/preliminar/train.csv')
    df.sort_values(by='timestamp')
    # Imprimir los nombres de las columnas para verificar
    # print("Nombres de las columnas:", df.columns)

    # Acceder a la columna 'KPI ID' por su índice (columna D es el índice 3)
    kpi_ids = df['KPI ID']

    # Encontrar los valores únicos en la columna 'KPI ID'
    unique_kpi_ids = kpi_ids.unique()

    # Contar el número de valores únicos
    num_unique_kpi_ids = len(unique_kpi_ids)

    # Imprimir el resultado
    print(f'Número de KPI distintos: {num_unique_kpi_ids}')

    # print('Valores únicos de KPI ID:')
    print(unique_kpi_ids)
    return {'unique_kpi_ids': unique_kpi_ids, 'df': df}