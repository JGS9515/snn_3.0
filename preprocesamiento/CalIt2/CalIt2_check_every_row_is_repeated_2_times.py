import pandas as pd
import os

def CalIt2_check_every_row_is_repeated_2_times():
    
    # Ruta del archivo a revisar LINUX
    # input_path = f'/home/javier/Practicas/Nuevos datasets/Callt2/preliminar/train.csv'

    # Obtener la ruta base del script actual
    base_path = os.path.abspath(os.path.dirname(__file__))
    # Ruta relativa del archivo a revisar
    input_path = os.path.join(base_path, '..', '..', 'Nuevos datasets', 'Callt2', 'preliminar', 'train.csv')

    # Leer el archivo CSV
    df = pd.read_csv(input_path)

    # Ordenar las filas por la columna 'timestamp'
    df = df.sort_values(by='timestamp')

    # Contar las veces que se repite cada fila
    row_counts = df['timestamp'].value_counts()

    # Verificar si cada fila se repite 2 veces
    if all(row_counts == 2):
        print('Cada fila se repite 2 veces')
    else:
        print('No todas las filas se repiten 2 veces')


# CalIt2_check_every_row_is_repeated_2_times()
# Verificar antes si cada fila se repite 2 veces