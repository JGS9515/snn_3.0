# import csv
# import time
# import pandas as pd
# from datetime import datetime


# def CalIt2_fill_missing_timestamps():

#     df = pd.read_csv('/home/javier/Practicas/Nuevos datasets/Callt2/preliminar/train.csv')

import pandas as pd
import os

def CalIt2_fill_missing_timestamps(add_status_column: bool):

    # Ruta del archivo a revisar LINUX
    # input_path = f'/home/javier/Practicas/Nuevos datasets/Callt2/preliminar/train.csv'
    # output_path = f'/home/javier/Practicas/Nuevos datasets/Callt2/preliminar/train_filled.csv'

    # Ruta relativa del archivo a revisar
    base_path = os.path.abspath(os.path.dirname(__file__))
    input_path = os.path.join(base_path, '..', '..', 'Nuevos datasets', 'Callt2', 'preliminar', 'train.csv')
    output_path = os.path.join(base_path, '..', '..', 'Nuevos datasets', 'Callt2', 'preliminar', 'train_filled.csv')


    # Leer el archivo CSV
    df = pd.read_csv(input_path)

    # Asegurarse de que la columna 'timestamp' está en formato numérico
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')

    # Ordenar las filas por la columna 'timestamp'
    df = df.sort_values(by='timestamp')

    # Calcular la diferencia entre valores consecutivos de 'timestamp'
    # timestamp_diff = df['timestamp'].diff().dropna()

    # Crear una lista para almacenar las nuevas tuplas
    new_rows = []

    # Iterar sobre las diferencias de timestamp
    for i in range(1, len(df), 2):
        current_timestamp = df.iloc[i]['timestamp']
        if i + 1 == len(df):
            break
        next_timestamp = df.iloc[i + 1]['timestamp']
        diff = next_timestamp - current_timestamp  
        
        # Si la diferencia es mayor de 1800, añadir las tuplas necesarias
        if diff != 1800:
            for j in range(1, int(diff) // 1800):
                new_timestamp = current_timestamp + 1800 * j
                if(add_status_column):
                    new_row = {'timestamp': int(new_timestamp), 'value': 0, 'label': 0, 'status': 'ADDED'}
                else:
                    new_row = {'timestamp': int(new_timestamp), 'value': 0, 'label': 0}
                new_rows.append(new_row)
                new_rows.append(new_row)

    # Convertir la lista de nuevas tuplas a un DataFrame
    new_rows_df = pd.DataFrame(new_rows)

    # Añadir las nuevas tuplas al DataFrame original
    df = pd.concat([df, new_rows_df], ignore_index=True)

    # Ordenar nuevamente el DataFrame por 'timestamp'
    df = df.sort_values(by='timestamp')

    # Guardar el DataFrame resultante en un nuevo archivo CSV
    df.to_csv(output_path, index=False)

    print(f'Archivo guardado en: {output_path}')

# CalIt2_fill_missing_timestamps(True)