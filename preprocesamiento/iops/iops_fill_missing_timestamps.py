import pandas as pd

def iops_fill_missing_timestamps(kpi,add_status_column: bool):

    # Ruta del archivo a revisar
    input_path = f'/home/javier/Practicas/Nuevos datasets/iops/preliminar/train_procesado_javi/{kpi}.csv'
    output_path = f'/home/javier/Practicas/Nuevos datasets/iops/preliminar/train_procesado_javi/{kpi}_filled.csv'

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
    for i in range(1, len(df)):
        current_timestamp = df.iloc[i]['timestamp']
        previous_timestamp = df.iloc[i - 1]['timestamp']
        diff = current_timestamp - previous_timestamp
        
        # Si la diferencia es mayor de 60, añadir las tuplas necesarias
        if diff > 60:
            for j in range(1, int(diff) // 60):
                new_timestamp = previous_timestamp + 60 * j
                if(add_status_column):
                    new_row = {'timestamp': int(new_timestamp), 'value': 0, 'label': 0, 'status': 'ADDED'}
                else:
                    new_row = {'timestamp': int(new_timestamp), 'value': 0, 'label': 0}
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