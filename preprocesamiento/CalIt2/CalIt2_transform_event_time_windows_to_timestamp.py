import pandas as pd
import os
from datetime import datetime
import pytz

def convert_to_timestamp(date_str, time_str, event_type):

    
    gmt_plus_1 = pytz.timezone('Etc/GMT-2')

    # Convertir las fechas a objetos datetime con la zona horaria GMT+01:00
    fecha1 = gmt_plus_1.localize(datetime.strptime(date_str + ' ' + time_str, '%m/%d/%y %H:%M:%S'))
    result = int(fecha1.timestamp())

    # dt = datetime.strptime(f"{date_str} {time_str}", "%m/%d/%y %H:%M:%S")
    # result = int(time.mktime(dt.timetuple()))
    if event_type == 'End':
        result += 1800 # Add 30 minutes to give time for the people to leave
    else:
        result -= 1800 # Subtract 30 minutes to give time for the people to arrive to the event

    return result

def CalIt2_transform_event_time_windows_to_timestamp():
    # Ruta del archivo a revisar LINUX
    # input_path = f'/home/javier/Practicas/Nuevos datasets/Callt2/preliminar/CalIt2.events.csv'
    # output_path = f'/home/javier/Practicas/Nuevos datasets/Callt2/preliminar/train_events.csv'
    
    base_path = os.path.abspath(os.path.dirname(__file__))

    input_path = os.path.join(base_path, '..', '..', 'Nuevos datasets', 'Callt2', 'preliminar', 'CalIt2.events.csv')
    output_path = os.path.join(base_path, '..', '..', 'Nuevos datasets', 'Callt2', 'preliminar', 'train_events.csv')


    # Leer el archivo CSV
    df = pd.read_csv(input_path)

    # Convertir las columnas Begin y End a timestamps
    df['Begin'] = df.apply(lambda row: convert_to_timestamp(row['Date'], row['Begin'], 'Begin'), axis=1)
    df['End'] = df.apply(lambda row: convert_to_timestamp(row['Date'], row['End'],'End'), axis=1)

    # Seleccionar las columnas requeridas
    df_output = df[['Begin', 'End', 'Name']]

    # Escribir el archivo CSV de salida
    df_output.to_csv(output_path, index=False)
    print(f'Archivo de eventos transformados a timestamps guardado en: {output_path}')    
        


# CalIt2_transform_event_time_windows_to_timestamp()
# Verificar antes si cada fila se repite 2 veces