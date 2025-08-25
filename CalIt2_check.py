import pandas as pd
import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), 'preprocesamiento/CalIt2'))

# Importar la función get_unique_kpi_ids desde el script iops_check_different_KPIs
from CalIt2_transform_date_to_timestamp import CalIt2_transform_date_to_timestamp
from CalIt2_check_every_row_is_repeated_2_times import CalIt2_check_every_row_is_repeated_2_times
from CalIt2_fill_missing_timestamps import CalIt2_fill_missing_timestamps
from CalIt2_transform_event_time_windows_to_timestamp import CalIt2_transform_event_time_windows_to_timestamp
from CalIt2_fill_label_field import CalIt2_fill_label_field



CalIt2_transform_date_to_timestamp()

CalIt2_check_every_row_is_repeated_2_times()

#En este caso no faltaban valores por eso dejo por defecto el argumento False
CalIt2_fill_missing_timestamps(False) #Si el valor es True, se añadirá una columna 'status' al DataFrame, si es false no se añadirá. El objetivo de añadir esta columna es para verificar cuantas columnas se han añadido.

CalIt2_transform_event_time_windows_to_timestamp()

#Primer argumento es para indicar a partir de que número de perosnas se debe analizar
    #Ejemplo: En el caso de utilizar 4, solamente entrará en consideración valores por encima de 4, el resto se considera no anómalo por lo que el label será 0.
#Segundo argumento es para añadir una columna 'rason' al DataFrame, si es false no se añadirá. El objetivo de añadir esta columna es para verificar El motivo por el que es considerado anomalía.
    #Posibles valores en la columna 'reason':
        # - 'Is an anomaly if many people are exiting near the start of an event'
        # - 'Is an anomaly if many people are entering near the end of an event'
        # - 'Is an anomaly if many people are exiting when there is no event'
        # - 'Is an anomaly if many people are entering when there is no event'
CalIt2_fill_label_field(4,False) 