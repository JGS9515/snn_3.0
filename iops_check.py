import pandas as pd
import sys
import os


sys.path.append(os.path.join(os.path.dirname(__file__), 'preprocesamiento/iops'))

# Importar la función get_unique_kpi_ids desde el script iops_check_different_KPIs
from iops_check_different_KPIs import iops_check_different_KPIs
from verify_that_all_KPI_files_exist import verify_that_all_KPI_files_exist
from create_new_dataset_per_every_different_kpi import create_new_dataset_per_every_different_kpi
from iops_fill_missing_timestamps import iops_fill_missing_timestamps


result = iops_check_different_KPIs()
unique_kpi_ids = result['unique_kpi_ids']
df = result['df']
verify_that_all_KPI_files_exist(unique_kpi_ids)

create_new_dataset_per_every_different_kpi(df,unique_kpi_ids)
#Si el valor es True, se añadirá una columna 'status' al DataFrame, si es false no se añadirá. El objetivo de añadir esta columna es para verificar cuantas columnas se han añadido.
for kpi in unique_kpi_ids:
    iops_fill_missing_timestamps(kpi, False)
