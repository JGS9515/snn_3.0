import os

def verify_that_all_KPI_files_exist(unique_kpi_ids):
    folder_path = '/home/javier/Practicas/Nuevos datasets/iops/preliminar/train_procesado'

    # Verificar la existencia de archivos correspondientes a cada KPI ID
    for kpi_id in unique_kpi_ids:
        file_path = os.path.join(folder_path, f'{kpi_id}')
        if not os.path.exists(file_path):
            print(f'Archivo NO encontrado para KPI ID {kpi_id}')