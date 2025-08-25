
def create_new_dataset_per_every_different_kpi(df, unique_kpi_ids):

    for kpi in unique_kpi_ids:
        output_path = f'/home/javier/Practicas/Nuevos datasets/iops/preliminar/train_procesado_javi/{kpi}.csv'
        filtered_df = df[df['KPI ID'] == kpi]

        # Seleccionar solamente las primeras 3 columnas
        filtered_df_first_3_columns = filtered_df.iloc[:, :3]

        # Guardar las filas filtradas en un nuevo archivo CSV con solo las primeras 3 columnas
        filtered_df_first_3_columns.to_csv(output_path, index=False)
