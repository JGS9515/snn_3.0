import torch, pandas as pd, numpy as np, os, argparse, json
from sklearn.model_selection import TimeSeriesSplit
from dependencias import *
import json
from sklearn.metrics import roc_curve, auc, precision_recall_fscore_support
# Load configuration from JSON file
with open('config.json', 'r') as f:
    config = json.load(f)

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Experimentación con STDP.')
parser.add_argument('-c', '--config', type=str, help='Configuration name', default='experimmentRequestedByTutor')
args = parser.parse_args()

# Get the configuration
cfg = config[args.config]

# Assign configuration values
nu1 = eval(cfg['nu1'])
nu2 = eval(cfg['nu2'])
snn_process_layer_neurons_size = cfg['snn_process_layer_neurons_size']
threshold = cfg['threshold']
decay = cfg['decay']
amp = cfg['ampliacion']
path = cfg['path']
epochs = eval(cfg['epochs'])
reso = cfg['resolucion']
recurrencia = eval(cfg['recurrencia'])

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


TIEMPO=100 #Valor razonablemente pequeño para no eliminar muchas muestras del entremiento.

#Ahora, es necesario leer los datos de entrada:
#Tienen que tener las columnas 'label' y 'value'.

data=pd.read_csv(path,na_values=['NA'])

#Asegurarse de que los tipos sean correctos:
data['value']=data['value'].astype('float64')
data['label']=data['label'].astype('Int64')

#Anulamos los valores ausentes del label para que funcione bien.
# Remplaza todos los valores ausentes en la columna label del DataFrame data con 0, 
# asegurando que no haya valores NaN en esa columna.
data.loc[data['label'].isna(),'label']=0

#Preparación de la validación cruzada:
# crea un objeto TimeSeriesSplit con 5 divisiones (se dividen los datos en 5 partes), 
# que se utilizará para realizar validación cruzada en datos de series temporales, 
# asegurando que el orden temporal se mantenga durante el proceso de entrenamiento y validación.
tscv=TimeSeriesSplit(n_splits=5)

split=0
# Iteramos sobre las divisiones de la validación cruzada: train_index y test_index son los índices 
# de los datos de entrenamiento y prueba, respectivamente, para cada división.
# Ejemplo: si hay 1000 datos y 5 divisiones.
# Split 1:
# Train indices: [  0  1  2 ... 165 166 167]
# Test indices:  [168 169 170 ... 331 332 333]

# Split 2:
# Train indices: [  0  1  2 ... 331 332 333]
# Test indices:  [334 335 336 ... 497 498 499]

# Split 3:
# Train indices: [  0  1  2 ... 497 498 499]
# Test indices:  [500 501 502 ... 663 664 665]

# Split 4:
# Train indices: [  0  1  2 ... 663 664 665]
# Test indices:  [666 667 668 ... 829 830 831]

# Split 5:
# Train indices: [  0  1  2 ... 829 830 831]
# Test indices:  [832 833 834 ... 995 996 997]
for train_index, test_index in tscv.split(data):
    split+=1
    print(f"Split {split}:")
    print(f"Train indices: {train_index}")
    print(f"Test indices: {test_index}")
    print()
    
    # Dividir divide el DataFrame data en dos subconjuntos: 
    # data_train para entrenamiento y data_test para prueba, utilizando los índices proporcionados 
    # por train_index y test_index, respectivamente. Esto es común en técnicas de validación cruzada, 
    # donde los datos se dividen en diferentes subconjuntos para evaluar el rendimiento del modelo.

    data_train, data_test = data.iloc[train_index], data.iloc[test_index]
    
    #Reseteamos el índice:
    data_train = data_train.reset_index(drop=True)
    data_test = data_test.reset_index(drop=True)
    
    data_train['value']=data_train['value'].astype('float64')
    data_test['label']=data_test['label'].astype('Int64')
    
    #Sacamos máximos y mínimos:
    minimo=min(data_train['value'][data_train['label']!=1])
    maximo=max(data_train['value'][data_train['label']!=1])
    
    #Anulamos valores donde el label sea 1, para el conjunto de entrenamiento:
    data_train.loc[data_train['label']==1,'value']=np.nan
    
    #Modelamos los rangos de las entradas:
    ancho_datos=maximo-minimo
    
    # Esta línea crea un tensor de tipo FloatTensor utilizando PyTorch y NumPy.
    # np.arange genera un array de valores que van desde (minimo - amp * ancho_datos) hasta (maximo + ancho_datos * amp) con un paso de (maximo - minimo) * reso.
    # Luego, este array se convierte en un tensor de PyTorch.
    cuantiles=torch.FloatTensor(
        np.arange(
            minimo-amp*ancho_datos,maximo+ancho_datos*amp,(maximo-minimo)*reso
            )
        )
    
    ################################################################################################################
    #Con cuantiles (pero no cambia nada respecto a hacerlo como en la línea anterior):
    #uno=torch.FloatTensor(np.arange(minimo-amp*ancho_datos,minimo,reso*(ancho_datos)))
    #dos=torch.FloatTensor(np.arange(maximo,maximo+amp*ancho_datos,reso*(ancho_datos)))
    #medio=torch.unique(torch.quantile(torch.FloatTensor(data_train['value'].dropna()),torch.FloatTensor(np.arange(0,1,reso))))
    #cuantiles=torch.cat((uno,medio,dos))
    
    ##############################################################################################################
    #Ahora, establecemos el valor de snn_input_layer_neurons_size, que será el número de neuronas de la capa de entrada:
    snn_input_layer_neurons_size=len(cuantiles)-1
    
    #Crea la red.
    network, source_monitor,target_monitor = crear_red(snn_input_layer_neurons_size,TIEMPO,snn_process_layer_neurons_size,threshold,decay,nu1,nu2,recurrencia,device)
    
    #Paddeamos el test:
    data_test=padd(data_test,TIEMPO)
    
    #Para cada secuencia del train, tenemos que pasarla y entrenar la red:
    network.learning=True

    #uno=torch.FloatTensor(np.arange(minimo-amp*ancho_datos,minimo,amp*(ancho_datos)))
    #dos=torch.FloatTensor(np.arange(maximo,maximo+amp*ancho_datos,amp*(ancho_datos)))
    #medio=torch.unique(torch.quantile(data_train,torch.FloatTensor(np.arange(0,1,reso)))
    #cuantiles=torch.cat((uno,medio,dos))
    #Procesemos secuencias de entrenamiento:
    secuencias2train=convertir_data(data_train,TIEMPO,cuantiles,snn_input_layer_neurons_size,device,is_train=True)
    print(f'Longitud de dataset de entrenamiento: {len(secuencias2train)}')
    secuencias2test=convertir_data(data_test,TIEMPO,cuantiles,snn_input_layer_neurons_size,device)
    print(f'Longitud de dataset de prueba: {len(secuencias2test)}')
            
    for e in range(max(epochs)):
        network.learning=True
        print(f'Epoch {e}')
        network=reset_voltajes(network,device)
        spikes_input,spikes,network=ejecutar_red(secuencias2train,network,source_monitor,target_monitor,TIEMPO)
        #Tras cada epoch, resetea voltajes:
        
        if e+1 in epochs:
            experiment_number = 1  # You can increment this for each experiment

            #Ahora, el test:
            network.learning=False
            network=reset_voltajes(network,device)
            spikes_input,spikes,network=ejecutar_red(secuencias2test,network,source_monitor,target_monitor,TIEMPO)
            
            #Ahora, guarda resultados convenientemente:
            
            base_output_path = f'resultados\\{args.config}'
            
            # Create base output directory if it doesn't exist
            os.makedirs(base_output_path, exist_ok=True)
            
            # Obtener la lista de carpetas en la ruta base
            existing_folders = [f for f in os.listdir(base_output_path) if os.path.isdir(os.path.join(base_output_path, f))]

            # Filtrar solo las carpetas que tienen nombres numéricos
            numeric_folders = [int(f) for f in existing_folders if f.isdigit()]

            # Encontrar el número más alto y sumar 1
            if numeric_folders:
                next_folder_number = max(numeric_folders) + 1
            else:
                next_folder_number = 1

            # Crear la nueva carpeta
            results_path = os.path.join(base_output_path, str(next_folder_number))

            os.makedirs(results_path, exist_ok=True)
            
            #Creemos el directorio de salida:
            nu1_str=str(nu1).replace('(','').replace(',','_').replace(')','')
            nu2_str=str(nu2).replace('(','').replace(',','_').replace(')','')

            # Save configuration values
            config = {
                'nu1': nu1_str,
                'nu2': nu2_str,
                'snn_process_layer_neurons_size': snn_process_layer_neurons_size,
                'threshold': threshold,
                'decay': decay,
                'amp': amp,
                'epoch': e + 1,
                'reso': reso,
                'recurrencia': recurrencia
             }
            with open(os.path.join(results_path, 'config.json'), 'w') as config_file:
                json.dump(config, config_file, indent=4)

            # Save results
            np.savetxt(os.path.join(results_path, 'spikes.csv'), spikes, delimiter=',')
            
            # Uncomment this if you want to turn NaN values into 0. If not, it throws an error when saving
            # NaN values
            data_test['label'] = data_test['label'].fillna(0).astype(float)
            np.savetxt(os.path.join(results_path, 'label.csv'), np.array(data_test['label']), delimiter=',')
            
            # Uncomment this if you want to save the label column with the NaN values
            # data_test.to_csv(os.path.join(results_path, 'label.csv'), columns=['label'], index=False)

            np.savetxt(os.path.join(results_path, 'value.csv'), np.array(data_test['value']), delimiter=',')

            experiment_number += 1  # Increment the experiment number for the next run
            
            # output_path=f'output/{path}/{nu1_str}/{nu2_str}/{snn_process_layer_neurons_size}/{threshold}/{decay}/{amp}/{e+1}/{reso}/{recurrencia}'
            # os.makedirs(output_path, exist_ok=True)
            # np.savetxt(f'{output_path}/spikes_{split}',spikes,delimiter=',')
            # np.savetxt(f'{output_path}/label_{split}',np.array(data_test['label']),delimiter=',')
            # np.savetxt(f'{output_path}/value_{split}',np.array(data_test['value']),delimiter=',')

            # Definir un umbral para la detección de anomalías
            umbral_anomalia = np.mean(spikes) + 2 * np.std(spikes)

            # Convertir spikes a predicciones binarias
            predicciones = (spikes.sum(axis=1) > umbral_anomalia).astype(int)

            # Calcular métricas
            precision, recall, f1, _ = precision_recall_fscore_support(data_test['label'], predicciones, average='binary')

            print(f"Precisión: {precision:.3f}")
            print(f"Recall: {recall:.3f}")
            print(f"F1-score: {f1:.3f}")

            # Calcular curva ROC y AUC
            fpr, tpr, _ = roc_curve(data_test['label'], spikes.sum(axis=1))
            roc_auc = auc(fpr, tpr)

            print(f"AUC: {roc_auc:.3f}")

            # Visualizar spikes (asegúrate de tener matplotlib instalado)
            from bindsnet.analysis.plotting import plot_spikes
            import torch

            spikes_tensor = torch.tensor(spikes, dtype=torch.float32)
            plot_spikes({"Capa B": spikes_tensor})
