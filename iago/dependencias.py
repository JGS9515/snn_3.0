import torch, pandas as pd, numpy as np
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes,AdaptiveLIFNodes
from bindsnet.network.topology import Connection
from bindsnet.network.monitors import Monitor
from bindsnet.analysis.plotting import plot_spikes, plot_voltages
from bindsnet.learning import PostPre
from datetime import datetime


#Archivo que contiene las funciones que se emplearán en el procesamiento con las SNNs.

def reset_voltajes(network,device):
    #Función que realiza el reseteo de los voltajes de las neuronas.
    network.layers['B'].v=torch.full(network.layers['B'].v.shape,-65).to(device)
    return network


def dividir(data,minimo):
    #Función que divide los datos de entrenamiento, para considerar aisladamente cada subsecuencia de datos normales.
    #En principio, no es la estrategia a usar con las SNNs, pero sí será necesaria con las baselines.
    intervals = []
    in_sequence = False
    
    #Iteramos para identificar los intervalos:
    for i in range(len(data)):
        if data.loc[i, 'label'] == 0:
            if not in_sequence:
                start_idx = i
                in_sequence = True
            end_idx = i+1
        else:
            if in_sequence:
                intervals.append((start_idx, end_idx))
                in_sequence = False
    
    # Agrega la posición del último elemento de los datos de entrada:
    if in_sequence:
        intervals.append((start_idx, end_idx))
    
    #Creamos un dataframe con los intervalos encontrados:
    intervals_df = pd.DataFrame(intervals, columns=['inicio', 'final'])
    
    subs=[]
    #Iteramos para dividir:
    for i,row in intervals_df.iterrows():
        inicio_tmp=row['inicio']
        final_tmp=row['final']
        if final_tmp-inicio_tmp>=minimo:
            subs.append(data.iloc[inicio_tmp:final_tmp].reset_index(drop=True))
    
    return subs


def padd(data, T):
    #Función que añade ceros al finalizar el conjunto de prueba, hasta llegar a un múltiplo del tiempo 
    #de exposición.
    
    lon = len(data)
    # Calcular el múltiplo más cercano de T superior al número actual de filas
    lon2 = ((lon // T) + 1) * T
    # Calcular el número de filas adicionales necesarias
    lon_adicional = lon2 - lon
    
    # Crear un DataFrame con filas adicionales llenas de NaN
    if lon_adicional > 0:
        datanul = pd.DataFrame(np.nan, index=range(lon_adicional), columns=data.columns)
        # Concatenar el DataFrame original con el DataFrame de padding
        data = pd.concat([data, datanul], ignore_index=True)
    
    return data


def expandir(serie, n):
    #Función que expande la serie (aunque es mejor que esta parte se haga a mano).
    # Crea gemelo de la serie:
    serie2 = np.zeros_like(serie)
    
    # Identificar los índices donde hay un 1:
    indices = np.where(serie == 1)[0]
    
    # Poner a 1 los valores en el rango [índice-n, índice+n]
    for idx in indices:
        start = max(0, idx - n)
        end = min(len(serie), idx + n + 1)
        serie2[start:end] = 1
    
    return pd.Series(serie2, index=serie.index)


#Función para convertir a spikes las entradas:
def podar(x,q1,q2,cuantiles=None):
    #Función que devuelve 1 (spike) si x está en el rango [q1,q2), y 0 en caso contrario.
    #Es parte de la codificación de los datos.
    
    s=torch.zeros_like(x)
    
    s[(x>=q1) & (x<q2)]=1
    return s


def convertir_data(data,T,cuantiles,R,device,is_train=False):
    #Función que lee los datos y los prepara.
    
    #Esta parte debe ser modificada para obtener la serie temporal de interés
    #almacenada en la variable serie.
    #
    serie=torch.FloatTensor(data['value'])
    
    #Tomamos la longitud de la serie.
    long=serie.shape[0]
    
    #Los valores inferiores al mínimo del vector de cuantiles se sustituyen por ese mínimo.
    #Los valores mayores que el máximo de los cuantiles se sustituyen por ese máximo.
    #Esto sirve para no perder spikes cuando se procesan los datos de prueba.
    
    serie[serie<torch.min(cuantiles)]=torch.min(cuantiles)
    serie[serie>torch.max(cuantiles)]=torch.max(cuantiles)
    
    #Construimos el tensor con los datos codificados. Básicamente, para cada dato de entrada tendremos una secuencia donde 
    #todos los valores son 0 menos 1, correspondiente al cuantil correspondiente al dato de entrada.
    serie2input=torch.cat([serie.unsqueeze(0)] * R, dim=0)
    
    for i in range(R):
        serie2input[i,:]=podar(serie2input[i,:],cuantiles[i],cuantiles[i+1])
    
    #Lo dividimos en función del tiempo de exposición T:
    secuencias = torch.split(serie2input.to(device),T,dim=1)
    
    if is_train:
        #Encaso de estar entrenando, tendríamos que quitar la última secuencia:
        secuencias=secuencias[0:len(secuencias)-1]
    
    return secuencias


def crear_red(snn_input_layer,T,snn_process_layer_neurons_size,threshold,decay,nu1,nu2,recurrencia,device):
    #Aquí creamos la red.
    
    network = Network()
    
    #Creamos las capas de entrada e interna:
    source_layer = Input(n=snn_input_layer,traces=True)
    target_layer = LIFNodes(n=snn_process_layer_neurons_size,traces=True,thresh=threshold, tc_decay=decay)
    
    network.add_layer(
        layer=source_layer, name="A"
    )
    network.add_layer(
        layer=target_layer, name="B"
    )
    
    #Creamos conexiones entre las capas de entrada y la recurrente:
    forward_connection = Connection(
        source=source_layer,
        target=target_layer,
        w=0.05 + 0.1 * torch.randn(source_layer.n, target_layer.n),
        #w=100 + 100 * torch.randn(source_layer.n, target_layer.n) #Poner parámetros más altos aquí, como estos valores de 100, ayuda a que se generen más spikes.
        update_rule=PostPre, nu=nu1
    )
    
    network.add_connection(
        connection=forward_connection, source="A", target="B"
    )
    if recurrencia: #Aquí decidimos si metemos o no la capa recurrente.
        # Creamos la conexión recurrente con pesos ligeramente negativos (si no, estamos metiendo posible ruido en el procesamiento de la red):
        recurrent_connection = Connection(
            source=target_layer,
            target=target_layer,
            w=0.025 * (torch.eye(target_layer.n) - 1),
            update_rule=PostPre, nu=nu2#nu=(1e-4, 1e-2)
        )
        
        network.add_connection(
            connection=recurrent_connection, source="B", target="B"
        )

    network=network.to(device)
    
    #Creamos los monitores. Sirven para registrar los spikes y voltajes:
    #Spikes de entrada (para depurar que se esté haciendo bien, si se quiere):
    source_monitor = Monitor(
        obj=source_layer,
        state_vars=("s",),  #Registramos sólo los spikes.
        time=T,
        # device=device, #No es necesario, ya que se registra en la CPU.
    )
    #Spikes de la capa recurrente (lo que nos interesa):
    target_monitor = Monitor(
        obj=target_layer,
        state_vars=("s", "v"),  #Registramos spikes y voltajes, por si nos interesa lo segundo también.
        time=T,
        # device=device, #No es necesario, ya que se registra en la CPU.
    )
    
    network.add_monitor(monitor=source_monitor, name="X")
    network.add_monitor(monitor=target_monitor, name="Y")
    
    
    return [network,source_monitor,target_monitor]


def ejecutar_red(secuencias, network, source_monitor, target_monitor, T):
    # Función para ejecutar la red con los datos que se quieran, ya sea para entrenamiento o evaluación.
    
    # Creamos los objetos lista en que almacenaremos los resultados:
    sp0 = []
    sp1 = []
    
    # Entrenamos:
    j = 1
    for i in secuencias:
        # Los datos de entrada serán una tupla con tensores de pytorch, pasamos cada una:
        # print(f'Ejecutando secuencia {j}')
        j += 1
        inputs = {'A': i.T}  # .to(device)
        
        # Debugging: Print the shape of the input tensor
        print(f"Shape of input tensor for sequence {j}: {i.shape}")
        
        inicio = datetime.now()
        
        # Debugging: Print the state of the network before running
        print(f"Running network with input shape: {inputs['A'].shape} and time: {T}")
            # Check if the index is within bounds
        if T > inputs['A'].shape[0]:
            print(f"Error: Time {T} is out of bounds for input shape {inputs['A'].shape}")
            continue
    
        try:
            network.run(inputs=inputs, time=T)
        except IndexError as e:
            print(f"IndexError: {e}")
            print(f"Input tensor shape: {inputs['A'].shape}")
            raise
        
        final = datetime.now()
        # print(final-inicio)
        
        # Obtenemos los spikes a lo largo de la simulación:
        spikes = {
            "X": source_monitor.get("s"), "B": target_monitor.get("s")
        }
        sp0.append(spikes['X'].sum(axis=2))
        sp1.append(spikes['B'].sum(axis=2))
        voltages = {"Y": target_monitor.get("v")}
        
    
    #Concatenamos y devolvemos:
    sp0 = torch.cat(sp0)
    sp0=sp0.detach().cpu().numpy()
    
    sp1=torch.cat(sp1)
    sp1=sp1.detach().cpu().numpy()
    
    return [sp0,sp1,network]

