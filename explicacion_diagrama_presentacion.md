# Explicación del Diagrama del Pipeline SNN para Presentación TFM

## Duración aproximada: 2 minutos

---

### Introducción (15 segundos)

Este diagrama muestra el flujo completo de nuestro sistema de detección de anomalías basado en redes neuronales de impulsos, desde los datos brutos hasta la evaluación final de resultados.

---

### 1. DATOS Y PREPROCESADO (30 segundos)

Comenzamos con los datos en formato CSV que contienen las series temporales. 

El preprocesado incluye la normalización de los valores y el manejo de datos faltantes para asegurar la calidad de la entrada.

La codificación transforma los valores numéricos continuos en impulsos discretos mediante cuantización. Se divide el rango de valores en intervalos (cuantiles), y cada intervalo corresponde a una neurona en la capa A. Por ejemplo, si creamos 150 intervalos, la capa A tendrá 150 neuronas, cada una especializada en detectar un rango específico de valores.

El agrupamiento temporal divide los datos en ventanas de 250 muestras, permitiendo que la red procese la información de manera secuencial y capture patrones temporales.

---

### 2. ARQUITECTURA SNN (40 segundos)

La Capa A de entrada recibe los datos codificados. Su tamaño es adaptativo (entre 20 y 150 neuronas) y depende del número de intervalos creados en la codificación: más intervalos, más neuronas, y más sensibilidad ante variaciones inusuales.

La Capa B de procesamiento es el núcleo de la red neuronal de impulsos. Utiliza neuronas LIF que procesan los impulsos y aprenden patrones mediante reglas de plasticidad sináptica. Esta capa puede detectar anomalías directamente.

La Capa C convolucional es de procesamiento que aplica filtros convolucionales para detectar patrones locales más complejos. 


---

### 3. EJECUCIÓN POR FASES (25 segundos)

En la fase de entrenamiento, la red aprende a distinguir entre comportamiento normal y anómalo. Los datos se procesan secuencia por secuencia, y los pesos sinápticos se ajustan automáticamente mediante aprendizaje no supervisado.

En la fase de prueba, evaluamos el rendimiento del modelo entrenado con datos que no ha visto antes, generando predicciones sobre qué puntos son anómalos.

---

### 4. RESULTADOS Y MÉTRICAS (30 segundos)

Los resultados se guardan incluyendo los impulsos generados por cada capa, las etiquetas reales y las predicciones del modelo.

Finalmente, calculamos las métricas de evaluación como precisión, recall y F1-score para cuantificar qué tan bien funciona nuestro modelo. Estas métricas nos permiten comparar el rendimiento de la capa B base con la arquitectura híbrida que incluye la capa C convolucional.

---

### Conclusión (10 segundos)

Este pipeline completo nos permite entrenar y evaluar diferentes configuraciones de la red, optimizando automáticamente los hiperparámetros mediante técnicas de optimización bayesiana para encontrar la mejor arquitectura para cada tipo de dataset.

---

## Notas para la presentación:

- Ritmo: Hablar a un ritmo moderado, aproximadamente 125-150 palabras por minuto
- Énfasis: Destacar la naturaleza opcional de la capa C y su impacto en datasets desbalanceados
- Transiciones: Usar pausas breves entre secciones para facilitar la comprensión
- Visualización: Señalar cada sección del diagrama mientras se explica

