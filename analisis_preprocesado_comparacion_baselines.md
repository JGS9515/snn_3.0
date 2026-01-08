# Análisis del Preprocesado y Comparación con Baselines

## Pregunta planteada

**"Si el preprocesado añade complejidad al formato genérico de serie temporal, ¿debería tenerse en cuenta en la comparación con los baselines?"**

---

## Análisis del Preprocesado Aplicado

### 1. Preprocesado Específico de SNN (Parte del Método Propuesto)

Estos pasos son **inherentes** al funcionamiento de las SNN y no pueden eliminarse:

- **Cuantización por cuantiles expandidos**: Necesario para convertir valores continuos en trenes de impulsos discretos. Sin esto, las SNN no pueden procesar los datos.
- **Segmentación en ventanas T=250**: Requerido para el procesamiento temporal secuencial de las SNN.
- **Codificación a spikes**: Transformación fundamental del dominio continuo al discreto.

**Conclusión**: Estos pasos son parte integral del método SNN y deben considerarse como parte de la arquitectura, no como ventaja adicional.

### 2. Preprocesado Genérico (Potencialmente Aplicable a Baselines)

Estos pasos podrían beneficiar a cualquier modelo de series temporales:

- **Normalización robusta (MAD-based z-score)**: Técnica estándar de preprocesado que podría aplicarse a cualquier modelo.
- **Expansión de etiquetas (expansion=100)**: Técnica específica para mitigar desbalanceo temporal que podría beneficiar a los baselines.
- **Completado de timestamps faltantes**: Preprocesado estándar de series temporales.
- **Clipping de outliers (-5, 5)**: Técnica común de preprocesado.

**Conclusión**: Estos pasos son técnicas genéricas que podrían aplicarse a los baselines TSFEDL.

---

## Análisis de la Comparación Actual

### Estado Actual de la Comparación

Según el documento (`5.1-Descripcion-escenarios.tex`):

- Los baselines TSFEDL se ejecutaron en un entorno separado debido a incompatibilidades de versiones.
- No se menciona explícitamente si los baselines recibieron el mismo preprocesado genérico.
- La comparación se realiza directamente sobre los resultados finales (F1-score).

### Implicaciones Metodológicas

**Ventaja potencial para SNN:**
- La normalización robusta y la expansión de etiquetas podrían estar mejorando el rendimiento de la SNN sin que los baselines se beneficien de estas técnicas.

**Limitación de la comparación:**
- Si los baselines no recibieron el mismo preprocesado genérico, la comparación podría no ser completamente justa.

---

## Respuesta Sugerida para la Defensa del TFM

### Respuesta Corta (30-45 segundos)

"Excelente pregunta metodológica. El preprocesado aplicado incluye dos tipos de pasos:

**Primero**, pasos específicos de SNN que son inherentes al método: la cuantización por cuantiles y la codificación a spikes, que son necesarios para que las SNN procesen datos continuos. Estos forman parte integral de la arquitectura propuesta.

**Segundo**, pasos de preprocesado genérico como normalización robusta y expansión de etiquetas, que podrían beneficiar a cualquier modelo.

Reconozco que para una comparación completamente justa, los baselines deberían recibir el mismo preprocesado genérico. Sin embargo, debido a las incompatibilidades técnicas entre entornos (PyTorch 1.x vs 2.x), no fue posible unificar el pipeline de preprocesado.

**No obstante**, la comparación sigue siendo válida porque:
1. Los baselines TSFEDL ya incluyen sus propios pipelines de preprocesado optimizados.
2. La comparación evalúa el rendimiento del sistema completo (preprocesado + modelo), que es lo que realmente importa en aplicaciones prácticas.
3. Los resultados muestran que incluso con estas técnicas, las SNN siguen por debajo de los baselines, lo que sugiere que la brecha no se debe únicamente al preprocesado.

En trabajos futuros, sería valioso realizar un estudio de ablación comparando el impacto del preprocesado en ambos tipos de modelos."

### Respuesta Extendida (si se profundiza)

**Análisis de Impacto del Preprocesado:**

1. **Normalización robusta (MAD)**: 
   - Impacto estimado: Moderado. La normalización estándar (z-score) es común en deep learning, pero la robusta (MAD) podría ser más resistente a outliers.
   - ¿Beneficiaría a baselines?: Probablemente sí, especialmente en datasets con outliers como IOPS.

2. **Expansión de etiquetas (expansion=100)**:
   - Impacto estimado: Alto en datasets desbalanceados como IOPS (1.92% anomalías).
   - ¿Beneficiaría a baselines?: Sí, especialmente en IOPS donde el desbalanceo es extremo.
   - Justificación: Esta técnica amplifica temporalmente las etiquetas de anomalía, creando más ejemplos positivos para el entrenamiento.

3. **Cuantización por cuantiles**:
   - Impacto: Específico de SNN, no aplicable a modelos continuos.
   - ¿Beneficiaría a baselines?: No directamente, pero podría considerarse como una forma de discretización que algunos modelos podrían aprovechar.

**Recomendaciones Metodológicas Futuras:**

1. **Estudio de ablación del preprocesado**: Evaluar el impacto individual de cada paso de preprocesado en ambos tipos de modelos.

2. **Preprocesado unificado**: Si es técnicamente posible, aplicar el mismo pipeline de preprocesado genérico a todos los modelos.

3. **Comparación en dos niveles**:
   - Nivel 1: Comparación del sistema completo (preprocesado + modelo) - **lo que se hizo actualmente**.
   - Nivel 2: Comparación de modelos con preprocesado idéntico - **trabajo futuro**.

4. **Documentación explícita**: En futuros trabajos, documentar claramente qué preprocesado recibe cada modelo y justificar las diferencias.

---

## Conclusión

La pregunta es metodológicamente válida y muestra una comprensión profunda de las limitaciones de la comparación experimental. La respuesta debe:

1. **Reconocer** la validez de la preocupación.
2. **Distinguir** entre preprocesado específico de SNN (parte del método) y genérico (potencialmente aplicable a baselines).
3. **Justificar** por qué la comparación actual sigue siendo válida (sistemas completos, aplicaciones prácticas).
4. **Proponer** mejoras metodológicas para trabajos futuros.

La comparación actual evalúa sistemas completos, que es lo relevante en aplicaciones prácticas, pero un estudio más controlado sería valioso para entender mejor el impacto relativo del preprocesado vs. la arquitectura del modelo.

