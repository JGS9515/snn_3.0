# Análisis de Datasets - Proyecto SNN 3.0

## Resumen Ejecutivo

Este documento presenta el análisis detallado de los datasets utilizados en el proyecto SNN 3.0, verificando la información proporcionada en la documentación y añadiendo insights adicionales.

**Fecha de análisis:** Diciembre 2024
**Analista:** Javier González Santos

## 1. Verificación de Información Documentada

### 1.1 Dataset CalIt2

| Aspecto | Documentación | Verificación | Estado |
|---------|---------------|--------------|---------|
| Observaciones totales | 10.080 | 10.080 | ✅ **CORRECTO** |
| Frecuencia de muestreo | 30 minutos | ~35 minutos | ⚠️ **DIFERENCIA** |
| Porcentaje de anomalías | 24.80% | 24.80% | ✅ **CORRECTO** |
| Número de semanas | 15 | ~12.5 | ⚠️ **DIFERENCIA** |
| Particionado | 50/50 temporal | 50/50 temporal | ✅ **CORRECTO** |

**Hallazgos:**
- ✅ El número total de observaciones coincide exactamente
- ✅ El porcentaje de anomalías es correcto (24.80%)
- ✅ El particionado temporal está correctamente implementado
- ⚠️ La frecuencia de muestreo real es de ~35 minutos en lugar de 30 minutos
- ⚠️ El período temporal cubre ~12.5 semanas en lugar de 15

### 1.2 Dataset IOPS

| Aspecto | Documentación | Verificación | Estado |
|---------|---------------|--------------|---------|
| Observaciones totales | 2.788.680 | 4.821.526 | ⚠️ **DIFERENCIA** |
| KPIs únicos | 26 | 26 | ✅ **CORRECTO** |
| Frecuencia de muestreo | 1 minuto | 1 minuto | ✅ **CORRECTO** |
| Porcentaje de anomalías | 1.92% | 1.92% | ✅ **CORRECTO** |
| Particionado | 50/50 temporal | ~53/47 temporal | ⚠️ **DIFERENCIA** |

**Hallazgos:**
- ✅ Número de KPIs correcto (26 archivos)
- ✅ Frecuencia de muestreo correcta (1 minuto)
- ✅ Porcentaje de anomalías correcto (1.92%)
- ⚠️ Mayor número de observaciones que lo documentado
- ⚠️ Particionado ligeramente desigual (53/47 vs 50/50 esperado)

## 2. Estadísticas Detalladas

### 2.1 CalIt2 - Estadísticas de Valores

| Métrica | Valor |
|---------|-------|
| Media | 3.81 |
| Desviación estándar | 6.44 |
| Mínimo | 0.0 |
| Máximo | 62.0 |
| Observaciones entrenamiento | 5,040 |
| Observaciones prueba | 5,040 |

**Distribución temporal:**
- Primera mitad: 5,040 observaciones (entrenamiento)
- Segunda mitad: 5,040 observaciones (prueba)
- Ratio train/test: 1.00 (perfectamente balanceado)

### 2.2 IOPS - Estadísticas por KPI

| Métrica | Valor |
|---------|-------|
| Número total de KPIs | 26 |
| Observaciones promedio por KPI | 185,443 |
| KPI con menos observaciones | 16,495 |
| KPI con más observaciones | 295,379 |
| Observaciones entrenamiento | 2,476,315 |
| Observaciones prueba | 2,345,211 |
| Ratio train/test | 1.06 |

**Distribución de anomalías por KPI:**
- La tasa de anomalías varía significativamente entre KPIs
- Algunos KPIs tienen tasas muy bajas (< 0.1%)
- Otros KPIs pueden tener tasas más altas de anomalías

## 3. Insights Adicionales

### 3.1 Comparación entre Datasets

| Aspecto | CalIt2 | IOPS | Relación |
|---------|--------|------|----------|
| Tamaño relativo | 1x | 478x | IOPS es 478 veces más grande |
| Ratio de anomalías | 24.80% | 1.92% | CalIt2 tiene 22.4x más anomalías |
| Frecuencia muestreo | ~35 min | 1 min | IOPS es 35x más frecuente |
| Balance de clases | Moderado | Altamente desbalanceado | - |

### 3.2 Recomendaciones para el Preprocesamiento

#### CalIt2:
- ✅ Dataset moderadamente balanceado - menor necesidad de técnicas de balanceo
- ✅ Particionado 50/50 temporal correcto
- ⚠️ Considerar investigar la discrepancia en la frecuencia de muestreo

#### IOPS:
- ⚠️ Dataset altamente desbalanceado - considerar técnicas de oversampling
- ⚠️ Ratio train/test ligeramente desigual (1.06) - podría afectar la evaluación
- ✅ Alta variabilidad entre KPIs - considerar análisis por KPI individual

## 4. Preprocesado Común - Verificación

### 4.1 Información Documentada
- ✅ **Tipado de columnas:** value en float64, label en Int64
- ✅ **Expansión de etiquetas:** expansion = 100 para mitigar desbalanceo temporal
- ✅ **Cálculo de cuantiles:** rango extendido con a = 0.1 y resolución r = 0.05
- ✅ **Segmentación:** ventanas de longitud T = 250 con padding del conjunto de prueba

### 4.2 Verificación de Implementación
Los scripts de preprocesamiento existentes confirman que estas técnicas están correctamente implementadas en los directorios `preprocesamiento/CalIt2/` y `preprocesamiento/iops/`.

## 5. Visualizaciones Generadas

Se han creado las siguientes visualizaciones (guardadas en `dataset_analysis_visualization.png`):

1. **Serie temporal CalIt2** - Muestra la evolución temporal de los valores
2. **Anomalías en CalIt2** - Resalta los puntos identificados como anomalías
3. **Serie temporal IOPS** - Muestra una muestra representativa de los datos IOPS
4. **Top 10 KPIs por tasa de anomalías** - Identifica los KPIs más problemáticos

## 6. Conclusiones y Recomendaciones

### ✅ Información Correcta:
- Número total de observaciones CalIt2
- Porcentaje de anomalías en ambos datasets
- Número de KPIs en IOPS
- Frecuencia de muestreo IOPS
- Estrategia de particionado temporal
- Metodología de preprocesamiento común

### ⚠️ Discrepancias Encontradas:
1. **CalIt2:** Frecuencia de muestreo (~35 vs 30 min) y período temporal (~12.5 vs 15 semanas)
2. **IOPS:** Número total de observaciones (4.8M vs 2.8M esperado) y particionado ligeramente desigual

### 💡 Recomendaciones Adicionales:
1. **Actualizar documentación** con los valores reales verificados
2. **Investigar** las discrepancias en CalIt2 (posiblemente datos faltantes o diferentes criterios de filtrado)
3. **Considerar estratificación por KPI** en IOPS debido a la alta variabilidad
4. **Implementar validación cruzada temporal** para evaluar la robustez de los modelos
5. **Documentar la variabilidad entre KPIs** como factor importante para el rendimiento del modelo

## 7. Archivos Generados

- `dataset_analysis.py` - Script completo de análisis
- `dataset_analysis_visualization.png` - Visualizaciones generadas
- `dataset_analysis_summary.md` - Este documento resumen

---

**Nota:** Este análisis proporciona una base sólida para entender los datasets y validar la información documentada. Las discrepancias encontradas sugieren la necesidad de actualizar la documentación con los valores reales observados en los datos procesados.
