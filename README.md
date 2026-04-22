# Práctica 1: Pipeline de Machine Learning para Detección de Impago

**Asignatura:** Modelización en Ingeniería de Datos  
**Universidad:** CUNEF  
**Autor:** Silvia Martínez Moreno  
**Fecha:** Abril 2026

## 📋 Descripción

Pipeline completo de Machine Learning para predecir impago de préstamos bancarios (dataset LendingClub). Incluye preprocesamiento avanzado con técnicas alternativas, filtrado de features y comparación de 3 familias de modelos diferentes contra un modelo base de referencia.

## 🎯 Objetivos

Desarrollar un pipeline alternativo completo que:
- Utilice **variables de expertos** (grade, sub_grade, FICO score, tasas de interés)
- Implemente **técnicas alternativas** de preprocesamiento y filtrado diferentes a la clase base
- Entrene y compare **3 familias de modelos** diferentes
- **Supere** las métricas del modelo base de referencia (FICO score)

## 📁 Estructura del Proyecto

```
practica_1/
├── data/
│   ├── df_train_small.csv              # Datos de entrenamiento
│   ├── df_test_small.csv               # Datos de test
│   ├── variables_withExperts.xlsx      # Configuración de variables (CON expertos)
│   └── variables_withoutExperts.xlsx   # Configuración de variables (SIN expertos)
├── src/
│   ├── preprocessing/
│   │   ├── __init__.py
│   │   ├── base_preprocessing.py       # Clase base (referencia)
│   │   └── practica1_preprocessing.py  # ⭐ Clase de preprocesamiento (NUEVA)
│   └── filtering/
│       ├── __init__.py
│       ├── base_filtering.py           # Clase base (referencia)
│       └── practica1_filtering.py      # ⭐ Clase de filtrado (NUEVA)
├── practica1_notebook.ipynb            # ⭐ Notebook principal (NUEVO)
├── requirements.txt                    # Dependencias
└── README.md                           # Este archivo
```

## 🚀 Instalación

### Requisitos previos
- Python 3.8+
- pip

### Dependencias

```bash
pip install -r requirements.txt
```

Las principales librerías utilizadas:
- `scikit-learn` (1.3+): modelos, preprocesamiento, métricas
- `pandas` (1.5+): manipulación de datos
- `numpy` (1.23+): operaciones numéricas
- `sentence-transformers` (2.2+): encoding de texto
- `openpyxl` (3.1+): lectura de Excel
- `matplotlib` (3.5+): visualización
- `seaborn` (0.12+): visualización estadística

## 📊 Pipeline Implementado

### 1. Preprocesamiento (`Practica1Preprocess`)

#### Diferencias respecto a la clase base:

| Aspecto | Clase Base | **Practica1Preprocess** (NUEVA) |
|---------|-----------|--------------------------------|
| **Variables** | `variables_withoutExperts.xlsx` | **`variables_withExperts.xlsx`** (incluye grade, FICO, int_rate) |
| **Imputación nulos** | Mediana/moda simple | **KNNImputer** (considera vecinos cercanos) |
| **Encoding categórico** | OneHotEncoder | **TargetEncoder + OrdinalEncoder** (grade ordenado A-G) |
| **Scaling numérico** | QuantileTransformer | **RobustScaler** (robusto a outliers, usa mediana/IQR) |
| **Nuevas features** | PolynomialFeatures (grado 2) | **Ratios financieros** (debt-to-income, payment-to-income, fico_mean, etc.) |

#### Features financieras creadas:
- `debt_to_income_ratio`: deuda total / ingreso anual
- `revolving_utilization_norm`: balance revolving / límite de crédito (normalizado)
- `payment_to_income_ratio`: cuota mensual * 12 / ingreso anual
- `fico_mean`: promedio de fico_range_low y fico_range_high
- `installment_burden`: cuota / (ingreso anual / 12)
- `loan_to_income`: monto del préstamo / ingreso anual

### 2. Filtrado de Features (`Practica1Filtering`)

#### Métodos implementados (en secuencia):

1. **VarianceThreshold** (umbral=0.01)
   - Elimina features con varianza muy baja (casi constantes)
   - Reduce ruido y dimensionalidad

2. **Mutual Information** (top 70%)
   - Selecciona features con mayor información mutua con el target
   - Captura relaciones no lineales

3. **SelectFromModel con Random Forest** (umbral=median)
   - Selecciona features por importancia según RF
   - Combina importancia de múltiples árboles

#### Ventajas del filtrado en secuencia:
- Reduce progresivamente dimensionalidad sin perder información
- Complementa diferentes criterios de selección
- Evita overfitting y mejora generalización

### 3. Modelos Entrenados

#### Familia 1: Ensemble - **Gradient Boosting**
```python
GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    random_state=42
)
```
**Por qué:** Potente, eficiente, maneja bien interacciones complejas

#### Familia 2: SVM - **Support Vector Machine**
```python
SVC(
    kernel='rbf',
    C=1.0,
    gamma='scale',
    class_weight='balanced',
    probability=True,
    random_state=42
)
```
**Por qué:** Efectivo en espacios de alta dimensión, kernel RBF captura no linealidades

#### Familia 3: Redes Neuronales - **MLP**
```python
MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    learning_rate='adaptive',
    early_stopping=True,
    random_state=42
)
```
**Por qué:** Flexible, aprende representaciones complejas, early stopping previene overfitting

## 📈 Evaluación

### Métricas utilizadas:

1. **Accuracy**: Proporción de predicciones correctas
   - ⚠️ Engañosa en datasets desbalanceados (80/20)

2. **Precision (impago)**: De los predichos como impago, cuántos realmente lo son
   - 🎯 Crítica para evitar rechazar préstamos buenos (falsos positivos)

3. **Recall (impago)**: De los impagos reales, cuántos detectamos
   - 🎯 Crítica para evitar aprobar préstamos malos (falsos negativos)

4. **PR-AUC**: Área bajo la curva Precision-Recall
   - 📊 **Más informativa que ROC-AUC** en datasets desbalanceados
   - Resume el balance precision-recall en un solo número

### Modelo Base de Referencia

El modelo base utiliza únicamente el **FICO score normalizado** con umbral 0.67:
- Accuracy: ~72%
- Precision (impago): ~26%
- Recall (impago): ~24%

Este modelo simple establece la línea base que debemos superar.

## 🏃‍♂️ Cómo Ejecutar

### 1. Clonar el repositorio
```bash
git clone https://github.com/silviamartinezcunef/practica1.git
cd practica1
```

### 2. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 3. Verificar datos
Los datos ya están incluidos en `data/`:
- ✅ `df_train_small.csv` (80,000 filas)
- ✅ `df_test_small.csv` (20,000 filas)
- ✅ `variables_withExperts.xlsx`

### 4. Abrir el notebook
```bash
jupyter notebook practica1_notebook.ipynb
```

El notebook **ya está ejecutado** con las salidas visibles. Puedes:
- Revisar los resultados directamente en el archivo
- Re-ejecutar todo: `Kernel` → `Restart & Run All` (~20-30 min)

### 5. Estructura del notebook
1. **Imports y configuración**
2. **Carga de datos** - 80k train, 20k test
3. **Preprocesamiento** - KNNImputer, TargetEncoder, RobustScaler
4. **Filtrado** - VarianceThreshold, Mutual Information, SelectFromModel
5. **Entrenamiento** - Gradient Boosting, SVM, MLP
6. **Evaluación** - Accuracy, Precision, Recall, PR-AUC
7. **Comparación con modelo base** - Análisis de mejoras
8. **Análisis de errores** - FP vs FN
9. **Conclusiones**

## 📝 Decisiones de Diseño

### ¿Por qué KNNImputer?
- Considera similitud entre muestras
- Más sofisticado que mediana/moda simple
- Preserva mejor la estructura de los datos

### ¿Por qué TargetEncoder?
- Reduce dimensionalidad vs OneHotEncoder
- Captura relación con el target
- Cross-validation evita overfitting

### ¿Por qué RobustScaler?
- Robusto a outliers (usa mediana e IQR)
- Datos financieros suelen tener outliers
- Más estable que StandardScaler

### ¿Por qué estos modelos?
- **Gradient Boosting**: Estado del arte en tabular data
- **SVM**: Teoría sólida, efectivo en alta dimensión
- **MLP**: Flexible, aprende patrones complejos
- Representan 3 enfoques fundamentalmente diferentes

## 🔍 Interpretación de Resultados

### En contexto bancario:

**Falsos Positivos (FP):**
- Préstamos buenos rechazados
- Costo: oportunidad de negocio perdida (intereses)
- Menor gravedad

**Falsos Negativos (FN):**
- Préstamos malos aprobados
- Costo: pérdida del principal del préstamo
- **Mayor gravedad** → preferir modelos con alto recall

### Trade-off precision-recall:
- Alta precision → pocos FP → rechazamos más (conservador)
- Alto recall → pocos FN → aprobamos más (agresivo)
- Umbral de decisión ajustable según aversión al riesgo

## 🎓 Criterios de Evaluación (Rúbrica)

- **Preprocesamiento** (3 pts): fit/transform, variables expertos, técnicas alternativas
- **Filtrado** (1.5 pts): fit/transform, métodos alternativos
- **Modelos y evaluación** (4.5 pts): 3 modelos, métricas completas, comparación
- **Calidad del notebook** (1 pt): comentarios, justificaciones, repositorio Git

**Total:** 10 puntos

## 📚 Referencias

- [scikit-learn Documentation](https://scikit-learn.org/)
- [feature-engine Documentation](https://feature-engine.readthedocs.io/)
- Material de clase: notebooks 03 y 04
- Clases base: `BasePreprocess`, `BaseFiltering`

## 🐛 Troubleshooting

### Errores comunes:

**1. ModuleNotFoundError: No module named 'sentence_transformers'**
```bash
pip install sentence-transformers
```

**2. FileNotFoundError: variables_withExperts.xlsx**
- Verifica que el archivo esté en `data/`
- Verifica el path relativo desde el notebook

**3. SVM tarda demasiado**
- Reduce el dataset de train con `sample()`
- Reduce `max_iter` o usa kernel 'linear'

**4. MLP no converge**
- Aumenta `max_iter` (ej: 500)
- Ajusta `learning_rate_init` (ej: 0.001)
- Verifica que las features estén normalizadas

## 📧 Contacto

Para dudas o comentarios:
- Email: [silvia.martinez@cunef.edu]
- GitHub Issues: [https://github.com/silviamartinezcunef/practica1]

## 📄 Licencia

Este proyecto es parte de una práctica académica de CUNEF.

---

**Última actualización:** Abril 2026
