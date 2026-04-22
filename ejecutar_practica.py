"""
Script para ejecutar el pipeline completo sin Jupyter.
Los resultados se guardarán en archivos de salida.
"""

import sys
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("PRÁCTICA 1: Pipeline ML para Detección de Impago")
print("Autor: Silvia Martínez Moreno")
print("="*80)

# Imports
print("\n[1/9] Importando librerías...")
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # No GUI
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append('src')
from preprocessing.practica1_preprocessing import Practica1Preprocess
from filtering.practica1_filtering import Practica1Filtering

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    precision_recall_curve, auc, confusion_matrix
)

RANDOM_STATE = 42
print("[OK] Librerias importadas")

# Carga de datos
print("\n[2/9] Cargando datos...")
df_train = pd.read_csv('data/df_train_small.csv')
df_test = pd.read_csv('data/df_test_small.csv')

y_train = (df_train['loan_status'] != 'Fully Paid').astype(int)
y_test = (df_test['loan_status'] != 'Fully Paid').astype(int)

X_train = df_train.drop(columns=['loan_status'])
X_test = df_test.drop(columns=['loan_status'])

print(f"[OK] Train: {X_train.shape}, Test: {X_test.shape}")
print(f"     Default rate: {100*y_train.mean():.1f}%")

# Preprocesamiento
print("\n[3/9] Preprocesamiento...")
preprocessor = Practica1Preprocess(variables_path='data/variables_withExperts.xlsx')
X_train_prep = preprocessor.fit_transform(X_train, y_train)
X_test_prep = preprocessor.transform(X_test)
print(f"[OK] Train: {X_train_prep.shape}, Test: {X_test_prep.shape}")

# Filtrado
print("\n[4/9] Filtrado de features...")
filtering = Practica1Filtering(
    variance_threshold=0.01,
    mutual_info_percentile=70,
    rf_threshold='median',
    random_state=RANDOM_STATE,
    verbose=True
)
X_train_filt = filtering.fit_transform(X_train_prep, y_train)
X_test_filt = filtering.transform(X_test_prep)
print(f"[OK] Train: {X_train_filt.shape}, Test: {X_test_filt.shape}")

# Entrenar modelos
print("\n[5/9] Entrenando Gradient Boosting...")
model_gb = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    min_samples_split=100,
    min_samples_leaf=50,
    random_state=RANDOM_STATE,
    verbose=0
)
model_gb.fit(X_train_filt, y_train)
print("[OK] Gradient Boosting entrenado")

print("\n[6/9] Entrenando SVM...")
print("   Nota: Usando muestra de 10,000 registros para acelerar (SVM es lento con datos grandes)")
# Reducir dataset para SVM con estratificacion para mantener balance de clases
sample_size = min(10000, len(X_train_filt))
if sample_size < len(X_train_filt):
    X_train_svm, _, y_train_svm, _ = train_test_split(
        X_train_filt, y_train,
        train_size=sample_size,
        stratify=y_train,
        random_state=RANDOM_STATE
    )
else:
    X_train_svm = X_train_filt
    y_train_svm = y_train

model_svm = SVC(
    kernel='rbf',
    C=10.0,  # Mayor C para datos complejos
    gamma='scale',
    class_weight='balanced',
    probability=True,
    max_iter=1000,
    random_state=RANDOM_STATE,
    verbose=False
)
model_svm.fit(X_train_svm, y_train_svm)
print("[OK] SVM entrenado")

print("\n[7/9] Entrenando Red Neuronal...")
model_mlp = MLPClassifier(
    hidden_layer_sizes=(100, 50),
    activation='relu',
    solver='adam',
    alpha=0.001,  # Regularizacion
    learning_rate='adaptive',
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
    random_state=RANDOM_STATE,
    verbose=False
)
model_mlp.fit(X_train_filt, y_train)
print(f"[OK] MLP entrenado (convergio en {model_mlp.n_iter_} iteraciones)")

# Evaluar
print("\n[8/9] Evaluando modelos...")

def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, pos_label=1)
    recall = recall_score(y_test, y_pred, pos_label=1)

    precision_vals, recall_vals, _ = precision_recall_curve(y_test, y_prob)
    pr_auc = auc(recall_vals, precision_vals)

    cm = confusion_matrix(y_test, y_pred)

    print(f"\n{'='*60}")
    print(f"[RESULTADOS] {model_name}")
    print(f"{'='*60}")
    print(f"Accuracy:           {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Precision (impago): {precision:.4f} ({precision*100:.2f}%)")
    print(f"Recall (impago):    {recall:.4f} ({recall*100:.2f}%)")
    print(f"PR-AUC:             {pr_auc:.4f}")
    print(f"\nMatriz de Confusión:")
    print(f"  TN={cm[0,0]:<6} FP={cm[0,1]:<6}")
    print(f"  FN={cm[1,0]:<6} TP={cm[1,1]:<6}")

    return {
        'Modelo': model_name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'PR-AUC': pr_auc
    }

results = []
results.append(evaluate_model(model_gb, X_test_filt, y_test, 'Gradient Boosting'))
results.append(evaluate_model(model_svm, X_test_filt, y_test, 'SVM (RBF)'))
results.append(evaluate_model(model_mlp, X_test_filt, y_test, 'Red Neuronal (MLP)'))

# Comparación con modelo base
modelo_base = {
    'Modelo': 'Modelo Base (FICO)',
    'Accuracy': 0.72,
    'Precision': 0.26,
    'Recall': 0.24,
    'PR-AUC': np.nan
}
results.insert(0, modelo_base)

df_results = pd.DataFrame(results)

print("\n" + "="*80)
print("[COMPARACION] TABLA COMPARATIVA DE RESULTADOS")
print("="*80)
print(df_results.to_string(index=False))
print("="*80)

# Guardar resultados
print("\n[9/9] Guardando resultados...")
df_results.to_csv('resultados_modelos.csv', index=False)
print("[OK] Resultados guardados en: resultados_modelos.csv")

# Análisis de mejoras
base_metrics = df_results[df_results['Modelo'] == 'Modelo Base (FICO)'].iloc[0]

print("\n" + "="*80)
print("[ANALISIS] MEJORAS RESPECTO AL MODELO BASE")
print("="*80)

for _, row in df_results[df_results['Modelo'] != 'Modelo Base (FICO)'].iterrows():
    print(f"\n{row['Modelo']}:")
    for metric in ['Accuracy', 'Precision', 'Recall']:
        base_val = base_metrics[metric]
        new_val = row[metric]
        diff_abs = new_val - base_val
        diff_rel = (diff_abs / base_val) * 100 if base_val > 0 else 0
        sign = '+' if diff_abs >= 0 else ''
        symbol = '[+]' if diff_abs >= 0 else '[-]'
        print(f"  {symbol} {metric}: {sign}{diff_abs:.3f} ({sign}{diff_rel:.1f}%)")

print("\n" + "="*80)
print("[COMPLETADO] PIPELINE EJECUTADO EXITOSAMENTE")
print("="*80)
print("\nPróximos pasos:")
print("1. Revisa los resultados en: resultados_modelos.csv")
print("2. Ejecuta el notebook en Jupyter para ver gráficos")
print("3. Sube todo a GitHub")
