"""
Clase de preprocesamiento para la Práctica 1 - Pipeline ML Detección de Impago
Incluye variables de expertos y técnicas alternativas de preprocesamiento.

Autores: Silvia Martínez Moreno
Fecha: Abril 2026
"""

import pandas as pd
import numpy as np
from typing import Optional
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import (
    TargetEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler
)
from sentence_transformers import SentenceTransformer


class Practica1Preprocess:
    """
    Clase de preprocesamiento alternativa que utiliza variables de expertos
    y técnicas diferentes a la clase base.

    Diferencias clave respecto a BasePreprocess:
    - Utiliza variables_withExperts.xlsx (incluye grade, sub_grade, fico, int_rate, etc.)
    - Imputación: KNNImputer para numéricas en lugar de mediana simple
    - Encoding categórico: TargetEncoder + OrdinalEncoder según tipo de variable
    - Scaling numérico: RobustScaler en lugar de QuantileTransformer (robusto a outliers)
    - Features nuevas: ratios financieros (debt-to-income, utilization rate, etc.)
    """

    def __init__(self, variables_path: str = "data/variables_withExperts.xlsx"):
        """
        Inicializa el preprocesador.

        Args:
            variables_path: Ruta al fichero de configuración de variables
        """
        self.variables_path = variables_path
        self.variables_df = None

        # Componentes de preprocesamiento (se ajustarán en fit)
        self.knn_imputer = None  # Para variables numéricas
        self.cat_imputer = None  # Para variables categóricas
        self.target_encoder = None  # Para categóricas sin orden
        self.ordinal_encoder = None  # Para grade/sub_grade
        self.robust_scaler = None  # Para normalización robusta
        self.text_encoder = None  # Para texto libre

        # Variables aprendidas en fit
        self.numerical_cols = []
        self.categorical_cols = []
        self.ordinal_cols = []  # grade, sub_grade
        self.text_cols = []
        self.target_col = None

        # Columnas finales después de preprocesamiento
        self.feature_names_out_ = []

    def _load_variables_config(self, X: pd.DataFrame):
        """Carga la configuración de variables desde el Excel."""
        self.variables_df = pd.read_excel(self.variables_path)

        # Filtrar variables a usar (posible_predictora = True/Yes/1)
        if 'posible_predictora' in self.variables_df.columns:
            self.variables_df = self.variables_df[
                self.variables_df['posible_predictora'].isin([True, 'True', 'Yes', 'yes', 1, 'TRUE'])
            ].copy()

        # Identificar tipos manualmente basado en las columnas disponibles en X
        available_cols = X.columns.tolist()

        # Variables numéricas (filtrar solo las que existen)
        all_numerical = [
            'loan_amnt', 'funded_amnt', 'int_rate', 'installment', 'annual_inc', 'dti',
            'fico_range_low', 'fico_range_high', 'open_acc', 'pub_rec',
            'revol_bal', 'revol_util', 'total_acc', 'delinq_2yrs', 'inq_last_6mths'
        ]
        self.numerical_cols = [col for col in all_numerical if col in available_cols]

        # Variables categóricas (filtrar solo las que existen)
        all_categorical = [
            'term', 'grade', 'sub_grade', 'emp_length', 'home_ownership',
            'verification_status', 'purpose'
        ]
        all_categorical = [col for col in all_categorical if col in available_cols]

        # Identificar variables ordinales (grade y sub_grade tienen orden natural)
        self.ordinal_cols = [col for col in ['grade', 'sub_grade'] if col in available_cols]

        # Remover ordinales de categóricas normales
        self.categorical_cols = [col for col in all_categorical
                                if col not in self.ordinal_cols]

        # Variables de texto (filtrar solo las que existen)
        all_text = ['title', 'emp_title']
        self.text_cols = [col for col in all_text if col in available_cols]

        # Target (loan_status se asume que ya fue removido antes)

    def _create_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crea features financieras basadas en dominio bancario.

        Features generadas:
        - debt_to_income_ratio: deuda total / ingreso anual
        - revolving_utilization: balance revolving / límite de crédito
        - payment_to_income_ratio: cuota mensual / ingreso mensual
        - fico_mean: promedio de fico_range_low y fico_range_high
        - installment_burden: cuota / ingreso anual (normalizado)
        """
        df = df.copy()

        # Debt-to-income ratio
        if 'annual_inc' in df.columns and 'dti' in df.columns:
            # dti ya es un ratio, pero podemos crear variantes
            df['debt_to_income_ratio'] = df['dti'] / 100.0

        # Revolving utilization rate
        if 'revol_bal' in df.columns and 'revol_util' in df.columns:
            # revol_util ya existe, crear versión normalizada
            df['revolving_utilization_norm'] = df['revol_util'] / 100.0

        # Payment to income ratio (monthly)
        if 'installment' in df.columns and 'annual_inc' in df.columns:
            df['payment_to_income_ratio'] = (df['installment'] * 12) / df['annual_inc']
            df['payment_to_income_ratio'] = df['payment_to_income_ratio'].fillna(0)

        # FICO score medio
        if 'fico_range_low' in df.columns and 'fico_range_high' in df.columns:
            df['fico_mean'] = (df['fico_range_low'] + df['fico_range_high']) / 2

        # Installment burden (cuota como % del ingreso anual)
        if 'installment' in df.columns and 'annual_inc' in df.columns:
            df['installment_burden'] = df['installment'] / (df['annual_inc'] / 12)
            df['installment_burden'] = df['installment_burden'].fillna(0)

        # Loan amount to income ratio
        if 'loan_amnt' in df.columns and 'annual_inc' in df.columns:
            df['loan_to_income'] = df['loan_amnt'] / df['annual_inc']
            df['loan_to_income'] = df['loan_to_income'].fillna(0)

        return df

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'Practica1Preprocess':
        """
        Ajusta todos los transformadores sobre los datos de entrenamiento.

        Args:
            X: DataFrame con las features
            y: Serie con el target (necesario para TargetEncoder)

        Returns:
            self
        """
        # Cargar configuración de variables
        self._load_variables_config(X)

        # Crear features financieras
        X = self._create_financial_features(X)

        # Actualizar lista de columnas numéricas con las nuevas features
        new_features = ['debt_to_income_ratio', 'revolving_utilization_norm',
                       'payment_to_income_ratio', 'fico_mean', 'installment_burden',
                       'loan_to_income']
        self.numerical_cols.extend([f for f in new_features if f in X.columns])

        # 1. IMPUTACIÓN DE MISSINGS
        print("[PREP] Ajustando imputadores...")

        # KNNImputer para variables numéricas (considera vecinos cercanos)
        if self.numerical_cols:
            self.knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
            X[self.numerical_cols] = self.knn_imputer.fit_transform(X[self.numerical_cols])

        # SimpleImputer con estrategia 'most_frequent' para categóricas
        cat_cols_all = self.categorical_cols + self.ordinal_cols
        if cat_cols_all:
            self.cat_imputer = SimpleImputer(strategy='most_frequent')
            X[cat_cols_all] = self.cat_imputer.fit_transform(X[cat_cols_all])

        # 2. ENCODING DE VARIABLES ORDINALES
        print("[ENC] Ajustando encoders ordinales...")
        if self.ordinal_cols and 'grade' in X.columns:
            # Orden natural de grade: A (mejor) -> G (peor)
            grade_order = ['A', 'B', 'C', 'D', 'E', 'F', 'G']

            # Si sub_grade existe, crear orden completo
            if 'sub_grade' in X.columns:
                subgrade_order = []
                for g in grade_order:
                    for i in range(1, 6):
                        subgrade_order.append(f"{g}{i}")

                self.ordinal_encoder = OrdinalEncoder(
                    categories=[grade_order, subgrade_order],
                    handle_unknown='use_encoded_value',
                    unknown_value=-1
                )
            else:
                self.ordinal_encoder = OrdinalEncoder(
                    categories=[grade_order],
                    handle_unknown='use_encoded_value',
                    unknown_value=-1
                )

            X[self.ordinal_cols] = self.ordinal_encoder.fit_transform(X[self.ordinal_cols])

        # 3. ENCODING DE VARIABLES CATEGÓRICAS (TargetEncoder)
        print("[TARGET] Ajustando Target Encoder...")
        if self.categorical_cols and y is not None:
            # TargetEncoder: codifica cada categoría por la media del target
            # Reduce dimensionalidad comparado con OneHotEncoder
            self.target_encoder = TargetEncoder(
                target_type='binary',
                smooth='auto',
                cv=5  # Cross-validation para evitar overfitting
            )
            X[self.categorical_cols] = self.target_encoder.fit_transform(
                X[self.categorical_cols], y
            )

        # 4. ENCODING DE TEXTO LIBRE
        print("[TEXT] Ajustando encoder de texto...")
        if self.text_cols:
            self.text_encoder = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

            for col in self.text_cols:
                if col in X.columns:
                    # Rellenar nulls con string vacío
                    texts = X[col].fillna('').astype(str).tolist()
                    embeddings = self.text_encoder.encode(texts, show_progress_bar=True)

                    # Crear columnas para cada dimensión del embedding
                    for i in range(embeddings.shape[1]):
                        X[f'{col}_emb_{i}'] = embeddings[:, i]

                    # Eliminar columna original de texto
                    X = X.drop(columns=[col])

        # 5. NORMALIZACIÓN DE VARIABLES NUMÉRICAS
        print("[SCALE] Ajustando RobustScaler...")
        numeric_cols_final = [col for col in X.columns if col in self.numerical_cols
                             or col.endswith('_emb_') or col in self.ordinal_cols]

        if numeric_cols_final:
            # RobustScaler: usa mediana e IQR, robusto a outliers
            self.robust_scaler = RobustScaler()
            X[numeric_cols_final] = self.robust_scaler.fit_transform(X[numeric_cols_final])

        # Guardar nombres de features finales
        self.feature_names_out_ = X.columns.tolist()

        print(f"[OK] Preprocesamiento ajustado. Features finales: {len(self.feature_names_out_)}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica las transformaciones ajustadas en fit().

        Args:
            X: DataFrame con las features

        Returns:
            DataFrame transformado
        """
        X = X.copy()

        # Crear features financieras
        X = self._create_financial_features(X)

        # 1. IMPUTACIÓN
        if self.knn_imputer is not None and self.numerical_cols:
            X[self.numerical_cols] = self.knn_imputer.transform(X[self.numerical_cols])

        cat_cols_all = self.categorical_cols + self.ordinal_cols
        if self.cat_imputer is not None and cat_cols_all:
            X[cat_cols_all] = self.cat_imputer.transform(X[cat_cols_all])

        # 2. ENCODING ORDINAL
        if self.ordinal_encoder is not None and self.ordinal_cols:
            X[self.ordinal_cols] = self.ordinal_encoder.transform(X[self.ordinal_cols])

        # 3. ENCODING CATEGÓRICO
        if self.target_encoder is not None and self.categorical_cols:
            X[self.categorical_cols] = self.target_encoder.transform(X[self.categorical_cols])

        # 4. ENCODING TEXTO
        if self.text_encoder is not None and self.text_cols:
            for col in self.text_cols:
                if col in X.columns:
                    texts = X[col].fillna('').astype(str).tolist()
                    embeddings = self.text_encoder.encode(texts, show_progress_bar=False)

                    for i in range(embeddings.shape[1]):
                        X[f'{col}_emb_{i}'] = embeddings[:, i]

                    X = X.drop(columns=[col])

        # 5. NORMALIZACIÓN
        numeric_cols_final = [col for col in X.columns if col in self.numerical_cols
                             or col.endswith('_emb_') or col in self.ordinal_cols]

        if self.robust_scaler is not None and numeric_cols_final:
            X[numeric_cols_final] = self.robust_scaler.transform(X[numeric_cols_final])

        # Asegurar que las columnas coincidan con las de training
        missing_cols = set(self.feature_names_out_) - set(X.columns)
        for col in missing_cols:
            X[col] = 0

        X = X[self.feature_names_out_]

        return X

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> pd.DataFrame:
        """Ajusta y transforma en un solo paso."""
        return self.fit(X, y).transform(X)
