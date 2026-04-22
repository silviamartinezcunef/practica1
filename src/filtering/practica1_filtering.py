"""
Clase de filtrado para la Práctica 1 - Pipeline ML Detección de Impago
Implementa métodos alternativos de selección de features.

Autores: Silvia Martínez Moreno
Fecha: Abril 2026
"""

import pandas as pd
import numpy as np
from typing import Optional
from sklearn.feature_selection import (
    SelectKBest,
    mutual_info_classif,
    VarianceThreshold,
    SelectFromModel
)
from sklearn.ensemble import RandomForestClassifier


class Practica1Filtering:
    """
    Clase de filtrado alternativa que utiliza métodos diferentes a BaseFiltering.

    Métodos implementados:
    1. VarianceThreshold: elimina features con varianza muy baja
    2. Mutual Information: selecciona features con mayor información mutua con el target
    3. SelectFromModel con Random Forest: selecciona features por importancia del modelo

    Estos métodos son complementarios y se aplican en secuencia.
    """

    def __init__(
        self,
        variance_threshold: float = 0.01,
        mutual_info_k: Optional[int] = None,
        mutual_info_percentile: int = 70,
        rf_threshold: str = 'median',
        random_state: int = 42,
        verbose: bool = True
    ):
        """
        Inicializa el filtrador.

        Args:
            variance_threshold: Umbral mínimo de varianza (features con menos se eliminan)
            mutual_info_k: Número fijo de features a seleccionar por MI (None = usar percentil)
            mutual_info_percentile: Percentil de features a mantener por MI (si k=None)
            rf_threshold: Umbral de importancia para RF ('mean', 'median', o valor float)
            random_state: Semilla para reproducibilidad
            verbose: Mostrar información de progreso
        """
        self.variance_threshold = variance_threshold
        self.mutual_info_k = mutual_info_k
        self.mutual_info_percentile = mutual_info_percentile
        self.rf_threshold = rf_threshold
        self.random_state = random_state
        self.verbose = verbose

        # Filtros (se ajustarán en fit)
        self.variance_filter = None
        self.mi_filter = None
        self.rf_filter = None

        # Variables aprendidas
        self.feature_names_in_ = []
        self.feature_names_out_ = []
        self.n_features_in_ = 0
        self.n_features_out_ = 0

        # Estadísticas por etapa
        self.n_features_after_variance_ = 0
        self.n_features_after_mi_ = 0
        self.n_features_after_rf_ = 0

    def _log(self, message: str):
        """Imprime mensaje si verbose=True."""
        if self.verbose:
            print(message)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'Practica1Filtering':
        """
        Ajusta los filtros sobre los datos de entrenamiento.

        Args:
            X: DataFrame con las features preprocesadas
            y: Serie con el target

        Returns:
            self
        """
        self.feature_names_in_ = X.columns.tolist()
        self.n_features_in_ = len(self.feature_names_in_)

        self._log(f"[FILTER] Iniciando filtrado de features...")
        self._log(f"   Features iniciales: {self.n_features_in_}")

        X_filtered = X.copy()

        # FILTRO 1: VarianceThreshold
        # Elimina features con varianza muy baja (casi constantes)
        self._log(f"\n[1] Aplicando VarianceThreshold (umbral={self.variance_threshold})...")

        self.variance_filter = VarianceThreshold(threshold=self.variance_threshold)
        X_filtered = pd.DataFrame(
            self.variance_filter.fit_transform(X_filtered),
            columns=X_filtered.columns[self.variance_filter.get_support()],
            index=X_filtered.index
        )

        self.n_features_after_variance_ = X_filtered.shape[1]
        removed_variance = self.n_features_in_ - self.n_features_after_variance_
        self._log(f"   [-] Eliminadas: {removed_variance} features")
        self._log(f"   [OK] Restantes: {self.n_features_after_variance_}")

        # FILTRO 2: Mutual Information
        # Selecciona features con mayor información mutua con el target
        self._log(f"\n[2] Aplicando Mutual Information...")

        if self.mutual_info_k is not None:
            # Seleccionar k mejores features
            k = min(self.mutual_info_k, X_filtered.shape[1])
            self.mi_filter = SelectKBest(score_func=mutual_info_classif, k=k)
            self._log(f"   Seleccionando top {k} features")
        else:
            # Seleccionar por percentil
            # Calcular scores primero para determinar k
            mi_scores = mutual_info_classif(X_filtered, y, random_state=self.random_state)
            threshold = np.percentile(mi_scores, 100 - self.mutual_info_percentile)
            k = int(np.sum(mi_scores >= threshold))
            k = max(k, 1)  # Al menos 1 feature

            self.mi_filter = SelectKBest(score_func=mutual_info_classif, k=k)
            self._log(f"   Seleccionando top {self.mutual_info_percentile}% (k={k})")

        X_filtered = pd.DataFrame(
            self.mi_filter.fit_transform(X_filtered, y),
            columns=X_filtered.columns[self.mi_filter.get_support()],
            index=X_filtered.index
        )

        self.n_features_after_mi_ = X_filtered.shape[1]
        removed_mi = self.n_features_after_variance_ - self.n_features_after_mi_
        self._log(f"   [-] Eliminadas: {removed_mi} features")
        self._log(f"   [OK] Restantes: {self.n_features_after_mi_}")

        # FILTRO 3: SelectFromModel con Random Forest
        # Selecciona features importantes según un modelo RF ligero
        self._log(f"\n[3] Aplicando SelectFromModel con Random Forest...")
        self._log(f"   Umbral de importancia: {self.rf_threshold}")

        # Entrenar RF ligero para obtener importancias
        rf = RandomForestClassifier(
            n_estimators=50,
            max_depth=10,
            min_samples_split=100,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )

        self.rf_filter = SelectFromModel(rf, threshold=self.rf_threshold, prefit=False)
        X_filtered = pd.DataFrame(
            self.rf_filter.fit_transform(X_filtered, y),
            columns=X_filtered.columns[self.rf_filter.get_support()],
            index=X_filtered.index
        )

        self.n_features_after_rf_ = X_filtered.shape[1]
        removed_rf = self.n_features_after_mi_ - self.n_features_after_rf_
        self._log(f"   [-] Eliminadas: {removed_rf} features")
        self._log(f"   [OK] Restantes: {self.n_features_after_rf_}")

        # Guardar nombres finales
        self.feature_names_out_ = X_filtered.columns.tolist()
        self.n_features_out_ = len(self.feature_names_out_)

        self._log(f"\n{'='*60}")
        self._log(f"[OK] Filtrado completado:")
        self._log(f"   Features iniciales: {self.n_features_in_}")
        self._log(f"   Features finales: {self.n_features_out_}")
        self._log(f"   Reducción: {self.n_features_in_ - self.n_features_out_} "
                 f"({100 * (1 - self.n_features_out_/self.n_features_in_):.1f}%)")
        self._log(f"{'='*60}")

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Aplica los filtros ajustados en fit().

        Args:
            X: DataFrame con las features preprocesadas

        Returns:
            DataFrame filtrado
        """
        X_filtered = X.copy()

        # Aplicar filtros en secuencia
        if self.variance_filter is not None:
            X_filtered = pd.DataFrame(
                self.variance_filter.transform(X_filtered),
                columns=X_filtered.columns[self.variance_filter.get_support()],
                index=X_filtered.index
            )

        if self.mi_filter is not None:
            X_filtered = pd.DataFrame(
                self.mi_filter.transform(X_filtered),
                columns=X_filtered.columns[self.mi_filter.get_support()],
                index=X_filtered.index
            )

        if self.rf_filter is not None:
            X_filtered = pd.DataFrame(
                self.rf_filter.transform(X_filtered),
                columns=X_filtered.columns[self.rf_filter.get_support()],
                index=X_filtered.index
            )

        return X_filtered

    def fit_transform(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Ajusta y transforma en un solo paso."""
        return self.fit(X, y).transform(X)

    def get_feature_importance_report(self) -> pd.DataFrame:
        """
        Genera un reporte de importancias de features del modelo Random Forest.

        Returns:
            DataFrame con features y sus importancias ordenadas
        """
        if self.rf_filter is None or not hasattr(self.rf_filter.estimator_, 'feature_importances_'):
            return pd.DataFrame()

        # Obtener features después de MI
        features_after_mi = []
        current_mask = self.variance_filter.get_support()
        for i, selected in enumerate(self.mi_filter.get_support()):
            if selected:
                original_idx = np.where(current_mask)[0][i]
                features_after_mi.append(self.feature_names_in_[original_idx])

        # Crear DataFrame de importancias
        importances = self.rf_filter.estimator_.feature_importances_
        df_importance = pd.DataFrame({
            'feature': features_after_mi,
            'importance': importances,
            'selected': self.rf_filter.get_support()
        })

        df_importance = df_importance.sort_values('importance', ascending=False)

        return df_importance
