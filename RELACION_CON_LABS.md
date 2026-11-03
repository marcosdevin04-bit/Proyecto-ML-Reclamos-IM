# Relación del notebook con los laboratorios del curso

El código está escrito con el estilo de los labs: `print` con f-strings, constantes en mayúscula, funciones cortas con docstring y bucles explícitos. Esta tabla dice de dónde sale cada parte.

| Parte del notebook | Laboratorio o clase | Qué se usa |
|---|---|---|
| Lectura, filtros y auditoría (sección 1) | Labs 2 y 2A | `read_csv`, fechas, `value_counts`, `crosstab`, filas repetidas |
| Preparación (`flujo`) | Lab 4 | `Pipeline` + `ColumnTransformer` con `OneHotEncoder` |
| Logística y AUC/AP (secciones 2, 5 y 7) | Labs 5, 6 y 8 | `LogisticRegression`, `roc_auc_score`, `average_precision_score`; la AP de un puntaje constante es la prevalencia |
| k vecinos y escalado (sección 3) | Lab 10 | El efecto de la escala de cada columna sobre la distancia |
| Árbol de decisión | Lab 12 | `DecisionTreeClassifier`, profundidad y hoja mínima |
| Bosque e importancias (secciones 6 y 8) | Lab 14 | `RandomForestClassifier`, `ParameterGrid`, `feature_importances_` sumadas por variable original, `permutation_importance` y el desempeño al quitar una variable |
| Boosting (sección 6) | Lab 16 | `AdaBoostClassifier` sobre árboles chicos; profundidad y número de etapas |
| SVM y medición de tiempos (sección 6) | Lab 17 | `SVC` con núcleo RBF, `C` y `gamma`, `time.perf_counter` |
| Red neuronal (sección 6) | Clase 18 | `MLPClassifier` con una capa oculta, `alpha` (L2) y `early_stopping` |
| Focos (sección 4) | Lab 21 | `KMeans` y la curva del silhouette, `linkage` y `fcluster` (Ward y enlace simple), `NearestNeighbors` para el gráfico de k-distancias, `DBSCAN` |
| Zonas dentro del pipeline (sección 5) | Labs 4 y 21 | `KMeans` como paso del `ColumnTransformer`: devuelve la distancia a cada centroide |

## Lo que pide la consigna y no está en un lab

- **Validación temporal.** La consigna avisa que se dicta después. El notebook la arma a mano: entrenar con los reclamos de 2017 y validar con los de 2018, y probar una sola vez sobre 2019.
- **La proyección a metros.** Es la fórmula de la consigna, en la función `proyectar`. Para comparar escalas, las dos coordenadas se dividen por el mismo factor fijo (5 km), como pide la guía de la sección 3, sin clases propias.
