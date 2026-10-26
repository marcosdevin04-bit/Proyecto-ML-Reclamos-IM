# Relación del proyecto con los laboratorios

Se revisaron los diez notebooks `lab*.ipynb` de `C:\ml_env`, sin usar como referencia las soluciones de Ingeniería de Datos ni los paquetes instalados.

| Parte del proyecto | Referencia revisada | Forma usada |
|---|---|---|
| Lectura y DataFrames | Lab 2A | `read_csv`, filtros, fechas, `groupby`, conteos y tablas |
| Auditoría | Lab 2 | Tipos, faltantes, duplicados, distribuciones y hallazgos |
| Preparación | Lab 4, celdas 63–70 | `Pipeline`, `ColumnTransformer`, imputación, one-hot y `StandardScaler` |
| Logística | Lab 5 | `LogisticRegression` y probabilidades |
| AUC y AP | Lab 8 | Función de métricas y comparación de resultados |
| Boosting | Lab 16, celdas 14 y 23 | `AdaBoostClassifier` con árboles de profundidad 1 |
| SVM y grillas | Lab 17, celda 25 | `SVC` y bucle explícito de parámetros |
| k-medias y DBSCAN | Lab 21, celdas 8, 15 y 51 | `KMeans`, silhouette, `NearestNeighbors`, k-distancias y radio visual |
| Jerárquico | Lab 21, celdas 36 y 39 | `linkage` y `fcluster`, Ward y enlace simple |

## Simplificaciones

- Proyección en una función corta con la fórmula de la consigna, sin clase propia para proyección, ajuste, transformación y nombres de columnas.
- Mes y día de semana como categorías, sin senos/cosenos de calendario.
- Grillas con bucles de ajuste, predicción y AP, como en lab 17. Se muestran todas las configuraciones.
- Radio de DBSCAN elegido desde el gráfico antes de agrupar, sin detección automática de rodilla.
- AdaBoost de lab 16 en lugar de GradientBoostingClassifier.
- Sentencias separadas por pasos y nuevas interpretaciones generadas desde las salidas.

## Lo adicional que exige la consigna

Bosque aleatorio, red neuronal, validación temporal e importancia por permutación son obligatorios. No encontré un laboratorio específico entre los diez archivos revisados. Se conservan sus APIs estándar y grillas pequeñas: quitarlos dejaría incompleta la entrega.

`EscalaEspacial` adapta mínimamente `StandardScaler`: aprende medias y varianzas de entrenamiento y usa el mismo divisor en ambos ejes. Cumple la indicación de no deformar la métrica espacial. La proyección es fija y se calcula antes; el divisor aprendido permanece dentro del pipeline.

Los scripts de ejecución, verificación y PDF son herramientas de reproducción y documentación. El código de aprendizaje está en el notebook.

Se mantienen las ocho secciones, las fuentes, la población, la muestra común de 50.000, SVM con 10.000 y la prueba de 2019. La revisión se decide por correspondencia con clase, sin elegir cambios para mejorar 2019.
