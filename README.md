# Predicción y análisis espacial de reclamos de Montevideo

Proyecto académico de Machine Learning orientado a anticipar qué reclamos de la Intendencia de Montevideo se resuelven en siete días corridos o menos e identificar focos espaciales de problemas urbanos.

El trabajo combina clasificación, agrupamiento y validación temporal sobre los datos abiertos del Sistema Único de Reclamos.

## Objetivos

- Construir un flujo reproducible de ingesta, auditoría y preparación de datos.
- Definir la respuesta y la población de estudio, documentando las exclusiones y sus posibles sesgos.
- Usar únicamente información disponible al ingresar el reclamo.
- Comparar modelos de clasificación mediante ROC-AUC y average precision (AP).
- Analizar el efecto de las unidades y el escalado de las coordenadas.
- Identificar concentraciones geográficas de un mismo tipo de problema.
- Evaluar el aporte de las zonas como atributos predictivos sin fugas de información.
- Recomendar un modelo y focos de intervención a partir del desempeño, el costo de cómputo y la interpretación de los resultados.

## Datos y evaluación

La fuente es el conjunto abierto [Reclamos registrados en el Sistema Único de Reclamos de la Intendencia de Montevideo](https://catalogodatos.gub.uy/dataset/reclamos-registrados-en-el-sistema-unico-de-reclamos-de-la-intendencia-de-montevideo).

- **Ventana de trabajo:** reclamos ingresados entre 2017 y 2019, inclusive.
- **Desarrollo y selección:** datos de 2017–2018, con un esquema de validación justificado por el uso futuro del modelo.
- **Prueba final:** reclamos ingresados en 2019, reservados antes de cualquier ajuste y utilizados solo en la evaluación final y el análisis de importancias.
- **Comparación de familias:** una submuestra común de 50.000 reclamos de 2017–2018, estratificada por año; SVM utiliza 10.000 de esos mismos reclamos.

La ruta del archivo y su fecha de descarga se declararán en la configuración del notebook para identificar la versión utilizada.

## Etapas del proyecto

1. **Datos, respuesta y población:** auditoría, construcción de la etiqueta, selección de entradas y esquema de validación.
2. **Líneas base:** prevalencia y regresión logística sin ubicación.
3. **Coordenadas:** comparación de grados y metros, y del escalado, con k vecinos y un árbol.
4. **Focos espaciales:** k-medias, agrupamiento jerárquico con Ward y enlace simple, y DBSCAN.
5. **Agrupamiento como atributo:** comparación de coordenadas y zonas en una regresión logística, con el agrupamiento dentro del pipeline.
6. **Seis familias:** k vecinos, árbol de decisión, SVM con núcleo RBF, bosque aleatorio, boosting y red neuronal, con grillas pequeñas de hiperparámetros y medición de tiempos.
7. **Evaluación sobre 2019:** comparación final y contraste entre validación temporal y validación aleatoria para el bosque.
8. **Importancias y conclusiones:** importancia por impureza y por permutación, recomendación del modelo, focos y límites del análisis.

## Entrega

El entregable será un notebook basado en el template de la cátedra, con las ocho secciones, ejecutado de punta a punta y con sus resultados, interpretaciones y justificaciones visibles.

**Fecha indicada en la consigna:** jueves 5 de noviembre de 2026. Trabajo en grupos de cuatro; 10 % de la calificación final.

## Estado

Proyecto en etapa inicial. El repositorio se actualizará con la consigna, el template, el análisis y los resultados a medida que se completen las etapas.

## Alcance

El análisis se realiza con fines académicos. Sus conclusiones no constituyen un diagnóstico sobre la gestión de la Intendencia de Montevideo. Las asociaciones observadas y las importancias predictivas no demuestran relaciones causales.

## Integrantes

- Marcos Devincenzi
- Santiago Alba
- Facundo Díaz
- Domingo Leiva

Universidad Católica del Uruguay · Machine Learning · Ingeniería en Inteligencia Artificial y Ciencia de Datos.
