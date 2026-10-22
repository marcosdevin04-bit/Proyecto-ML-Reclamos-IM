# Guía para defender el proyecto de reclamos de Montevideo

Esta guía acompaña al notebook ejecutado. Explica qué se hizo, por qué y qué números lo respaldan. Las cifras de validación son de 2018, con entrenamiento en 2017; las de prueba son de 2019. No son intercambiables. Los cuatro integrantes deben poder explicar cualquier sección y reconocer los límites del trabajo.

## Presentación de un minuto

> Queremos estimar al ingresar un reclamo si su cierre se registra dentro de siete días y encontrar concentraciones de baches. Auditamos los datos, construimos la respuesta y usamos únicamente información conocida al ingreso. Elegimos con 2017–2018 y reservamos 2019. La recomendación entre las seis familias es Bosque: en prueba obtiene AUC 0,760 y AP 0,703, frente a 0,746 y 0,706 de la logística sin ubicación. Analizamos también agrupamientos: una zona que divide la ciudad no equivale a un foco denso. El cierre publicado no garantiza reparación física, y los resultados no son un diagnóstico de la gestión municipal.

## Cómo leer el código

**`flujo(modelo)`** arma dos pasos juntos: preparación y algoritmo. Por ejemplo, primero convierte categorías y coordenadas, y después ajusta el bosque. Se llama pipeline porque aplica los pasos en orden.

**`ColumnTransformer`** reparte columnas: códigos de problema van a one-hot; calendario a escalado; ubicación a proyección y escalado espacial. One-hot convierte categorías en columnas 0/1, sin suponer que un código 400 sea el doble de 200.

**`Coordenadas`** convierte latitud/longitud a metros locales y, si corresponde, divide ambos ejes por el mismo factor aprendido del entrenamiento. No deforma la geometría usando un desvío diferente para cada eje.

**`GridSearchCV`** prueba combinaciones declaradas y selecciona por AP sobre la partición temporal. El pipeline completo se vuelve a ajustar en cada candidato. Se clonan los estimadores al comparar: las métricas y los modelos finales no dependen del estado dejado por el último fold.

## 1. Datos, respuesta y población

Leímos **1,709,419 filas**. **395,643** corresponden a ingresos de 2017–2019. Quitamos **14 copias exactas**, excluimos **17,492 abiertos o anulados** y **1 demora imposible/ausente**. No hubo exclusiones geográficas por el rectángulo local. La población final es de **378,136 reclamos**, de los cuales **133,211** son de prueba.

**La etiqueta:** calculamos días entre ingreso y fecha de entrada al estado actual, solo si ese estado es Finalizado. Positivo significa demora no negativa y menor o igual a siete días: exactamente siete cuenta como positivo. El negativo es un finalizado que demoró más de siete días.

**Por qué no usar todos los estados:** la fecha de un reclamo En Proceso es la de entrada a ese estado, no la de resolución. Anulado tampoco significa necesariamente no resuelto. Asignarles una etiqueta automática inventaría un resultado. Excluirlos hace observable la etiqueta, pero introduce selección: el modelo representa finalizados con fechas válidas, no todos los ingresos.

**Límite de la palabra resuelto:** el catálogo dice que Finalizado incluye cierres por no corresponder y otros motivos. Por eso nuestra respuesta es un indicador de cierre administrativo semanal, no prueba de reparación física. También puede haber reaperturas que la foto actual no reconstruye.

**Contraste con metadatos:** la descripción menciona COORD_X/COORD_Y UTM, pero el archivo trae LATITUD/LONGITUD en grados. Se usa lo observado y se documenta la discrepancia. No se inventa que el archivo ya esté en metros.

**Entradas:** códigos de área, grupo y tipo como categorías; calendario derivado de la fecha de ingreso; latitud y longitud. No entran identificador, estado actual ni fecha de cambio de estado. Los dos últimos se conocen después y producirían fuga. Los nombres de área/grupo/tipo se descartan porque duplican sus códigos.

**Validación:** entrenamos en 2017 y validamos en 2018. La alternativa aleatoria mezcla pasado y futuro relativo, y puede compartir episodios y tipos vigentes. El uso previsto recibe nuevos reclamos, por eso elegimos la frontera temporal. Es una sola partición anual, no cinco folds ni un intervalo de confianza. La muestra de familias tiene 50.000 casos proporcionalmente por año; SVM usa 10.000 de esos mismos casos.

**Advertencia temporal concreta:** conocemos las etiquetas desde una foto posterior. Algunos cierres de reclamos de 2017 ocurrieron después de 2018. El ejercicio evalúa transferencia entre años de ingreso; para simular estrictamente lo disponible en una fecha habría que reconstruir historial y cortar también las etiquetas de entrenamiento. Este límite se declara y no se usa esa fecha como entrada.

## 2. Líneas base y métricas

Una línea base es una referencia simple que el modelo tiene que superar. La trivial asigna la misma probabilidad a todos, la prevalencia del entrenamiento. La otra es logística con tipo y calendario, sin ubicación.

| Modelo | AUC en 2018 | AP en 2018 |
| --- | --- | --- |
| Prevalencia | 0,500 | 0,403 |
| Logística sin ubicación | 0,708 | 0,626 |

**AUC:** capacidad de ordenar positivos por encima de negativos. Un AUC de 0,70 no significa 70% de aciertos. Con todos los scores iguales es 0,5.

**AP:** average precision resume precisión y recall al recorrer umbrales. Su referencia trivial es la proporción de positivos del conjunto evaluado, no cero. AP cambia también cuando cambia la prevalencia: no se compara un año con otro ignorando ese cambio.

No elegimos umbral por costos: esta entrega pide scores o probabilidades y métricas de ordenamiento. La SVM usa función de decisión; no se presenta su score como una probabilidad calibrada.

## 3. Coordenadas: unidades y escalado

Cinco kilómetros y cinco mil metros son el mismo lugar, pero cambian la magnitud numérica de las entradas. KNN busca reclamos cercanos en el espacio de atributos; sin escalado una coordenada en miles puede dominar a las categorías 0/1. El árbol pregunta si una coordenada supera un umbral y puede convertir ese umbral sin cambiar las filas separadas.

Usamos la proyección local permitida: x = R por diferencia de longitud en radianes por coseno de la latitud de referencia; y = R por diferencia de latitud. R vale 6.371.000 m y la referencia fija es longitud -56,17°, latitud -34,87°. No se aprende del test.

| Contraste | Variante | Familia | AUC | AP |
| --- | --- | --- | --- | --- |
| Unidad | Grados | KNN | 0,676 | 0,588 |
| Unidad | Metros | KNN | 0,580 | 0,455 |
| Escala | Metros estandarizados | KNN | 0,701 | 0,606 |
| Escala | Metros sin escalar | KNN | 0,580 | 0,455 |
| Unidad | Grados | Árbol | 0,650 | 0,566 |
| Unidad | Metros | Árbol | 0,650 | 0,566 |
| Escala | Metros estandarizados | Árbol | 0,650 | 0,566 |
| Escala | Metros sin escalar | Árbol | 0,650 | 0,566 |

En cada contraste solo cambia unidad o escalado espacial. Los modelos, parámetros, resto de preparación y partición permanecen iguales. Un factor común para ambos ejes conserva proporciones físicas. Se interpretan las diferencias observadas, no se exige que todos los modelos mejoren.

## 4. Agrupamientos y focos

Elegimos **baches, tipo 401**, con **4626 ingresos en 2017–2018**. Tiene ubicación interpretable y supera los 2.000 pedidos. Aquí también contamos abiertos/anulados porque buscamos concentración de reportes, no solo de finalizados. No se usan ubicaciones de 2019.

**K-medias:** fuerza K grupos y asigna todos los puntos. Probamos una curva de K entre 2 y 12; el máximo medido fue **K=4**. La curva y su forma importan: el mejor valor de una grilla no prueba que existan exactamente esos barrios naturales.

**Jerárquico Ward:** agrupa favoreciendo compacidad. **Enlace simple:** puede conectar zonas mediante cadenas de puntos, aunque estén muy extendidas. Se comparan con el mismo K para no cambiar dos cosas a la vez.

**DBSCAN:** busca densidad y puede dejar reclamos sin grupo. Primero se graficaron k-distancias de 5, 10 y 20 vecinos. Elegimos mínimo 10 como soporte intermedio y una rodilla de la curva como radio: **eps=543 m**. La rodilla es una regla aproximada, no una norma de la Intendencia.

| Método | Grupos | Sin grupo | Silhouette | Cobertura |
| --- | --- | --- | --- | --- |
| K-medias | 4 | 0 | 0,466 | 1,000 |
| Jerárquico Ward | 4 | 0 | 0,416 | 1,000 |
| Jerárquico simple | 4 | 0 | 0,325 | 1,000 |
| DBSCAN | 12 | 232 | -0,006 | 0,950 |

**Silhouette** evalúa cohesión y separación geométrica; alto no significa automáticamente utilidad para intervención. En DBSCAN se calcula sin ruido y se informa la cobertura. No se trata a todos los puntos dispersos como un solo grupo. El valor negativo observado obliga a reconocer superposición o encadenamiento y no a declarar éxito por el nombre del método.

**Qué recomendamos:** de los grupos DBSCAN, revisar los que tengan al menos 30 reportes y radio p90 no mayor a 2,5 km. Es un criterio operativo propuesto que excluye grupos demasiado extensos para intervención conjunta. p90 es el radio desde su centro aproximado que contiene al 90% de los puntos. El personal municipal debería confirmar la conveniencia y si hay reclamos reiterados del mismo bache.

| Foco | Longitud aprox. | Latitud aprox. | Reportes | Radio p90 en m |
| --- | --- | --- | --- | --- |
| 5 | -56,145 | -34,818 | 225 | 2128,619 |
| 7 | -56,251 | -34,798 | 50 | 955,738 |
| 1 | -56,134 | -34,787 | 44 | 755,603 |
| 6 | -56,219 | -34,766 | 42 | 1074,902 |

Los centros son coordenadas, no nombres de barrios verificados. Contamos reportes, no defectos físicos únicos ni tasas por habitante.

## 5. Zonas como atributos

Una logística con x e y aprende un efecto espacial lineal en log-odds. Distancias a varios centros permiten describir un patrón más flexible sin cambiar el clasificador. Usamos KMeans dentro del pipeline: aprende centros solo en entrenamiento.

Elegimos **K=12** entre 4, 8 y 12 por AP temporal. Este K se elige para predecir todos los tipos; no se copia el de focos de baches.

| Representación | AUC | AP |
| --- | --- | --- |
| Logística coordenadas | 0,725 | 0,648 |
| Logística distancias a 12 centros | 0,743 | 0,663 |

Ajustar centros antes con toda la muestra revela distribución de ubicaciones de validación, aunque no use etiquetas. Usar tasas de cierre por zona revelaría además respuesta si no se construye fuera de muestra dentro de cada entrenamiento. Nuestra representación no usa la respuesta para agrupar. Si fueran etiquetas de zona, habría que codificarlas como categorías, no números ordenados.

## 6. Seis familias y elección

- **KNN:** estima por semejanza con vecinos; es sensible a escala y puede costar al predecir muchos casos.
- **Árbol:** reglas por cortes; fácil de inspeccionar, pero puede sobreajustar.
- **SVM RBF:** frontera no lineal con distancias; C regula penalización y gamma el alcance del núcleo.
- **Bosque:** promedia muchos árboles con aleatoriedad para reducir variación; no equivale a un único árbol legible.
- **Boosting:** añade árboles para corregir errores del conjunto anterior; profundidad y cantidad controlan flexibilidad.
- **Red neuronal:** combina capas no lineales; se limita tamaño y regularización y usa parada temprana interna.

Las grillas son pequeñas, de dos a cuatro valores por parámetro. Cada elegido y sus bordes aparecen en el notebook. Caer en borde reconoce exploración limitada, no certifica un óptimo. La red no usa test para parar. Los tiempos corresponden al equipo de esta ejecución y no son una promesa para cualquier computadora.

| Familia | AUC 2018 | AP 2018 | Ajuste s | Predicción s |
| --- | --- | --- | --- | --- |
| KNN | 0,694 | 0,604 | 0,036 | 4,689 |
| Árbol | 0,659 | 0,579 | 0,166 | 0,034 |
| SVM RBF | 0,715 | 0,619 | 0,487 | 0,736 |
| Bosque | 0,747 | 0,669 | 1,223 | 0,134 |
| Boosting | 0,741 | 0,657 | 3,398 | 0,058 |
| Red neuronal | 0,751 | 0,669 | 1,485 | 0,046 |

**Criterio fijado antes del test:** entre familias a menos de 0,01 AP del máximo de desarrollo, preferimos menor tiempo de ajuste más predicción. Esa tolerancia es operativa, no una prueba de equivalencia. Resultó **Bosque**, con AP 0,669. La capacidad explicativa se considera en la recomendación: un ensamble ofrece importancias, pero no una explicación causal ni una regla breve para cada caso.

## 7. Evaluación final y validación mezclada

Con configuración cerrada se entrena en la muestra de ambos años y se evalúa una vez en 2019. SVM mantiene sus 10.000. No cambiamos grillas al ver esta tabla. En prueba, el bosque mejora AUC frente a logística sin ubicación, pero su AP queda ligeramente por debajo. Por eso no se demuestra una mejora uniforme ni se justifica reemplazar automáticamente la alternativa simple y barata: la recomendación de desarrollo es un candidato para evaluar prospectivamente, no una garantía de superioridad.

| Familia | AUC 2019 | AP 2019 | Ajuste s | Predicción s |
| --- | --- | --- | --- | --- |
| Logística sin ubicación | 0,746 | 0,706 | 1,199 | 0,175 |
| KNN | 0,694 | 0,629 | 0,099 | 51,611 |
| Árbol | 0,726 | 0,619 | 0,480 | 0,185 |
| SVM RBF | 0,705 | 0,645 | 2,843 | 47,840 |
| Bosque | 0,760 | 0,703 | 2,756 | 0,832 |
| Boosting | 0,756 | 0,707 | 14,008 | 0,867 |
| Red neuronal | 0,723 | 0,679 | 3,097 | 0,212 |

| Esquema bosque | AUC | AP | Diferencia AUC vs. 2019 |
| --- | --- | --- | --- |
| Temporal 2017→2018 | 0,747 | 0,669 | -0,013 |
| 5 folds mezclados | 0,880 | 0,812 | 0,120 |
| 2017–2018 → 2019 | 0,760 | 0,703 | 0,000 |

Con mezcla aleatoria hay reclamos de ambos años en cada lado, posiblemente del mismo entorno, episodio o catálogo. La temporal exige cruzar una frontera de año. También cambia la cantidad de entrenamiento por fold: mezclada usa 80%; temporal solo 2017. Por eso la brecha observada no se atribuye exclusivamente a fuga; influyen tamaño, prevalencia y cambio temporal. La tabla informa el signo real, no el que esperábamos.

## 8. Importancias y límites

El mejor modelo de árboles por AP de desarrollo fue **Bosque**. Impureza describe cómo se ajustaron sus cortes; permutación mide cuánto cae AP al romper una entrada sobre una submuestra fija de **10000 casos de 2019**, con cinco repeticiones. No se ajusta otra decisión con esas importancias.

| Variable | Impureza | Caída AP al permutar | Desvío |
| --- | --- | --- | --- |
| CODIGO_TIPOPROBLEMA | 0,268 | 0,145 | 0,003 |
| CODIGO_GRUPO | 0,133 | 0,063 | 0,004 |
| LONGITUD | 0,200 | 0,043 | 0,002 |
| CODIGO_AREA | 0,069 | 0,041 | 0,001 |
| LATITUD | 0,142 | 0,009 | 0,001 |
| mes_cos | 0,029 | 0,007 | 0,001 |
| dia_indice | 0,098 | 0,000 | 0,000 |
| semana_cos | 0,015 | -0,001 | 0,001 |
| semana_sin | 0,019 | -0,001 | 0,001 |
| mes_sin | 0,028 | -0,003 | 0,001 |

Impureza favorece variables con más oportunidades de corte. Permutación habla de utilidad predictiva fuera de muestra, pero correlaciones pueden ocultar aportes o crear combinaciones irreales. Área, grupo y tipo están relacionados. Para ubicación también permutamos latitud/longitud como bloque: caída AP **0,0449**. No se suman automáticamente las dos importancias individuales. Ninguna importancia demuestra causalidad.

**Límites a reconocer:** cierre administrativo no equivale a reparación; exclusión de abiertos/anulados; foto retrospectiva con etiquetas tardías; cambio de tipos y prevalencia; una sola frontera temporal; posibles reportes del mismo episodio; proyección y rectángulo aproximados; grillas pequeñas; datos prepandemia que no garantizan el comportamiento de 2026. Ninguna conclusión constituye un diagnóstico sobre la gestión de la Intendencia.

## Preguntas para practicar

**¿Por qué no usamos la fecha del estado actual?** Porque al ingresar aún no se conoce; revela información posterior y construye el objetivo.

**¿Por qué la AP base no es cero?** Porque incluso sin distinguir casos hay una proporción positiva en el conjunto; con scores constantes AP coincide con esa prevalencia.

**¿Por qué no elegimos por accuracy?** La consigna pide AUC y AP; accuracy depende de un umbral y puede ocultar el desempeño sobre una clase.

**¿Por qué separar 2019 y no un 20% aleatorio?** Porque queremos medir sobre ingresos posteriores. Mezclar años puede hacer más fácil la validación sin representar ese uso.

**¿Por qué KMeans dentro del pipeline?** Los centros aprenden de los datos. Si ven validación antes, revelan su distribución espacial y contaminan el experimento.

**¿Por qué un silhouette alto no basta?** Porque grupos geométricamente separados pueden ser solo zonas impuestas; una intervención necesita concentración, tamaño y ubicación útiles.

**¿Por qué no usamos costos como en OSE?** Esta consigna no pide una política de umbral; pide estimar resolución semanal y comparar probabilidades o scores.

**¿Podemos decir que un barrio causa demora?** No. Las entradas y su importancia muestran asociación predictiva dentro de la población analizada.

**¿Qué revisar antes de defender?** Rehacer la etiqueta a mano con dos fechas, explicar los conteos de población, leer la tabla de validación contra prueba y poder describir qué aprende cada paso del pipeline.

## Archivos y reproducción

El notebook contiene el código, las salidas, gráficos y respuestas. Los CSV y el resumen JSON preservan las cifras de esta guía. El ZIP oficial está versionado y se verifica su SHA-256; el CSV se extrae automáticamente al ejecutar. La fecha de descarga queda en el manifiesto y se declara en la configuración.

Para ejecutar desde un kernel nuevo: instalar requirements.txt y correr `python ejecutar_notebook.py`. Para regenerar esta guía: instalar requirements-documentacion.txt y correr `python scripts/crear_guia.py`. No hay que regenerar el notebook desde el constructor para leer resultados: hacerlo reinicia sus salidas.

Fuentes: consigna de la cátedra incluida en el repo; catálogo oficial y descripción de campos conservados con los datos. Los parámetros y resultados exactos están en el notebook, no dependen de esta explicación resumida.
