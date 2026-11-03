# Guía para defender la Entrega 2

Resume qué hicimos en cada sección, por qué, y con qué números lo sostenemos. Todas las cifras salen de las salidas de `entrega_reclamos_im.ipynb`. En la defensa le pueden pedir a cualquiera de los cuatro que explique cualquier celda: esta guía ayuda, pero no reemplaza leer el código.

Las cifras de validación son de entrenar con 2017 y validar con 2018 sobre la muestra de 50 000. Las de «2019» son la prueba, usada una sola vez.

## Presentación de un minuto

> Usamos el registro abierto de reclamos de la Intendencia, de 2017 a 2019. Antes de modelar encontramos tres cosas que condicionan todo: el catálogo de Limpieza cambió el 26/12/2017, la proporción de reclamos cerrados en una semana sube de 33 % a 43 % en dos años, y el 12 % de los reclamos se cierra administrativamente más de un año después. Por eso validamos en el tiempo: entrenamos con 2017, validamos con 2018 y probamos una sola vez con 2019. Lo que más determina que un reclamo se cierre en una semana es el tipo de problema; la ubicación suma algo. En 2019, el mejor modelo es una regresión logística con distancias a 40 zonas de k-medias: AP 0,720 contra 0,700 sin ubicación, más barata y más explicable que el boosting o la red. Los modelos de árboles eran los mejores en validación pero no en 2019. Para los focos usamos bocas de tormenta obstruidas: DBSCAN encuentra 52 focos y recomendamos los 5 mayores. Nada de esto es un diagnóstico de la gestión de la Intendencia.

## Cómo está armado el código

- **`proyectar(longitud, latitud)`**: la fórmula de la consigna para pasar de grados a metros respecto de un punto fijo (34,87 °S, 56,17 °O).
- **`flujo(modelo, geo, k_zonas)`**: el `Pipeline` del lab 4. Hace one-hot de área, grupo, tipo, mes y día de la semana; agrega las coordenadas (`geo`) o, si se pide, las distancias a `k_zonas` centroides de k-medias.
- **`evaluar(modelo, entrenamiento, evaluacion, columnas)`**: ajusta, puntúa y devuelve AUC, AP y los dos tiempos. Si el modelo no da probabilidades (la SVM), usa `decision_function`.
- **`crear(familia, p)`**: devuelve el estimador de cada familia con los hiperparámetros `p`. La sección 6 recorre cada grilla con `ParameterGrid`, como el lab 14.
- **`ent`, `val`**: la muestra de 2017 y la de 2018. **`muestra`**: los 50 000. **`prueba`**: los 133 211 reclamos de 2019.

## 1. Datos, respuesta y población

- **Ingesta:** 1 709 419 reclamos en el archivo; 395 643 en la ventana (130 178, 125 227 y 140 238 por año).
- **Respuesta:** `resuelto_7d = 1` si entre el ingreso y la fecha del estado Finalizado hay 7 días o menos. Las fechas no tienen hora, así que se cuentan días.
- **Población:** quitamos 14 copias, 17 492 anulados o abiertos y 1 reclamo con cierre antes del ingreso. Quedan 378 136. Sacar los abiertos sube un poco la prevalencia, y más en 2019 (0,9 % en proceso) que en 2017 (0,2 %).
- **Defectos clave:**
  - El cambio de catálogo: «Muebles y electrodomésticos» pasa del código 301 al 1402, «Podas» del 241 al 1381, etc. El 43,8 % de la validación 2018 tiene tipos con menos de 100 casos en 2017.
  - La prevalencia cambia: 33,2 %, 40,6 % y 42,9 %.
  - Los cierres tardíos: la mediana de la demora es 13 días, el percentil 90 es 533; el 23/6/2021 se cerraron 1 565 reclamos de la ventana el mismo día.
  - La descripción de campos dice UTM y el archivo trae grados.
- **Entradas:** área, grupo y tipo (códigos), mes y día de la semana, y ubicación. Afuera: estado y fecha del estado (no existen al ingresar), nombres (redundantes) y el número de reclamo.
- **Validación:** 2017 → 2018. Reproduce el uso: entrenar con el pasado y aplicar al año siguiente. La validación mezclada pone reclamos del mismo día de los dos lados (sección 7).

## 2. Línea base

- Prevalencia: AUC 0,500, AP 0,406. **La AP de un puntaje constante es la prevalencia del conjunto evaluado**, no cero: cualquier corte toma una porción al azar y su precisión es la proporción de positivos.
- Logística sin ubicación: AUC 0,718, AP 0,635.
- La fecha como tendencia (años desde 2017) no aporta (AP 0,634) y siempre cae fuera del rango de entrenamiento: por eso entra como mes y día de la semana.

## 3. Coordenadas

| Variante | k vecinos (AUC / AP) | Árbol |
|---|---|---|
| Grados | 0,721 / 0,641 | 0,642 / 0,567 |
| Metros sin escalar | 0,563 / 0,447 | igual |
| Metros estandarizados | 0,728 / 0,645 | igual |

- **El árbol no cambia:** corta un eje por vez y el orden de los valores no cambia.
- **En k vecinos manda la escala frente a las indicadoras.** En metros (desvío 4 600) la distancia física tapa al tipo de problema y la AP cae casi a la prevalencia. En grados (desvío 0,05) la ubicación casi no cuenta.
- La estandarización divide x e y por el **mismo** factor fijo (5 km): así no se deforma la ciudad y no se aprende nada de los datos.

## 4. Focos

- **Tipo:** boca de tormenta obstruida (2 984 reclamos en 1 976 puntos). Bache no servía: DBSCAN unía 3 677 baches en un grupo de 9 km.
- **k-medias:** curva irregular de 0,347 a 0,454 (máximo en K = 8). No hay un número natural de zonas.
- **ε:** la mediana de la distancia al 10.º vecino, 270 m. El codo de la curva (600–700 m) une todo el casco urbano: separa ciudad de periferia, no focos de fondo.
- **Resultados:** k-medias y Ward reparten la ciudad en 8 zonas; enlace simple encadena (un grupo de 2 968); DBSCAN encuentra 52 focos y deja 1 065 reclamos como ruido.
- **Silhouette:** elige k-medias (0,454 contra 0,407 de DBSCAN sin ruido), que no encuentra focos. Premia la compacidad, no la concentración.
- **Focos recomendados:** los 5 grupos de DBSCAN con 70 reclamos o más: 770 reclamos en total, radios de 0,6 a 1,1 km. El quinto tiene 74 reclamos en solo 29 puntos: son pocas bocas reclamadas muchas veces.

## 5. Zonas como atributo

- Distancias a los centroides de k-medias, **dentro** del `ColumnTransformer`.
- Grilla K ∈ {5, 10, 20, 40, 80, 160}. La AP sube cada vez menos; criterio: el menor K a menos de 0,005 de la mejor AP, que es **K = 40** (AP 0,667 contra 0,653 de las coordenadas crudas).
- **Por qué ayuda a un modelo lineal:** con x e y, la logística solo puede representar un efecto en línea recta hacia algún lado. Con 40 distancias puede dar un efecto distinto a cada parte de la ciudad.
- **Afuera del pipeline,** los centroides se calcularían con las ubicaciones de validación. **Con la respuesta** (tasa de cierre por zona), cada reclamo ayudaría a definir su propia entrada.

## 6. Seis familias

| Familia | Elegida | AP validación |
|---|---|---|
| k vecinos | k = 51 | 0,641 |
| Árbol | sin límite de profundidad, hoja 120 | 0,657 |
| SVM RBF (10 000) | C = 0,5, γ = 0,1 | 0,614 |
| Bosque | 70 % de variables, hoja 5 | 0,707 |
| Boosting (AdaBoost) | profundidad 8, 600 etapas | 0,715 |
| Red neuronal | una capa de 32, α = 0,01 | 0,698 |

Bordes declarados: la profundidad del árbol (sin límite es el extremo natural), el 70 % del bosque (meseta: 0,705 con 40 %) y la profundidad 8 del boosting (la AP sigue subiendo).

## 7. 2019

| Modelo | AUC | AP |
|---|---|---|
| Logística sin ubicación | 0,741 | 0,700 |
| Logística con 40 zonas | 0,767 | **0,720** |
| Boosting | **0,767** | 0,683 |
| Red neuronal | 0,749 | 0,711 |
| Bosque | 0,753 | 0,651 |

- **Bosque, tres estimaciones:** validación 2017 → 2018 AUC 0,761, mezclada 0,882 y 2019 0,753. La mezclada sobrestima en 0,13.
- **Mecanismo de la brecha:** el 85 % de los reclamos de validación mezclada tiene en entrenamiento otro del mismo tipo y del mismo día, y la etiqueta de esos pares acierta el 69 % (contra 63 %). Al usarse, eso no existe.
- **Recomendado:** logística con 40 zonas. Tiene la mejor AP, empata el AUC del boosting, ajusta en unos segundos (el boosting tarda casi un minuto) y se explica con coeficientes. La red queda a 0,009 de AP: no alcanza para elegirla.

## 8. Importancias

- **Modelo:** el boosting, el mejor basado en árboles por AP de validación.
- **Lectura:** la impureza infla las variables con muchos cortes (coordenada y 0,216, mes 0,132), que en 2019 casi no aportan (permutación 0,013 y 0,007). **La impureza habla del ajuste; la permutación sobre 2019, del problema.**
- **El tipo de problema domina** (permutación 0,151):
  - Se cierran en una semana casi siempre: conexión obstruida (94 %) y animales muertos (95 %).
  - Casi nunca: reposición de vereda (0,4 %) y traslado de contenedor (1,5 %).
- **Ubicación:** sin ella, el boosting pierde 0,048 de AUC; en la logística, las zonas suman 0,020 de AP.

## Preguntas para practicar

**¿Por qué la AP trivial es 0,406 y no 0?** Porque es la prevalencia de 2018: un puntaje constante ordena al azar.

**¿Por qué no validación cruzada mezclada?** Porque pone reclamos del mismo día en entrenamiento y validación; infla el AUC del bosque de 0,753 a 0,882.

**¿Por qué cambiaron de bache a boca de tormenta?** Porque los baches están repartidos de forma pareja: DBSCAN no encuentra focos, encuentra la ciudad.

**¿Por qué ε = 270 m y no el codo?** El codo une todo el casco urbano. La mediana marca «más denso que lo típico», que es lo que es un foco.

**¿Por qué el silhouette no elige DBSCAN?** Porque mide compacidad y separación, no concentración; una partición en zonas compactas lo maximiza aunque no haya focos.

**¿Qué pasa si k-medias se ajusta fuera del pipeline?** Los centroides usarían las ubicaciones de validación: la estimación sale optimista.

**¿Por qué recomiendan una logística y no el boosting?** En 2019 tiene mejor AP y el mismo AUC, ajusta unas 20 veces más rápido y se puede explicar.

**¿Qué es el cambio de catálogo y por qué importa?** El 26/12/2017 Limpieza cambió sus códigos; en 2018, casi la mitad de la validación tiene tipos casi sin casos en 2017.

**¿Qué mide la respuesta?** El cierre administrativo en el sistema, no la reparación: hay cierres en bloque años después.

## Antes de entregar

1. Que los cuatro puedan explicar `flujo`, `evaluar` y el esquema 2017 → 2018.
2. Repasar por qué la AP trivial es la prevalencia y por qué el silhouette no elige los focos.
3. Abrir el notebook y comprobar que estas cifras aparecen en las salidas.
4. Entregar por Moodle el notebook ejecutado.
