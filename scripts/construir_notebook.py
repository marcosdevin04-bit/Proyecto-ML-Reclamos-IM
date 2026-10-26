from pathlib import Path
import json, textwrap
root=Path(__file__).resolve().parents[1]
nb=json.loads((root/'entrega_reclamos_im_template.ipynb').read_text(encoding='utf-8'))
C={};M={}
def code(i,s): C[i]=textwrap.dedent(s).strip()+'\n'
def md(i,s): M[i]=textwrap.dedent(s).strip()+'\n'

code(2, r'''
import os, json, zipfile, hashlib, time
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import roc_auc_score, average_precision_score, silhouette_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from scipy.cluster.hierarchy import linkage, fcluster
from sklearn.inspection import permutation_importance
from IPython.display import display, Markdown
ARCHIVO = Path('data/reclamos.csv')
ZIP_ORIGINAL = Path('data/reclamos.zip')
METADATOS = Path('data/metadata.txt')
MANIFIESTO = Path('data/fuentes.json')
FECHA_DESCARGA = json.loads(MANIFIESTO.read_text(encoding='utf-8'))['fecha_descarga']
SEMILLA = 42
np.random.seed(SEMILLA)
SALIDA = Path('resultados')
SALIDA.mkdir(exist_ok=True)
plt.rcParams.update({'figure.figsize': (9, 4), 'axes.grid': True, 'grid.alpha': 0.2})
pd.set_option('display.max_colwidth', 90)
RES = {}

def guardar(nombre, tabla):
    tabla.to_csv(SALIDA / (nombre + '.csv'), index=False)
    display(tabla)

def mostrar(texto):
    display(Markdown(texto))

def metricas(y, p):
    return {'AUC': roc_auc_score(y, p), 'AP': average_precision_score(y, p)}

def reloj_fit_predict(modelo, Xa, ya, Xv):
    t = time.perf_counter()
    modelo.fit(Xa, ya)
    ajuste = time.perf_counter() - t
    t = time.perf_counter()
    p = modelo.predict_proba(Xv)[:, 1] if hasattr(modelo, 'predict_proba') else modelo.decision_function(Xv)
    return (p, ajuste, time.perf_counter() - t)
print('Fecha de descarga:', FECHA_DESCARGA)

def proyectar(longitud, latitud):
    """Fórmula local de la consigna; no aprende nada del conjunto."""
    radio = 6371000
    lon0 = np.radians(-56.17)
    lat0 = np.radians(-34.87)
    x = radio * (np.radians(longitud) - lon0) * np.cos(lat0)
    y = radio * (np.radians(latitud) - lat0)
    return np.column_stack([x, y])
''')

code(4, r'''
manifest = json.loads(MANIFIESTO.read_text(encoding='utf-8'))
assert hashlib.sha256(ZIP_ORIGINAL.read_bytes()).hexdigest() == manifest['reclamos.zip']['sha256']
if not ARCHIVO.exists():
    with zipfile.ZipFile(ZIP_ORIGINAL) as z:
        with z.open('Reclamos.csv') as origen, ARCHIVO.open('wb') as destino:
            import shutil
            shutil.copyfileobj(origen, destino)
crudo = pd.read_csv(ARCHIVO, low_memory=False)
crudo['ingreso'] = pd.to_datetime(crudo.FECHA_INGRESO_RECLAMO, errors='coerce')
crudo['estado_desde'] = pd.to_datetime(crudo.FECHA_DESDE_EN_ESTADO, errors='coerce')
print('Original:', crudo.shape, 'fechas de ingreso inválidas:', crudo.ingreso.isna().sum())
print(METADATOS.read_text(encoding='utf-8-sig'))
ventana = crudo.loc[crudo.ingreso.dt.year.between(2017, 2019)].copy()
ventana['anio'] = ventana.ingreso.dt.year
guardar('filas_por_anio', ventana.groupby('anio').size().rename('reclamos').reset_index())
guardar('estados_por_anio', pd.crosstab(ventana.anio, ventana.DESC_ESTADO).reset_index())
guardar('auditoria_columnas', pd.DataFrame({'columna': crudo.columns, 'tipo': crudo.dtypes.astype(str).values, 'faltantes': crudo.isna().sum().values, 'cardinalidad': crudo.nunique().values}))
print(ventana[['LATITUD', 'LONGITUD']].describe())
for col in ['DESC_AREA', 'DESC_GRUPO', 'DESC_TIPOPROBLEMA']:
    print('\nDistribución', col)
    print(ventana[col].value_counts().to_string())
fig, ax = plt.subplots(1, 3, figsize=(13, 3.5))
ventana.LONGITUD.hist(bins=50, ax=ax[0])
ax[0].set_title('Longitud')
ventana.LATITUD.hist(bins=50, ax=ax[1])
ax[1].set_title('Latitud')
ventana.groupby(ventana.ingreso.dt.to_period('M')).size().plot(ax=ax[2])
ax[2].set_title('Ingresos por mes')
plt.tight_layout()
plt.show()
pasos = [['Ventana 2017–2019', len(ventana), 'Ninguno', 'Años prepandemia']]
duplicadas = ventana.duplicated(subset=list(crudo.columns)).sum()
datos = ventana.drop_duplicates(subset=list(crudo.columns)).copy()
assert datos.NUMERO_RECLAMO.is_unique, 'IDs conflictivos: inspeccionar antes de continuar'
pasos.append(['Sin copias', len(datos), f'{duplicadas} copias', 'Se supone duplicación de publicación'])
no_final = (datos.DESC_ESTADO != 'Finalizado').sum()
datos = datos[datos.DESC_ESTADO == 'Finalizado'].copy()
pasos.append(['Finalizados', len(datos), f'{no_final} abiertos/anulados', 'Condiciona al cierre registrado'])
datos['dias'] = (datos.estado_desde - datos.ingreso).dt.total_seconds() / 86400
invalidos = (datos.dias.isna() | (datos.dias < 0)).sum()
datos = datos[datos.dias.notna() & (datos.dias >= 0)].copy()
pasos.append(['Demora válida', len(datos), f'{invalidos} negativas/ausentes', 'Resultado desconocido de los excluidos'])
geo = datos.LATITUD.between(-35.1, -34.5) & datos.LONGITUD.between(-56.6, -55.9)
n_geo = (~geo).sum()
datos = datos[geo].sort_values('ingreso').reset_index(drop=True)
pasos.append(['Población final', len(datos), f'{n_geo} ubicaciones inválidas', 'Geolocalizables; rectángulo no es límite oficial'])
guardar('poblacion', pd.DataFrame(pasos, columns=['paso', 'reclamos', 'exclusion', 'sesgo']))
datos['resuelto_7d'] = (datos.dias <= 7).astype(int)
guardar('prevalencia_anual', datos.groupby('anio').resuelto_7d.agg(['size', 'mean']).reset_index())
print('Demora en días:')
print(datos.dias.describe(percentiles=[0.5, 0.9, 0.99]))
hallazgos = [['Copias exactas', str(duplicadas), 'Eliminar antes de particionar'], ['No finalizados', str(no_final), 'Excluir: fecha actual no es cierre observable'], ['Demoras imposibles/ausentes', str(invalidos), 'Excluir antes de etiquetar'], ['Coordenadas', 'CSV LATITUD/LONGITUD; metadatos COORD_X/Y UTM', 'Usar grados y proyectar localmente'], ['Estados por año', 'Tabla estados_por_anio', 'Declarar sesgo de selección por cierre'], ['Precisión de fechas', str(datos.ingreso.dt.hour.value_counts().to_dict()), 'Usar calendario diario, no hora']]
guardar('hallazgos', pd.DataFrame(hallazgos, columns=['defecto', 'evidencia', 'decision']))
exclusiones = [['NUMERO_RECLAMO', 'Identificador, no describe el problema', 'Identificación'], ['CODIGO_ESTADO / DESC_ESTADO', 'Estado actual; define población', 'Disponibilidad'], ['FECHA_DESDE_EN_ESTADO', 'Fecha posterior; construye la respuesta', 'Disponibilidad'], ['FECHA_INGRESO_RECLAMO', 'Se transforma a calendario', 'Representación'], ['DESC_AREA / DESC_GRUPO / DESC_TIPOPROBLEMA', 'Se usan códigos categóricos, no duplicar nombres', 'Redundancia'], ['dias / resuelto_7d / anio', 'Respuesta y partición, no entradas', 'Respuesta/partición']]
guardar('columnas_excluidas', pd.DataFrame(exclusiones, columns=['columna', 'razon', 'motivo']))
for col in ['CODIGO_AREA', 'CODIGO_GRUPO', 'CODIGO_TIPOPROBLEMA']:
    datos[col] = datos[col].astype(str)
datos['dia_indice'] = (datos.ingreso - pd.Timestamp('2017-01-01')).dt.days
datos['mes_ingreso'] = datos.ingreso.dt.month.astype(str)
datos['dia_semana'] = datos.ingreso.dt.dayofweek.astype(str)
datos[['x', 'y']] = proyectar(datos.LONGITUD, datos.LATITUD)
CAT = ['CODIGO_AREA', 'CODIGO_GRUPO', 'CODIGO_TIPOPROBLEMA', 'mes_ingreso', 'dia_semana']
NUM = ['dia_indice']
GEO = ['x', 'y']
FEATURES = CAT + NUM + GEO
desarrollo = datos[datos.anio < 2019].copy()
prueba = datos[datos.anio == 2019].copy()
indices = train_test_split(np.arange(len(desarrollo)), train_size=50000, random_state=SEMILLA, stratify=desarrollo.anio)[0]
muestra = desarrollo.iloc[indices].sort_values('ingreso').reset_index(drop=True)
X = muestra[FEATURES]
y = muestra.resuelto_7d.to_numpy()
it = np.flatnonzero(muestra.anio.to_numpy() == 2017)
iv = np.flatnonzero(muestra.anio.to_numpy() == 2018)
CV = [(it, iv)]
sv_idx = train_test_split(np.arange(len(muestra)), train_size=10000, random_state=SEMILLA, stratify=muestra.anio)[0]
Xs = X.iloc[sv_idx].reset_index(drop=True)
ys = y[sv_idx]
sv_anio = muestra.anio.iloc[sv_idx].to_numpy()
CVS = [(np.flatnonzero(sv_anio == 2017), np.flatnonzero(sv_anio == 2018))]
RES['datos'] = {'original': len(crudo), 'ventana': len(ventana), 'poblacion': len(datos), 'desarrollo': len(desarrollo), 'prueba': len(prueba), 'muestra': len(X), 'train_2017': len(it), 'validacion_2018': len(iv), 'copias': int(duplicadas), 'no_final': int(no_final), 'demoras_invalidas': int(invalidos), 'geo_invalidas': int(n_geo)}
del crudo
''')

md(5, r'''
**Población y respuesta.** Cierre registrado como Finalizado en siete días corridos o menos. Según el catálogo, Finalizado puede incluir cierres por no corresponder: no garantiza reparación física. No se confunde la fecha de En Proceso con resolución, ni se etiqueta automáticamente Anulado como fracaso. Excluir abiertos y anulados permite observar una etiqueta, pero condiciona la población a un estado futuro. La tabla anterior cuantifica cada exclusión y su sesgo.

**Contraste con metadatos.** La descripción publicada habla de COORD_X/COORD_Y en UTM; el archivo real tiene LATITUD/LONGITUD en grados. Se respeta el archivo observado y se proyecta localmente. No se interpretan grados como metros. Fechas de precisión diaria impiden analizar horas exactas.

**Entradas.** Códigos de área, grupo y tipo se tratan como categorías, no magnitudes: los niveles amplios pueden aportar ante tipos raros. Se descartan sus nombres redundantes. Fecha relativa, mes y día de semana representan calendario; identificador, estado y fecha de cambio de estado quedan fuera por razones distintas, explicitadas en la tabla.

**Validación:** entrenar con ingresos de 2017 y validar en 2018. Se descarta mezclar filas como criterio principal porque el uso recibe reclamos posteriores y cambian prevalencia y catálogo. No es una media de cinco folds; es una única frontera anual. La muestra común tiene 50.000 casos estratificados por año. 2019 queda reservado para las secciones 7 y 8.

**Límite temporal:** el archivo es una foto retrospectiva. Algunas etiquetas de ingresos de 2017 se registraron después de 2018. Para simular estrictamente un despliegue histórico habría que cortar también la disponibilidad de etiquetas en cada frontera y reconstruir reaperturas. Se declara esa limitación: el ejercicio evalúa transferencia por año de ingreso, no disponibilidad histórica completa de las etiquetas.
''')

code(7, r'''
# StandardScaler se vio en lab 4. Esta adaptación mínima cumple el factor común de la consigna.
class EscalaEspacial(StandardScaler):
    def fit(self, X, y=None):
        super().fit(X, y)
        factor = np.sqrt(self.var_.mean())
        if factor == 0:
            factor = 1
        self.scale_ = np.array([factor, factor])
        return self

def preparar(ubicacion=True, escalar=True, zona_k=None):
    """Una rama por tipo de columna, como en el laboratorio 4."""
    rama_cat = Pipeline([
        ('imputar', SimpleImputer(strategy='constant', fill_value='sin_dato')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    rama_num = Pipeline([
        ('imputar', SimpleImputer(strategy='median')),
        ('escalar', StandardScaler())
    ])
    ramas = [('tipo', rama_cat, CAT), ('calendario', rama_num, NUM)]
    if ubicacion:
        pasos = []
        if escalar:
            pasos.append(('escalar', EscalaEspacial()))
        if zona_k is not None:
            pasos.append(('zonas', KMeans(n_clusters=zona_k, n_init=10, random_state=SEMILLA)))
        if pasos:
            ramas.append(('ubicacion', Pipeline(pasos), GEO))
        else:
            ramas.append(('ubicacion', 'passthrough', GEO))
    return ColumnTransformer(ramas)

def flujo(modelo, ubicacion=True, escalar=True, zona_k=None):
    return Pipeline([
        ('preparar', preparar(ubicacion, escalar, zona_k)),
        ('modelo', modelo)
    ])
BASE = {}
p, tf, tp = reloj_fit_predict(DummyClassifier(strategy='prior'), X.iloc[it], y[it], X.iloc[iv])
BASE['Prevalencia'] = metricas(y[iv], p)
log_sin = flujo(LogisticRegression(max_iter=1500), ubicacion=False)
p, tf, tp = reloj_fit_predict(log_sin, X.iloc[it], y[it], X.iloc[iv])
BASE['Logística sin ubicación'] = metricas(y[iv], p)
guardar('lineas_base', pd.DataFrame([{'modelo': k, **v} for k, v in BASE.items()]))
RES['base'] = BASE
mostrar(f"La AP trivial es {BASE['Prevalencia']['AP']:.3f}: coincide con prevalencia de **validación**, no con cero. La probabilidad constante empata los riesgos y tiene AUC=0,5. La logística sin ubicación es la referencia predictiva posterior.")
''')

md(8, r'''
Se ajusta solo en 2017 y se evalúa en 2018. AP es average precision, no accuracy: para puntuaciones constantes su referencia es la prevalencia del conjunto evaluado. No se necesita un umbral ni se elige uno por costos. Las tablas ejecutadas reemplazan cifras hipotéticas y toda comparación posterior informa ambas referencias.


**Relación con los labs:** preparación y pipeline del lab 4; probabilidades/logística del lab 5; AUC/AP del lab 8. La proyección fija proviene de la consigna. `EscalaEspacial` solo adapta StandardScaler: conserva la media por eje, pero usa el mismo divisor para x e y, porque la consigna prohíbe deformar las distancias. No implementa otro algoritmo de aprendizaje.
''')

code(10, r'''
filas = []
modelos = {
    'KNN': KNeighborsClassifier(n_neighbors=31, n_jobs=2),
    'Árbol': DecisionTreeClassifier(max_depth=8, min_samples_leaf=30, random_state=SEMILLA)
}
variantes = [
    ('Unidad', 'Grados', False),
    ('Unidad', 'Metros', False),
    ('Escala', 'Metros estandarizados', True),
    ('Escala', 'Metros sin escalar', False)
]
for nombre, modelo in modelos.items():
    for par, variante, escalar in variantes:
        X_variante = X.copy()
        if variante == 'Grados':
            X_variante[GEO] = muestra[['LONGITUD','LATITUD']].to_numpy()
        actual = flujo(clone(modelo), escalar=escalar)
        p, tiempo_fit, tiempo_pred = reloj_fit_predict(actual, X_variante.iloc[it], y[it], X_variante.iloc[iv])
        filas.append({'par': par, 'variante': variante, 'familia': nombre, **metricas(y[iv], p)})
guardar('experimento_coordenadas', pd.DataFrame(filas))
RES['coordenadas'] = filas
''')

md(11, r'''
**Proyección:** aproximación local de la consigna, radio 6.371.000 m y referencia fija (-56,17°, -34,87°). Las fórmulas usan radianes. Es una aproximación local, no un levantamiento geodésico.

En cada par se cambia una sola cosa: grados/metros, o el factor de escala espacial. Tipo, calendario, hiperparámetros y partición permanecen iguales. El factor espacial común se aprende solo en entrenamiento y preserva la geometría; dividir cada eje por su propio desvío la deformaría.

KNN combina las entradas en una distancia y puede cambiar sus vecinos. El árbol puede mover umbrales sin cambiar grupos ante la transformación afín positiva de cada eje que usamos. Se leen los cambios medidos; no se presupone que toda métrica tenga que moverse o mejorar.
''')

code(13, r'''
T = ventana[(ventana.anio < 2019) & (ventana.CODIGO_TIPOPROBLEMA == 401)].drop_duplicates(subset='NUMERO_RECLAMO').copy()
T = T[T.LATITUD.between(-35.1, -34.5) & T.LONGITUD.between(-56.6, -55.9)]
assert len(T) >= 2000
print('Tipo 401:', T.DESC_TIPOPROBLEMA.unique(), 'reclamos:', len(T))
P = proyectar(T.LONGITUD, T.LATITUD)
KS = [2, 3, 4, 6, 8, 10, 12]
curva = []
ajustes = {}
for k in KS:
    km = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit(P)
    s = silhouette_score(P, km.labels_, sample_size=min(2500, len(P)), random_state=SEMILLA)
    curva.append({'K': k, 'silhouette': s})
    ajustes[k] = km
curva = pd.DataFrame(curva)
guardar('curva_kmeans', curva)
plt.plot(curva.K, curva.silhouette, 'o-')
plt.xlabel('K')
plt.ylabel('Silhouette')
plt.show()
K_FOCOS = int(curva.loc[curva.silhouette.idxmax(), 'K'])
distancias = {}
fig, ax = plt.subplots()
for minimo in [5, 10, 20]:
    d = NearestNeighbors(n_neighbors=minimo).fit(P).kneighbors(P)[0][:, -1]
    distancias[minimo] = np.sort(d)
    ax.plot(np.linspace(0, 1, len(d)), np.sort(d), label=f'min_samples={minimo}')
ax.set_ylim(0, np.quantile(distancias[10], 0.995) * 1.2)
ax.set_xlabel('Fracción ordenada')
ax.set_ylabel('k-distancia (m)')
ax.legend()
plt.show()
MIN_VECINOS = 10
EPS = 550
d = distancias[MIN_VECINOS]
print('Se elige eps=550 m: zona de cambio de pendiente alrededor de 500–600 m; mínimo 10 vecinos.')
plt.plot(d)
plt.axhline(EPS, color='red', label='eps=550 m')
plt.ylim(0, np.quantile(d, 0.995) * 1.2)
plt.legend()
plt.show()
Z_ward = linkage(P, method='ward')
Z_simple = linkage(P, method='single')
labels = {'K-medias': ajustes[K_FOCOS].labels_, 'Jerárquico Ward': fcluster(Z_ward, K_FOCOS, criterion='maxclust'), 'Jerárquico simple': fcluster(Z_simple, K_FOCOS, criterion='maxclust'), 'DBSCAN': DBSCAN(eps=EPS, min_samples=MIN_VECINOS, n_jobs=2).fit_predict(P)}
filas = []
fig, axes = plt.subplots(2, 2, figsize=(12, 9))
for (nombre, lab), ax in zip(labels.items(), axes.ravel()):
    ok = lab != -1
    grupos = len(set(lab[ok]))
    sil = np.nan
    if grupos >= 2 and ok.sum() > grupos:
        sil = silhouette_score(P[ok], lab[ok], sample_size=min(2500, int(ok.sum())), random_state=SEMILLA)
    filas.append({'metodo': nombre, 'grupos': grupos, 'sin_grupo': int((~ok).sum()), 'silhouette': sil, 'cobertura': float(ok.mean())})
    ax.scatter(P[:, 0], P[:, 1], c=lab, cmap='tab20', s=5, alpha=0.7)
    ax.set_title(nombre)
    ax.set_aspect('equal')
    ax.set_xlabel('x local (m)')
    ax.set_ylabel('y local (m)')
plt.tight_layout()
plt.show()
guardar('comparacion_clusters', pd.DataFrame(filas))
lab = labels['DBSCAN']
f = []
for grupo in sorted(set(lab) - {-1}):
    idx = lab == grupo
    centro = P[idx].mean(axis=0)
    f.append({'foco': int(grupo), 'longitud': float(T.LONGITUD.to_numpy()[idx].mean()), 'latitud': float(T.LATITUD.to_numpy()[idx].mean()), 'reclamos': int(idx.sum()), 'radio_p90_m': float(np.quantile(np.linalg.norm(P[idx] - centro, axis=1), 0.9)), 'criterio': f'Densidad: eps {EPS:.0f} m, min_samples {MIN_VECINOS}'})
focos = pd.DataFrame(f).sort_values('reclamos', ascending=False) if f else pd.DataFrame(columns=['foco', 'reclamos', 'radio_p90_m'])
guardar('grupos_dbscan', focos)
recomendados = focos[(focos.reclamos >= 30) & (focos.radio_p90_m <= 2500)].head(5)
guardar('focos_recomendados', recomendados)
RES['focos'] = {'tipo': 'Bache (401)', 'reclamos': len(T), 'K': K_FOCOS, 'eps': EPS, 'min_samples': MIN_VECINOS, 'comparacion': filas, 'recomendados': recomendados.to_dict('records')}
mostrar(f'K con mayor silhouette observado: {K_FOCOS}. DBSCAN identifica {len(focos)} grupos. Ruido excluido de silhouette y reportado como sin grupo: no se considera una sola agrupación de puntos dispersos. Si hay menos de dos grupos, silhouette no está definido.')
''')

md(14, r'''
**Elección:** baches (tipo 401), problema localizado y con más de 2.000 ingresos de desarrollo. Se incluyen estados no finales para medir concentración de reportes, no solo de cierres. La tabla da las ubicaciones aproximadas de hasta cinco focos compactos con más soporte y su radio p90.

**K-medias:** se elige K por la curva completa; un máximo en la grilla no prueba un número natural de barrios, especialmente si la curva es plana o cae desde K=2. Se conserva ese K en Ward y enlace simple para comparar enlaces. Ward favorece compacidad; simple puede encadenar puntos por puentes.

**DBSCAN:** se comparan k-distancias de 5, 10 y 20 vecinos, se fija 10 como soporte intermedio y se elige visualmente eps=550 m antes de agrupar, cerca del cambio de pendiente de la curva, como en el lab 21. Se descarta la detección automática de rodilla para que el criterio sea directo y defendible. Es una aproximación, no un radio municipal oficial. Se informa cobertura junto con silhouette: excluir mucho ruido puede mejorar esa métrica sin representar toda la ciudad.

K-medias reparte todos los puntos en zonas; DBSCAN busca densidad y puede dejar ruido, pero también puede encadenar gran parte de la ciudad en un grupo. El mejor silhouette no elige automáticamente la mejor intervención. Para recomendar focos se exige al menos 30 reportes y radio p90 de hasta 2,5 km: criterio operativo propuesto, no norma oficial ni parámetro ajustado sobre 2019. Se excluyen grupos demasiado extensos de la recomendación y se muestran todos en la tabla. Sus centros son coordenadas aproximadas, no barrios identificados por nombre. Los reportes pueden ser reiteraciones: no cuentan baches físicos únicos ni incidencia por población.


**Relación con el lab 21:** KMeans, curva de silhouette, NearestNeighbors y DBSCAN; jerárquico mediante `linkage` y `fcluster`. Se emplean las funciones de biblioteca que se vieron, no un algoritmo jerárquico propio.
''')

code(16, r'''
log_coord = flujo(LogisticRegression(max_iter=1500))
p, tiempo_fit, tiempo_pred = reloj_fit_predict(log_coord, X.iloc[it], y[it], X.iloc[iv])
comparacion = [{'modelo': 'Logística coordenadas', **metricas(y[iv],p)}]
filas_zonas = []
for k in [4,8,12]:
    modelo = flujo(LogisticRegression(max_iter=1500), zona_k=k)
    p, tiempo_fit, tiempo_pred = reloj_fit_predict(modelo, X.iloc[it], y[it], X.iloc[iv])
    filas_zonas.append({'K': k, **metricas(y[iv],p)})
tabla_zonas = pd.DataFrame(filas_zonas)
guardar('grilla_zonas', tabla_zonas)
mejor = tabla_zonas.loc[tabla_zonas.AP.idxmax()]
K_ZONAS = int(mejor.K)
comparacion.append({'modelo': f'Logística distancias a {K_ZONAS} centros', 'AUC': mejor.AUC, 'AP': mejor.AP})
guardar('logistica_zonas', pd.DataFrame(comparacion))
RES['zonas'] = {'K': K_ZONAS, 'comparacion': comparacion}
''')

md(17, r'''
Se eligen distancias a centroides para no imponer orden a etiquetas de zona. K se selecciona por AP temporal entre 4, 8 y 12, sobre todos los tipos. El K de focos de baches no se supone óptimo para predicción general. La logística con x/y tiene un efecto espacial lineal en log-odds; las distancias agregan una representación no lineal. La mejora se mide, no se garantiza.

KMeans está dentro del pipeline y aprende solo de entrenamiento. Ajustarlo antes con toda la muestra revelaría distribución de ubicaciones de validación. Una representación que use la respuesta (por ejemplo, tasa de cierre por zona) requeriría construcción fuera de muestra dentro de cada partición para evitar fuga de etiquetas. Aquí no se usa la respuesta para agrupar. Si se usaran etiquetas de zona, se codificarían como categorías, no como números ordenados.
''')

code(19, r'''
# Grillas pequeñas y bucle explícito, como buscar_grilla del lab 17.
modelos = {
    'KNN': KNeighborsClassifier(n_jobs=2),
    'Árbol': DecisionTreeClassifier(random_state=SEMILLA),
    'SVM RBF': SVC(kernel='rbf', cache_size=800),
    'Bosque': RandomForestClassifier(n_estimators=120, max_features='sqrt', n_jobs=2, random_state=SEMILLA),
    'Boosting': AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1, random_state=SEMILLA), random_state=SEMILLA),
    'Red neuronal': MLPClassifier(max_iter=120, early_stopping=True, n_iter_no_change=10, random_state=SEMILLA)
}
grillas = {
    'KNN': [{'n_neighbors': k} for k in [15,51,151]],
    'Árbol': [{'max_depth': profundidad, 'min_samples_leaf': minimo} for profundidad in [6,12] for minimo in [20,60]],
    'SVM RBF': [{'C': c, 'gamma': gamma} for c in [.5,2.] for gamma in ['scale',.03]],
    'Bosque': [{'max_depth': profundidad, 'min_samples_leaf': minimo} for profundidad in [10,None] for minimo in [5,20]],
    'Boosting': [{'n_estimators': cantidad, 'learning_rate': tasa} for cantidad in [50,100] for tasa in [.5,1.]],
    'Red neuronal': [{'hidden_layer_sizes': capa, 'alpha': alpha} for capa in [(32,),(64,)] for alpha in [.001,.1]]
}
ELEGIDOS = {}
VALIDACION = []
todas_configuraciones = []
for nombre in modelos:
    if nombre == 'SVM RBF':
        xx, yy = Xs, ys
        a, b = CVS[0]
    else:
        xx, yy = X, y
        a, b = it, iv
    print('Buscando', nombre, flush=True)
    inicio = time.perf_counter()
    mejor_ap = -1
    for parametros in grillas[nombre]:
        modelo = clone(modelos[nombre]).set_params(**parametros)
        actual = flujo(modelo)
        p, tiempo_fit, tiempo_pred = reloj_fit_predict(actual, xx.iloc[a], yy[a], xx.iloc[b])
        resultado = metricas(yy[b], p)
        todas_configuraciones.append({'familia': nombre, 'parametros': str(parametros), **resultado})
        if resultado['AP'] > mejor_ap:
            mejor_ap = resultado['AP']
            ELEGIDOS[nombre] = clone(actual)
            mejor_fila = {'familia': nombre, 'n': len(xx), 'configuracion': str(parametros),
                'grilla': str(grillas[nombre]), **resultado, 'ajuste_s': tiempo_fit, 'prediccion_s': tiempo_pred}
            mejores_parametros = parametros
    bordes = []
    for parametro in mejores_parametros:
        valores = []
        for configuracion in grillas[nombre]:
            valor = configuracion[parametro]
            if valor not in valores:
                valores.append(valor)
        if mejores_parametros[parametro] in [valores[0], valores[-1]]:
            bordes.append(parametro)
    mejor_fila['bordes'] = ', '.join(bordes)
    mejor_fila['busqueda_s'] = time.perf_counter() - inicio
    VALIDACION.append(mejor_fila)
    print(mejor_fila, flush=True)
    guardar('familias_validacion', pd.DataFrame(VALIDACION))
guardar('grillas_familias', pd.DataFrame(todas_configuraciones))
RES['familias_validacion'] = VALIDACION
tabla_validacion = pd.DataFrame(VALIDACION)
max_ap = tabla_validacion.AP.max()
candidatos = tabla_validacion[tabla_validacion.AP >= max_ap-.01].copy()
candidatos['tiempo_total'] = candidatos.ajuste_s + candidatos.prediccion_s
RECOMENDADO = candidatos.sort_values('tiempo_total').iloc[0].familia
print('Recomendación fijada en desarrollo:', RECOMENDADO)
''')

md(20, r'''
AP temporal elige hiperparámetros, AUC se reporta también. Se conserva proporción de clases. El bucle de grilla es explícito, como en lab 17: ajustar en 2017, predecir en 2018, calcular AP y conservar la mejor configuración. Boosting usa AdaBoost con árboles de profundidad 1, como en lab 16. SVM usa 10.000 de los 50.000 y sus scores de decisión sirven para AUC/AP, no son probabilidades calibradas.

La preparación es común: categorías one-hot, calendario estandarizado y coordenadas proyectadas con factor común. No se mezcla el experimento de zonas con la comparación de familias. Árboles no necesitan el escalado, pero mantenerlo no altera sus particiones por eje.

Grillas pequeñas: con dos valores ambos son bordes; se informa la limitación, no se afirma óptimo global. Se distinguen tiempo de búsqueda, ajuste de una configuración y predicción. La red usa early stopping sobre una fracción interna de entrenamiento, no sobre 2018 ni 2019. Se revisan avisos de convergencia.

La recomendación se fija en desarrollo: tolerancia operativa de 0,01 AP frente al máximo y menor tiempo de ajuste+predicción entre candidatos cercanos. No es una prueba estadística de equivalencia. Se discute además lo que permite explicar la familia; no se cambia el criterio al ver el test.


**Alcance del material local:** SVM y barrido de grillas aparecen en lab 17; AdaBoost en lab 16. Los archivos consultados no muestran un lab de bosque o red neuronal, pero la consigna exige ambas familias: se usan sus estimadores estándar con grillas pequeñas, sin implementaciones desde cero.
''')

code(22, r'''
X19 = prueba[FEATURES]
y19 = prueba.resuelto_7d.to_numpy()
FINAL = []
AJUSTADOS = {}
for nombre, modelo in [('Logística sin ubicación', clone(log_sin))] + list(ELEGIDOS.items()):
    xx, yy = (Xs, ys) if nombre == 'SVM RBF' else (X, y)
    mod = clone(modelo)
    p, tf, tp = reloj_fit_predict(mod, xx, yy, X19)
    AJUSTADOS[nombre] = mod
    FINAL.append({'familia': nombre, 'n_entrenamiento': len(xx), 'n_prueba': len(X19), **metricas(y19, p), 'ajuste_s': tf, 'prediccion_s': tp})
    print(nombre, FINAL[-1], flush=True)
guardar('familias_2019', pd.DataFrame(FINAL))
aleatorias = []
for a, b in StratifiedKFold(n_splits=5, shuffle=True, random_state=SEMILLA).split(X, y):
    mod = clone(ELEGIDOS['Bosque'])
    p, _, _ = reloj_fit_predict(mod, X.iloc[a], y[a], X.iloc[b])
    aleatorias.append(metricas(y[b], p))
rf_val = next((f for f in VALIDACION if f['familia'] == 'Bosque'))
rf_test = next((f for f in FINAL if f['familia'] == 'Bosque'))
ale = pd.DataFrame(aleatorias)
brechas = pd.DataFrame([{'esquema': 'Temporal 2017→2018', 'AUC': rf_val['AUC'], 'AP': rf_val['AP']}, {'esquema': '5 folds mezclados', 'AUC': ale.AUC.mean(), 'AP': ale.AP.mean()}, {'esquema': '2017–2018 → 2019', 'AUC': rf_test['AUC'], 'AP': rf_test['AP']}])
brechas['brecha_AUC_vs_2019'] = brechas.AUC - rf_test['AUC']
brechas['brecha_AP_vs_2019'] = brechas.AP - rf_test['AP']
guardar('validacion_bosque', brechas)
mostrar(f'**Recomendado en desarrollo:** {RECOMENDADO}. Se comprueba sobre 2019, sin abrir una nueva búsqueda.')
RES['final'] = FINAL
RES['brechas_bosque'] = brechas.to_dict('records')
RES['recomendado'] = RECOMENDADO
''')

md(23, r'''
La validación mezclada deja reclamos de ambos años en entrenamiento y validación: pueden compartir ubicaciones, tipos vigentes y episodios cercanos. La temporal exige pasar a otro año. La tabla cuantifica signo y tamaño de la brecha, sin presuponer que sea positiva.

También cambia el tamaño de entrenamiento: CV mezclada usa 80% por fold; la frontera anual solo la parte de 2017. La brecha no se atribuye exclusivamente a fuga: intervienen tamaño, prevalencia y cambio temporal. Son los mismos 50.000 casos, configuración y preparación, con distintos esquemas; no una prueba causal.

Los tiempos finales predicen toda la prueba. SVM entrena con menos datos por exigencia de la consigna. Árboles permiten revisar reglas e importancias; logística coeficientes; SVM y red son menos directas para explicar una decisión individual. Unas milésimas no prueban que una familia sea mejor. La recomendación se decidió en la sección 6; test es la comprobación final.
''')

code(25, r'''
MEJOR_ARBOL = max((f for f in VALIDACION if f['familia'] in ['Árbol', 'Bosque', 'Boosting']), key=lambda f: f['AP'])['familia']
mod = AJUSTADOS[MEJOR_ARBOL]
prep = mod.named_steps['preparar']
est = mod.named_steps['modelo']
nombres = prep.get_feature_names_out()
agregadas = {c: 0.0 for c in FEATURES}
for nombre, valor in zip(nombres, est.feature_importances_):
    if nombre.startswith('tipo__'):
        for col in CAT:
            if nombre.startswith('tipo__' + col + '_'):
                agregadas[col] += float(valor)
                break
    elif nombre.startswith('calendario__'):
        agregadas[nombre.split('__', 1)[1]] += float(valor)
    elif nombre == 'ubicacion__x':
        agregadas['x'] += float(valor)
    elif nombre == 'ubicacion__y':
        agregadas['y'] += float(valor)
ids = train_test_split(np.arange(len(X19)), train_size=min(10000, len(X19) - 1), stratify=y19, random_state=SEMILLA)[0]
per = permutation_importance(mod, X19.iloc[ids], y19[ids], scoring='average_precision', n_repeats=5, random_state=SEMILLA, n_jobs=1)
tabla_imp = pd.DataFrame({'variable': FEATURES, 'impureza': [agregadas[c] for c in FEATURES], 'permutacion_caida_AP': per.importances_mean, 'permutacion_desvio': per.importances_std}).sort_values('permutacion_caida_AP', ascending=False)
guardar('importancias', tabla_imp)
print('Modelo:', MEJOR_ARBOL, 'permutación sobre', len(ids), 'reclamos de 2019')
RES['importancias'] = {'modelo': MEJOR_ARBOL, 'n': len(ids), 'tabla': tabla_imp.to_dict('records')}
rng = np.random.default_rng(SEMILLA)
xx = X19.iloc[ids].copy()
yy = y19[ids]
ap0 = average_precision_score(yy, mod.predict_proba(xx)[:, 1])
caidas = []
for _ in range(5):
    xp = xx.copy()
    xp[GEO] = xx[GEO].to_numpy()[rng.permutation(len(xx))]
    caidas.append(ap0 - average_precision_score(yy, mod.predict_proba(xp)[:, 1]))
RES['importancias']['permutacion_ubicacion_conjunta_AP'] = float(np.mean(caidas))
mostrar(f'Permutar ubicación como bloque reduce AP en {np.mean(caidas):.4f} ± {np.std(caidas, ddof=1):.4f}. No se suman sin más latitud y longitud.')
(SALIDA / 'resumen.json').write_text(json.dumps(RES, ensure_ascii=False, indent=2, default=lambda v: float(v)), encoding='utf-8')
''')

md(26, r'''
**Importancias:** impureza describe reducciones del criterio interno durante entrenamiento, y puede favorecer variables continuas o con muchas alternativas de corte. Permutación mide caída de AP al romper entradas en 2019: su utilidad predictiva fuera de muestra. Ninguna prueba causas. Se comparan rangos y lectura, no los números como porcentajes equivalentes: las unidades son distintas.

Las dummies se agregan por columna original para compararlas con permutar esa columna. Área/grupo/tipo se relacionan: permutar una puede crear combinaciones imposibles y otra sustituir su información. Se añade permutación conjunta de ubicación para preservar los pares de coordenadas entre sí. Importancias bajas pueden reflejar redundancia.

**Recomendación:** el modelo seleccionado con desarrollo se compara con logística sin ubicación en 2019. Los focos concretos de baches están en sección 4. La tabla de importancias y la permutación conjunta muestran qué usa el modelo y cuánto aporta la ubicación en este conjunto, no qué causa la demora.

**Límites del caso:**

1. Finalizado es cierre de gestión, no reparación física confirmada. La fecha corresponde al estado actual; no reconstruye resolución inicial ni reaperturas.
2. Excluir abiertos/anulados condiciona a un estado futuro. Las métricas no garantizan rendimiento sobre todos los ingresados.
3. La foto es posterior a los años estudiados. Etiquetas tardías pueden no haber estado disponibles en la frontera histórica. Se requiere historial y cortes de disponibilidad para simular estrictamente el despliegue.
4. Cambian prevalencia, catálogo y operación. Una frontera 2017→2018 y test 2019 no garantizan estabilidad después de pandemia ni en 2026.
5. Reclamos cercanos pueden ser reiteraciones. Focos de reportes no equivalen a defectos únicos ni a incidencia por habitante.
6. La proyección es local y el rango usa un rectángulo, no límites administrativos. No se inventan nombres de barrios para los centros.
7. Las grillas son pequeñas y SVM tiene menos entrenamiento. No se pide un desempeño mínimo ni pruebas estadísticas formales.
8. **Ninguna conclusión constituye un diagnóstico sobre la gestión de la Intendencia de Montevideo.** Las asociaciones predictivas no establecen causalidad.
''')

for i,s in C.items():
    nb['cells'][i]['source']=s.splitlines(keepends=True)
    nb['cells'][i]['outputs']=[];nb['cells'][i]['execution_count']=None
for i,s in M.items():nb['cells'][i]['source']=s.splitlines(keepends=True)
s=''.join(nb['cells'][0]['source']).replace('1.\n2.\n3.\n4.','1. Marcos Devincenzi\n2. Santiago Alba\n3. Facundo Díaz\n4. Domingo Leiva')
nb['cells'][0]['source']=s.splitlines(keepends=True)
nb['metadata']['kernelspec']={'display_name':'Python 3','language':'python','name':'python3'}
(root/'entrega_reclamos_im.ipynb').write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8',newline='\n')
print('Notebook preparado:',len(nb['cells']),'celdas')
