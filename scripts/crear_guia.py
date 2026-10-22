"""Genera guía Markdown y PDF a partir de los resultados reales del notebook."""
import json, re, html
from pathlib import Path
root=Path(__file__).resolve().parents[1]
r=json.loads((root/'resultados/resumen.json').read_text(encoding='utf-8'))
def num(v,n=3):return f'{v:.{n}f}'.replace('.',',')
def tabla(filas,columnas):
    return '\n'.join(['| '+' | '.join(t for k,t in columnas)+' |','| '+' | '.join('---' for _ in columnas)+' |']+
        ['| '+' | '.join(num(f[k]) if isinstance(f.get(k),float) else str(f.get(k,'')) for k,t in columnas)+' |' for f in filas])
d=r['datos'];elegido=r['recomendado']
v=next(f for f in r['familias_validacion'] if f['familia']==elegido)
t=next(f for f in r['final'] if f['familia']==elegido)
b=next(f for f in r['final'] if f['familia']=='Logística sin ubicación')
guia=f'''# Guía para defender el proyecto de reclamos de Montevideo

Esta guía acompaña al notebook ejecutado. Explica qué se hizo, por qué y qué números lo respaldan. Las cifras de validación son de 2018, con entrenamiento en 2017; las de prueba son de 2019. No son intercambiables. Los cuatro integrantes deben poder explicar cualquier sección y reconocer los límites del trabajo.

## Presentación de un minuto

> Queremos estimar al ingresar un reclamo si su cierre se registra dentro de siete días y encontrar concentraciones de baches. Auditamos los datos, construimos la respuesta y usamos únicamente información conocida al ingreso. Elegimos con 2017–2018 y reservamos 2019. La recomendación entre las seis familias es {elegido}: en prueba obtiene AUC {num(t['AUC'])} y AP {num(t['AP'])}, frente a {num(b['AUC'])} y {num(b['AP'])} de la logística sin ubicación. Analizamos también agrupamientos: una zona que divide la ciudad no equivale a un foco denso. El cierre publicado no garantiza reparación física, y los resultados no son un diagnóstico de la gestión municipal.

## Cómo leer el código

**`flujo(modelo)`** arma dos pasos juntos: preparación y algoritmo. Por ejemplo, primero convierte categorías y coordenadas, y después ajusta el bosque. Se llama pipeline porque aplica los pasos en orden.

**`ColumnTransformer`** reparte columnas: códigos de problema van a one-hot; calendario a escalado; ubicación a proyección y escalado espacial. One-hot convierte categorías en columnas 0/1, sin suponer que un código 400 sea el doble de 200.

**`Coordenadas`** convierte latitud/longitud a metros locales y, si corresponde, divide ambos ejes por el mismo factor aprendido del entrenamiento. No deforma la geometría usando un desvío diferente para cada eje.

**`GridSearchCV`** prueba combinaciones declaradas y selecciona por AP sobre la partición temporal. El pipeline completo se vuelve a ajustar en cada candidato. Se clonan los estimadores al comparar: las métricas y los modelos finales no dependen del estado dejado por el último fold.

## 1. Datos, respuesta y población

Leímos **{d['original']:,} filas**. **{d['ventana']:,}** corresponden a ingresos de 2017–2019. Quitamos **{d['copias']} copias exactas**, excluimos **{d['no_final']:,} abiertos o anulados** y **{d['demoras_invalidas']} demora imposible/ausente**. No hubo exclusiones geográficas por el rectángulo local. La población final es de **{d['poblacion']:,} reclamos**, de los cuales **{d['prueba']:,}** son de prueba.

**La etiqueta:** calculamos días entre ingreso y fecha de entrada al estado actual, solo si ese estado es Finalizado. Positivo significa demora no negativa y menor o igual a siete días: exactamente siete cuenta como positivo. El negativo es un finalizado que demoró más de siete días.

**Por qué no usar todos los estados:** la fecha de un reclamo En Proceso es la de entrada a ese estado, no la de resolución. Anulado tampoco significa necesariamente no resuelto. Asignarles una etiqueta automática inventaría un resultado. Excluirlos hace observable la etiqueta, pero introduce selección: el modelo representa finalizados con fechas válidas, no todos los ingresos.

**Límite de la palabra resuelto:** el catálogo dice que Finalizado incluye cierres por no corresponder y otros motivos. Por eso nuestra respuesta es un indicador de cierre administrativo semanal, no prueba de reparación física. También puede haber reaperturas que la foto actual no reconstruye.

**Contraste con metadatos:** la descripción menciona COORD_X/COORD_Y UTM, pero el archivo trae LATITUD/LONGITUD en grados. Se usa lo observado y se documenta la discrepancia. No se inventa que el archivo ya esté en metros.

**Entradas:** códigos de área, grupo y tipo como categorías; calendario derivado de la fecha de ingreso; latitud y longitud. No entran identificador, estado actual ni fecha de cambio de estado. Los dos últimos se conocen después y producirían fuga. Los nombres de área/grupo/tipo se descartan porque duplican sus códigos.

**Validación:** entrenamos en 2017 y validamos en 2018. La alternativa aleatoria mezcla pasado y futuro relativo, y puede compartir episodios y tipos vigentes. El uso previsto recibe nuevos reclamos, por eso elegimos la frontera temporal. Es una sola partición anual, no cinco folds ni un intervalo de confianza. La muestra de familias tiene 50.000 casos proporcionalmente por año; SVM usa 10.000 de esos mismos casos.

**Advertencia temporal concreta:** conocemos las etiquetas desde una foto posterior. Algunos cierres de reclamos de 2017 ocurrieron después de 2018. El ejercicio evalúa transferencia entre años de ingreso; para simular estrictamente lo disponible en una fecha habría que reconstruir historial y cortar también las etiquetas de entrenamiento. Este límite se declara y no se usa esa fecha como entrada.

## 2. Líneas base y métricas

Una línea base es una referencia simple que el modelo tiene que superar. La trivial asigna la misma probabilidad a todos, la prevalencia del entrenamiento. La otra es logística con tipo y calendario, sin ubicación.

{tabla([dict(modelo=k,**x) for k,x in r['base'].items()],[('modelo','Modelo'),('AUC','AUC en 2018'),('AP','AP en 2018')])}

**AUC:** capacidad de ordenar positivos por encima de negativos. Un AUC de 0,70 no significa 70% de aciertos. Con todos los scores iguales es 0,5.

**AP:** average precision resume precisión y recall al recorrer umbrales. Su referencia trivial es la proporción de positivos del conjunto evaluado, no cero. AP cambia también cuando cambia la prevalencia: no se compara un año con otro ignorando ese cambio.

No elegimos umbral por costos: esta entrega pide scores o probabilidades y métricas de ordenamiento. La SVM usa función de decisión; no se presenta su score como una probabilidad calibrada.

## 3. Coordenadas: unidades y escalado

Cinco kilómetros y cinco mil metros son el mismo lugar, pero cambian la magnitud numérica de las entradas. KNN busca reclamos cercanos en el espacio de atributos; sin escalado una coordenada en miles puede dominar a las categorías 0/1. El árbol pregunta si una coordenada supera un umbral y puede convertir ese umbral sin cambiar las filas separadas.

Usamos la proyección local permitida: x = R por diferencia de longitud en radianes por coseno de la latitud de referencia; y = R por diferencia de latitud. R vale 6.371.000 m y la referencia fija es longitud -56,17°, latitud -34,87°. No se aprende del test.

{tabla(r['coordenadas'],[('par','Contraste'),('variante','Variante'),('familia','Familia'),('AUC','AUC'),('AP','AP')])}

En cada contraste solo cambia unidad o escalado espacial. Los modelos, parámetros, resto de preparación y partición permanecen iguales. Un factor común para ambos ejes conserva proporciones físicas. Se interpretan las diferencias observadas, no se exige que todos los modelos mejoren.

## 4. Agrupamientos y focos

Elegimos **baches, tipo 401**, con **{r['focos']['reclamos']} ingresos en 2017–2018**. Tiene ubicación interpretable y supera los 2.000 pedidos. Aquí también contamos abiertos/anulados porque buscamos concentración de reportes, no solo de finalizados. No se usan ubicaciones de 2019.

**K-medias:** fuerza K grupos y asigna todos los puntos. Probamos una curva de K entre 2 y 12; el máximo medido fue **K={r['focos']['K']}**. La curva y su forma importan: el mejor valor de una grilla no prueba que existan exactamente esos barrios naturales.

**Jerárquico Ward:** agrupa favoreciendo compacidad. **Enlace simple:** puede conectar zonas mediante cadenas de puntos, aunque estén muy extendidas. Se comparan con el mismo K para no cambiar dos cosas a la vez.

**DBSCAN:** busca densidad y puede dejar reclamos sin grupo. Primero se graficaron k-distancias de 5, 10 y 20 vecinos. Elegimos mínimo 10 como soporte intermedio y una rodilla de la curva como radio: **eps={num(r['focos']['eps'],0)} m**. La rodilla es una regla aproximada, no una norma de la Intendencia.

{tabla(r['focos']['comparacion'],[('metodo','Método'),('grupos','Grupos'),('sin_grupo','Sin grupo'),('silhouette','Silhouette'),('cobertura','Cobertura')])}

**Silhouette** evalúa cohesión y separación geométrica; alto no significa automáticamente utilidad para intervención. En DBSCAN se calcula sin ruido y se informa la cobertura. No se trata a todos los puntos dispersos como un solo grupo. El valor negativo observado obliga a reconocer superposición o encadenamiento y no a declarar éxito por el nombre del método.

**Qué recomendamos:** de los grupos DBSCAN, revisar los que tengan al menos 30 reportes y radio p90 no mayor a 2,5 km. Es un criterio operativo propuesto que excluye grupos demasiado extensos para intervención conjunta. p90 es el radio desde su centro aproximado que contiene al 90% de los puntos. El personal municipal debería confirmar la conveniencia y si hay reclamos reiterados del mismo bache.

{tabla(r['focos']['recomendados'],[('foco','Foco'),('longitud','Longitud aprox.'),('latitud','Latitud aprox.'),('reclamos','Reportes'),('radio_p90_m','Radio p90 en m')])}

Los centros son coordenadas, no nombres de barrios verificados. Contamos reportes, no defectos físicos únicos ni tasas por habitante.

## 5. Zonas como atributos

Una logística con x e y aprende un efecto espacial lineal en log-odds. Distancias a varios centros permiten describir un patrón más flexible sin cambiar el clasificador. Usamos KMeans dentro del pipeline: aprende centros solo en entrenamiento.

Elegimos **K={r['zonas']['K']}** entre 4, 8 y 12 por AP temporal. Este K se elige para predecir todos los tipos; no se copia el de focos de baches.

{tabla(r['zonas']['comparacion'],[('modelo','Representación'),('AUC','AUC'),('AP','AP')])}

Ajustar centros antes con toda la muestra revela distribución de ubicaciones de validación, aunque no use etiquetas. Usar tasas de cierre por zona revelaría además respuesta si no se construye fuera de muestra dentro de cada entrenamiento. Nuestra representación no usa la respuesta para agrupar. Si fueran etiquetas de zona, habría que codificarlas como categorías, no números ordenados.

## 6. Seis familias y elección

- **KNN:** estima por semejanza con vecinos; es sensible a escala y puede costar al predecir muchos casos.
- **Árbol:** reglas por cortes; fácil de inspeccionar, pero puede sobreajustar.
- **SVM RBF:** frontera no lineal con distancias; C regula penalización y gamma el alcance del núcleo.
- **Bosque:** promedia muchos árboles con aleatoriedad para reducir variación; no equivale a un único árbol legible.
- **Boosting:** añade árboles para corregir errores del conjunto anterior; profundidad y cantidad controlan flexibilidad.
- **Red neuronal:** combina capas no lineales; se limita tamaño y regularización y usa parada temprana interna.

Las grillas son pequeñas, de dos a cuatro valores por parámetro. Cada elegido y sus bordes aparecen en el notebook. Caer en borde reconoce exploración limitada, no certifica un óptimo. La red no usa test para parar. Los tiempos corresponden al equipo de esta ejecución y no son una promesa para cualquier computadora.

{tabla(r['familias_validacion'],[('familia','Familia'),('AUC','AUC 2018'),('AP','AP 2018'),('ajuste_s','Ajuste s'),('prediccion_s','Predicción s')])}

**Criterio fijado antes del test:** entre familias a menos de 0,01 AP del máximo de desarrollo, preferimos menor tiempo de ajuste más predicción. Esa tolerancia es operativa, no una prueba de equivalencia. Resultó **{elegido}**, con AP {num(v['AP'])}. La capacidad explicativa se considera en la recomendación: un ensamble ofrece importancias, pero no una explicación causal ni una regla breve para cada caso.

## 7. Evaluación final y validación mezclada

Con configuración cerrada se entrena en la muestra de ambos años y se evalúa una vez en 2019. SVM mantiene sus 10.000. No cambiamos grillas al ver esta tabla. En prueba, el bosque mejora AUC frente a logística sin ubicación, pero su AP queda ligeramente por debajo. Por eso no se demuestra una mejora uniforme ni se justifica reemplazar automáticamente la alternativa simple y barata: la recomendación de desarrollo es un candidato para evaluar prospectivamente, no una garantía de superioridad.

{tabla(r['final'],[('familia','Familia'),('AUC','AUC 2019'),('AP','AP 2019'),('ajuste_s','Ajuste s'),('prediccion_s','Predicción s')])}

{tabla(r['brechas_bosque'],[('esquema','Esquema bosque'),('AUC','AUC'),('AP','AP'),('brecha_AUC_vs_2019','Diferencia AUC vs. 2019')])}

Con mezcla aleatoria hay reclamos de ambos años en cada lado, posiblemente del mismo entorno, episodio o catálogo. La temporal exige cruzar una frontera de año. También cambia la cantidad de entrenamiento por fold: mezclada usa 80%; temporal solo 2017. Por eso la brecha observada no se atribuye exclusivamente a fuga; influyen tamaño, prevalencia y cambio temporal. La tabla informa el signo real, no el que esperábamos.

## 8. Importancias y límites

El mejor modelo de árboles por AP de desarrollo fue **{r['importancias']['modelo']}**. Impureza describe cómo se ajustaron sus cortes; permutación mide cuánto cae AP al romper una entrada sobre una submuestra fija de **{r['importancias']['n']} casos de 2019**, con cinco repeticiones. No se ajusta otra decisión con esas importancias.

{tabla(r['importancias']['tabla'],[('variable','Variable'),('impureza','Impureza'),('permutacion_caida_AP','Caída AP al permutar'),('permutacion_desvio','Desvío')])}

Impureza favorece variables con más oportunidades de corte. Permutación habla de utilidad predictiva fuera de muestra, pero correlaciones pueden ocultar aportes o crear combinaciones irreales. Área, grupo y tipo están relacionados. Para ubicación también permutamos latitud/longitud como bloque: caída AP **{num(r['importancias']['permutacion_ubicacion_conjunta_AP'],4)}**. No se suman automáticamente las dos importancias individuales. Ninguna importancia demuestra causalidad.

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
'''
(root/'GUIA_DEFENSA.md').write_text(guia,encoding='utf-8')

from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib.pagesizes import A4
fonts=Path('C:/Windows/Fonts')
if fonts.exists():
    for name,file in [('Guia','arial.ttf'),('Guia-Bold','arialbd.ttf')]:pdfmetrics.registerFont(TTFont(name,str(fonts/file)))
    pdfmetrics.registerFontFamily('Guia',normal='Guia',bold='Guia-Bold',italic='Guia',boldItalic='Guia-Bold')
    normal='Guia';bold='Guia-Bold'
else:normal='Helvetica';bold='Helvetica-Bold'
styles=getSampleStyleSheet()
styles.add(ParagraphStyle(name='Texto',fontName=normal,fontSize=10.2,leading=14.5,spaceAfter=7,textColor=colors.HexColor('#233044')))
styles.add(ParagraphStyle(name='Titulo',fontName=bold,fontSize=21,leading=26,spaceAfter=15,textColor=colors.HexColor('#164761')))
styles.add(ParagraphStyle(name='Seccion',fontName=bold,fontSize=13,leading=17,spaceBefore=13,spaceAfter=8,keepWithNext=True,textColor=colors.HexColor('#164761')))
styles.add(ParagraphStyle(name='Celda',parent=styles['Texto'],fontSize=8,leading=10.5,spaceAfter=0))
def fmt(s):
    s=s.replace('→',' a ').replace('–','-').replace('—','-').replace('≤','<=').replace('≥','>=')
    s=html.escape(s);s=re.sub(r'\*\*(.+?)\*\*',r'<b>\1</b>',s);s=re.sub(r'`([^`]+)`',r'\1',s)
    return s
story=[];lines=guia.splitlines();i=0
while i<len(lines):
    line=lines[i].strip()
    if not line:i+=1;continue
    if line.startswith('|'):
        rows=[]
        while i<len(lines) and lines[i].strip().startswith('|'):
            row=[x.strip() for x in lines[i].strip().strip('|').split('|')]
            if not all(re.fullmatch(r'[:\- ]+',x) for x in row):rows.append(row)
            i+=1
        cells=[[Paragraph(fmt(x),styles['Celda']) for x in row] for row in rows]
        n=len(rows[0]);width=A4[0]-96
        widths=[width*.30]+[width*.70/(n-1)]*(n-1) if n>2 else [width/n]*n
        t=Table(cells,colWidths=widths,repeatRows=1,hAlign='LEFT')
        t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#dfedf3')),
         ('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#f4f7f9')]),('VALIGN',(0,0),(-1,-1),'TOP'),
         ('LEFTPADDING',(0,0),(-1,-1),5),('RIGHTPADDING',(0,0),(-1,-1),5),('TOPPADDING',(0,0),(-1,-1),5),('BOTTOMPADDING',(0,0),(-1,-1),5)]))
        story.extend([t,Spacer(1,9)]);continue
    if line.startswith('# '):story.append(Paragraph(fmt(line[2:]),styles['Titulo']))
    elif line.startswith('## '):story.append(Paragraph(fmt(line[3:]),styles['Seccion']))
    elif line.startswith('- '):story.append(Paragraph(fmt(line[2:]),styles['Texto'],bulletText='•'))
    elif line.startswith('> '):story.append(Paragraph(fmt(line[2:]),styles['Texto']))
    else:story.append(Paragraph(fmt(line),styles['Texto']))
    i+=1
def pie(canvas,doc):
    canvas.setFont(normal,8);canvas.setFillColor(colors.HexColor('#65798b'))
    canvas.drawString(48,28,'Reclamos IM · Guía de defensa');canvas.drawRightString(A4[0]-48,28,str(doc.page))
    canvas.setStrokeColor(colors.HexColor('#d5e0e7'));canvas.line(48,41,A4[0]-48,41)
SimpleDocTemplate(str(root/'GUIA_DEFENSA.pdf'),pagesize=A4,leftMargin=48,rightMargin=48,topMargin=43,bottomMargin=54,
 title='Guía de defensa - Reclamos de Montevideo',author='Equipo Proyecto ML Reclamos IM').build(story,onFirstPage=pie,onLaterPages=pie)
print('Guía Markdown y PDF generados con resultados reales')
