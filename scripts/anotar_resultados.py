"""Añade interpretaciones numéricas sin volver a ajustar ni seleccionar modelos."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[1]
r=json.loads((root/'resultados/resumen.json').read_text(encoding='utf-8'))
p=root/'entrega_reclamos_im.ipynb';nb=json.loads(p.read_text(encoding='utf-8'))
d=r['datos'];base=r['base']['Logística sin ubicación'];fin=r['final']
rf=next(x for x in fin if x['familia']=='Bosque');lb=next(x for x in fin if x['familia']=='Logística sin ubicación')
z=r['zonas']['comparacion'];c=r['focos']['comparacion'];db=next(x for x in c if x['metodo']=='DBSCAN')
respuestas={
5:f"**Lectura de los conteos:** {d['ventana']:,} ingresos en ventana; {d['copias']} copias, {d['no_final']:,} no finalizados y {d['demoras_invalidas']} demora inválida excluidos. Quedan {d['poblacion']:,}, con {d['prueba']:,} de prueba. El cierre no equivale necesariamente a reparación física.",
11:"**Resultado medido:** KNN pasa de AUC 0,676 en grados sin escalar a 0,580 en metros sin escalar; con metros estandarizados alcanza 0,701. Cambiar a una unidad física correcta sin controlar su peso frente al resto no garantiza mejor predicción. El árbol conserva AUC 0,650 y AP 0,566 en las cuatro variantes: sus particiones no cambian en este experimento.",
14:f"**Lectura de la curva y mapas:** silhouette tiene máximo interior en K=4 (0,466), por encima de K=3 (0,422) y K=6 (0,434), aunque no demuestra cuatro zonas naturales únicas. DBSCAN produce {db['grupos']} grupos y {db['sin_grupo']} puntos de ruido (cobertura {db['cobertura']:.1%}), con silhouette {db['silhouette']:.3f}. El mayor grupo encadena miles de reclamos y no se recomienda como foco compacto. La tabla de focos filtra por soporte y radio p90; un silhouette negativo impide presentar todo el agrupamiento como separación exitosa.",
17:f"**Resultado:** AP temporal sube de {z[0]['AP']:.3f} con coordenadas directas a {z[1]['AP']:.3f} con distancias a {r['zonas']['K']} centroides; diferencia {z[1]['AP']-z[0]['AP']:+.3f}. Frente a la logística sin ubicación (AP {base['AP']:.3f}), ambas representaciones aportan en esta frontera. K=12 es el extremo superior de la grilla; no se afirma que sea el óptimo global.",
20:f"**Selección real:** el bosque obtuvo AP 0,6692 y la red 0,6690 en desarrollo. La diferencia es de unas diezmilésimas y no elige un ganador por sí sola. La regla de costo de ajuste+predicción entre candidatos cercanos seleccionó {r['recomendado']}; el bosque ofrece análisis de importancias y no exige una red para un resultado prácticamente igual en validación.",
23:f"**Comprobación de la recomendación:** el bosque alcanza AUC {rf['AUC']:.3f} y AP {rf['AP']:.3f}; logística sin ubicación AUC {lb['AUC']:.3f} y AP {lb['AP']:.3f}. La mejora de AUC no viene con mejora de AP (diferencia {rf['AP']-lb['AP']:+.4f}). Por eso la recomendación del bosque, fijada en desarrollo, es un candidato para evaluación adicional: no hay evidencia de mejora uniforme para reemplazar automáticamente la logística simple, que además es más barata. No se retocan hiperparámetros con 2019. La CV mezclada del bosque da AUC 0,880 frente a 0,760 de test: +0,120; la temporal de 0,747 es ligeramente más pesimista que test (-0,013).",
26:f"**Conclusión cuantitativa:** no se demuestra superioridad uniforme del bosque sobre la base logística en 2019: mejora ordenamiento AUC pero no AP. El tipo de problema es la variable con mayor caída de AP al permutar (~0,145); la ubicación conjunta reduce AP {r['importancias']['permutacion_ubicacion_conjunta_AP']:.4f}. La fecha relativa tiene importancia de impureza ~0,098 pero caída por permutación nula en 2019: sirvió para cortar entrenamiento, sin discriminar ese año posterior por los umbrales aprendidos. Se recomienda revisar los focos compactos de sección 4 y validar prospectivamente antes de usar el modelo para comunicar plazos. No se garantizan cumplimiento individual ni causas de demora."
}
marca='\n\n<!-- resultados-verificados -->\n\n'
for i,texto in respuestas.items():
    s=''.join(nb['cells'][i]['source']).split(marca)[0]
    nb['cells'][i]['source']=(s.rstrip()+marca+texto+'\n').splitlines(keepends=True)
p.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
print('Interpretaciones basadas en salidas verificadas añadidas')
