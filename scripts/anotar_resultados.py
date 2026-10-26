"""Lecturas numéricas desde resultados reales, sin volver a ajustar modelos."""
from pathlib import Path
import json
root=Path(__file__).resolve().parents[1]
r=json.loads((root/'resultados/resumen.json').read_text(encoding='utf-8'))
p=root/'entrega_reclamos_im.ipynb'
nb=json.loads(p.read_text(encoding='utf-8'))
d=r['datos']
base=next(x for x in r['final'] if x['familia']=='Logística sin ubicación')
recomendado=next(x for x in r['final'] if x['familia']==r['recomendado'])
db=next(x for x in r['focos']['comparacion'] if x['metodo']=='DBSCAN')
z=r['zonas']['comparacion']
mezcla=next(x for x in r['brechas_bosque'] if x['esquema']=='5 folds mezclados')
temporal=next(x for x in r['brechas_bosque'] if x['esquema']=='Temporal 2017→2018')
seleccion=sorted(r['familias_validacion'],key=lambda x:x['AP'],reverse=True)
coordenadas='\n'.join(f"- {x['familia']}, {x['variante']}: AUC {x['AUC']:.3f}, AP {x['AP']:.3f}." for x in r['coordenadas'])
top=r['importancias']['tabla'][0]
comparacion=(f"**Comprobación final:** {r['recomendado']} obtiene AUC {recomendado['AUC']:.3f} y AP {recomendado['AP']:.3f}; "
 f"logística sin ubicación, AUC {base['AUC']:.3f} y AP {base['AP']:.3f}. "
 f"Diferencias: AUC {recomendado['AUC']-base['AUC']:+.4f}, AP {recomendado['AP']-base['AP']:+.4f}. "
 "La recomendación se fijó en desarrollo. Se informa si la mejora es pequeña o cambia de signo según la métrica; "
 "no se afirma superioridad uniforme ni se ajusta otra grilla mirando el test.")
respuestas={
5:f"**Conteos:** {d['ventana']} ingresos; {d['copias']} copias, {d['no_final']} no finalizados y {d['demoras_invalidas']} demora inválida excluidos. Población: {d['poblacion']}; prueba: {d['prueba']}.",
11:'**Resultados del contraste:**\n\n'+coordenadas+'\n\nEn KNN cambia el peso relativo de la ubicación. En el árbol se comparan cortes equivalentes. Las cifras son de esta ejecución, sin exigir que cambien en un sentido predeterminado.',
14:f"**Resultado:** K={r['focos']['K']} maximiza silhouette en la grilla. DBSCAN con eps={r['focos']['eps']:.0f} m y mínimo {r['focos']['min_samples']} forma {db['grupos']} grupos, deja {db['sin_grupo']} sin grupo y tiene silhouette {db['silhouette']:.3f}. Se recomiendan {len(r['focos']['recomendados'])} focos compactos. Los grupos extensos se muestran, pero no se presentan como una intervención conjunta.",
17:f"**Resultado:** coordenadas AP {z[0]['AP']:.3f}; distancias a {r['zonas']['K']} centros AP {z[1]['AP']:.3f}; diferencia {z[1]['AP']-z[0]['AP']:+.3f}. Se muestra la grilla completa. Un extremo no prueba óptimo global.",
20:f"**Selección:** las dos mayores AP son {seleccion[0]['familia']} ({seleccion[0]['AP']:.4f}) y {seleccion[1]['familia']} ({seleccion[1]['AP']:.4f}). La regla declarada seleccionó {r['recomendado']}. No se afirma superioridad por pocas milésimas.",
23:comparacion+f" CV mezclada del bosque: AUC {mezcla['AUC']:.3f}, brecha {mezcla['brecha_AUC_vs_2019']:+.3f} frente a 2019. Temporal: AUC {temporal['AUC']:.3f}, brecha {temporal['brecha_AUC_vs_2019']:+.3f}.",
26:comparacion+f" En permutación, {top['variable']} tiene la mayor caída de AP ({top['permutacion_caida_AP']:.4f}). Ubicación conjunta: caída {r['importancias']['permutacion_ubicacion_conjunta_AP']:.4f}. Se revisan los focos de sección 4. No se demuestran causas de demora ni se garantizan plazos individuales."
}
marca='\n\n<!-- resultados-verificados -->\n\n'
for i,texto in respuestas.items():
 s=''.join(nb['cells'][i]['source']).split(marca)[0]
 nb['cells'][i]['source']=(s.rstrip()+marca+texto+'\n').splitlines(keepends=True)
p.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8',newline='\n')
print('Interpretaciones actualizadas desde resultados reales')
