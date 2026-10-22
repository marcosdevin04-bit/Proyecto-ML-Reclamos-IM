"""Comprueba integridad de fuentes, salidas, estructura y coherencia de resultados."""
from pathlib import Path
import hashlib,json,zipfile
import nbformat
root=Path(__file__).resolve().parents[1]
nb=nbformat.read(root/'entrega_reclamos_im.ipynb',as_version=4)
nbformat.validate(nb)
original=nbformat.read(root/'entrega_reclamos_im_template.ipynb',as_version=4)
assert len(nb.cells)==len(original.cells)==27
for i in [1,3,6,9,12,15,18,21,24]:
    assert nb.cells[i].source==original.cells[i].source,f'Consigna modificada en celda {i}'
codes=[c for c in nb.cells if c.cell_type=='code']
assert [c.execution_count for c in codes]==list(range(1,len(codes)+1))
for i,c in enumerate(nb.cells):
    if c.cell_type=='code':
        compile(c.source,f'celda-{i}','exec')
        assert c.outputs,f'Sin salidas: {i}'
        assert not any(o.output_type=='error' for o in c.outputs),f'Error de ejecución: {i}'
r=json.loads((root/'resultados/resumen.json').read_text(encoding='utf-8'))
assert r['datos']['muestra']==50000
assert r['datos']['train_2017']+r['datos']['validacion_2018']==50000
assert len(r['familias_validacion'])==6 and len(r['final'])==7
assert len({f['n_prueba'] for f in r['final']})==1
assert r['final'][0]['n_prueba']==r['datos']['prueba']
for f in r['familias_validacion']+r['final']:
    assert 0<=f['AP']<=1 and 0<=f['AUC']<=1
    assert f['ajuste_s']>=0 and f['prediccion_s']>=0
sv=next(f for f in r['familias_validacion'] if f['familia']=='SVM RBF')
assert sv['n']==10000
for f in r['focos']['recomendados']:
    assert f['reclamos']>=30 and f['radio_p90_m']<=2500
m=json.loads((root/'data/fuentes.json').read_text(encoding='utf-8'))
for name in ['reclamos.zip','metadata.txt']:
    with (root/'data'/name).open('rb') as source:
        assert hashlib.file_digest(source,'sha256').hexdigest()==m[name]['sha256']
with zipfile.ZipFile(root/'data/reclamos.zip') as z:assert z.testzip() is None
assert (root/'GUIA_DEFENSA.pdf').read_bytes().startswith(b'%PDF-')
assert r['recomendado'] in (root/'GUIA_DEFENSA.md').read_text(encoding='utf-8')
print('OK: fuentes, nueve celdas ejecutadas, ocho consignas, seis familias, muestra, test, focos y guía')
