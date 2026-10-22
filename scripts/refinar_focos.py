"""Reejecuta solo el análisis de desarrollo afectado por la recomendación de focos."""
import os,json,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1]
for key,folder in {'JUPYTER_PATH':'.jupyter','IPYTHONDIR':'.ipython','JUPYTER_RUNTIME_DIR':'.jupyter/runtime','MPLCONFIGDIR':'.matplotlib'}.items():os.environ[key]=str(root/folder)
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
import nbformat
from nbclient import NotebookClient
ns={'__file__':str(root/'scripts/construir_notebook.py')}
src=(root/'scripts/construir_notebook.py').read_text(encoding='utf-8')
exec(src.split('for i,s in C.items():')[0],ns)
p=root/'entrega_reclamos_im.ipynb';nb=nbformat.read(p,as_version=4)
nb.cells[13].source=ns['C'][13];nb.cells[14].source=ns['M'][14]
seleccion=[2,4,7,13]
parte=nbformat.v4.new_notebook(cells=[nbformat.v4.new_code_cell(nb.cells[i].source) for i in seleccion])
parte.cells.append(nbformat.v4.new_code_cell("p=SALIDA/'resumen.json'\nr=json.loads(p.read_text(encoding='utf-8'))\nr['focos']=RES['focos']\np.write_text(json.dumps(r,ensure_ascii=False,indent=2),encoding='utf-8')"))
NotebookClient(parte,timeout=1200,kernel_name='reclamos-im',resources={'metadata':{'path':str(root)}}).execute()
nb.cells[13].outputs=parte.cells[3].outputs
nbformat.write(nb,p)
print('Focos revisados con solo 2017–2018; no se repite ni ajusta test 2019')
