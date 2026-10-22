"""Ejecuta la entrega en un kernel nuevo; no descarga datos ni elige otro test."""
import json, os, sys, time
from pathlib import Path
root=Path(__file__).resolve().parent
for key,folder in {'JUPYTER_PATH':'.jupyter','IPYTHONDIR':'.ipython',
                   'JUPYTER_RUNTIME_DIR':'.jupyter/runtime','MPLCONFIGDIR':'.matplotlib'}.items():
    os.environ[key]=str(root/folder)
for key in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']:os.environ[key]='1'
os.environ['PYTHONIOENCODING']='utf-8'
kernel=root/'.jupyter/kernels/reclamos-im';kernel.mkdir(parents=True,exist_ok=True)
(kernel/'kernel.json').write_text(json.dumps({'argv':[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}'],
    'display_name':'Reclamos IM','language':'python'}),encoding='utf-8')
import nbformat
from nbclient import NotebookClient
p=root/'entrega_reclamos_im.ipynb';nb=nbformat.read(p,as_version=4)
for i,c in enumerate(nb.cells):
    if c.cell_type=='code':compile(c.source,f'celda-{i}','exec')
def inicio(cell,cell_index,**kwargs):
    if cell.cell_type=='code':print(f'Celda {cell_index}: {cell.source.splitlines()[0]}',flush=True)
def fin(cell,cell_index,**kwargs):nbformat.write(nb,p)
cliente=NotebookClient(nb,timeout=7200,kernel_name='reclamos-im',
    resources={'metadata':{'path':str(root)}},on_cell_start=inicio,on_cell_executed=fin)
t=time.perf_counter()
try:cliente.execute()
finally:nbformat.write(nb,p)
print(f'Completado en {time.perf_counter()-t:.1f} segundos',flush=True)
