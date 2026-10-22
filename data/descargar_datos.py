import hashlib,json,urllib.request
from pathlib import Path
from datetime import datetime,timezone,timedelta
root=Path(__file__).resolve().parent
root.mkdir(exist_ok=True)
base='https://ckan-data.montevideo.gub.uy/dataset/c34e11ea-c547-46d5-80c6-d0bf283c002f/resource/'
files={'reclamos.zip':base+'ccb644b1-79b0-4caa-a353-6cfa816f3f70/download/reclamos.zip','metadata.txt':base+'39172145-d4c3-452a-98e9-8f74b6e7fc33/download/metadata.txt'}
manifest={}
previo=json.loads((root/'fuentes.json').read_text(encoding='utf-8')) if (root/'fuentes.json').exists() else None
for name,url in files.items():
 p=root/name
 if not p.exists():
  print('Descargando',name,flush=True)
  temporal=p.with_suffix(p.suffix+'.part')
  urllib.request.urlretrieve(url,temporal)
  temporal.replace(p)
 with p.open('rb') as f:
  manifest[name]={'url':url,'bytes':p.stat().st_size,'sha256':hashlib.file_digest(f,'sha256').hexdigest()}
 if previo and manifest[name]['sha256']!=previo[name]['sha256']:
  raise RuntimeError('El recurso cambió: conservar versiones y auditar antes de reemplazar el manifiesto')
manifest['fecha_descarga']=previo['fecha_descarga'] if previo else datetime.now(timezone(timedelta(hours=-3))).isoformat()
(root/'fuentes.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(manifest,ensure_ascii=False,indent=2))
