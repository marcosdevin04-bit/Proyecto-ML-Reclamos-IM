import hashlib,json,urllib.request
from pathlib import Path
from datetime import datetime,timezone,timedelta
root=Path(__file__).resolve().parent
root.mkdir(exist_ok=True)
base='https://ckan-data.montevideo.gub.uy/dataset/c34e11ea-c547-46d5-80c6-d0bf283c002f/resource/'
files={'reclamos.zip':base+'ccb644b1-79b0-4caa-a353-6cfa816f3f70/download/reclamos.zip','metadata.txt':base+'39172145-d4c3-452a-98e9-8f74b6e7fc33/download/metadata.txt'}
manifest={}
for name,url in files.items():
 p=root/name
 if not p.exists():
  print('Descargando',name,flush=True)
  urllib.request.urlretrieve(url,p)
 manifest[name]={'url':url,'bytes':p.stat().st_size,'sha256':hashlib.file_digest(p.open('rb'),'sha256').hexdigest()}
manifest['fecha_descarga']=datetime.now(timezone(timedelta(hours=-3))).isoformat()
(root/'fuentes.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(manifest,ensure_ascii=False,indent=2))
