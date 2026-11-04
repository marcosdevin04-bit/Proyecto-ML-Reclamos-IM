# Predicción y focos de reclamos urbanos de Montevideo

Entrega 2 de Machine Learning (UCU, Ingeniería en Inteligencia Artificial y Ciencia de Datos). Con el registro abierto de reclamos de la Intendencia de Montevideo se busca:

1. anticipar, en el momento del ingreso, si un reclamo se va a cerrar en 7 días corridos o menos;
2. encontrar focos: zonas donde un mismo problema se concentra lo suficiente para justificar una intervención de conjunto.

## Archivos

- [`entrega_reclamos_im.ipynb`](entrega_reclamos_im.ipynb): el notebook de la entrega, armado sobre la plantilla oficial y ejecutado de punta a punta.
- [`entrega_reclamos_im.pdf`](entrega_reclamos_im.pdf) y [`entrega_reclamos_im_template.ipynb`](entrega_reclamos_im_template.ipynb): la consigna y la plantilla de la cátedra, sin cambios.
- [`GUIA_DEFENSA.md`](GUIA_DEFENSA.md): decisiones, resultados y preguntas para preparar la defensa oral.
- [Guía de defensa en PDF](GUIA_DEFENSA.pdf): la misma guía, lista para descargar, estudiar o imprimir.
- [`RELACION_CON_LABS.md`](RELACION_CON_LABS.md): qué parte del notebook sale de qué laboratorio del curso.
- `data/`: el ZIP oficial descargado el 22/10/2026, la descripción de campos publicada y `fuentes.json` con las URL y el SHA-256.
- `ejecutar_notebook.py`: ejecuta el notebook completo desde un kernel nuevo.

## Datos y evaluación

- Fuente: [Reclamos registrados en el Sistema Único de Reclamos](https://catalogodatos.gub.uy/dataset/reclamos-registrados-en-el-sistema-unico-de-reclamos-de-la-intendencia-de-montevideo), Licencia de Datos Abiertos del Uruguay.
- Ventana: reclamos ingresados entre 2017 y 2019. La población son los 378 136 reclamos finalizados con demora válida.
- Prueba: los reclamos de 2019, apartados antes de cualquier ajuste y usados solo en las secciones 7 y 8.
- Validación: entrenar con 2017 y validar con 2018, sobre una muestra de 50 000 reclamos estratificada por año (la SVM usa 10 000 de ellos).

## Resultados principales

- En 2019, la mejor AP es la de la **logística con distancias a 40 zonas de k-medias**: AP 0,720 y AUC 0,767, contra 0,700 y 0,741 de la logística sin ubicación y una prevalencia de 0,429. Es además la más barata y la única que se explica con sus coeficientes.
- Los modelos basados en árboles eran los mejores en validación, pero en 2019 quedan por debajo de la logística en AP. La validación cruzada con filas mezcladas sobrestima el AUC del bosque en 0,13, porque pone reclamos del mismo día en entrenamiento y en validación.
- Lo que más determina el cierre en una semana es el tipo de problema. La ubicación suma, pero bastante menos.
- Con las bocas de tormenta obstruidas, DBSCAN encuentra 52 focos; se recomiendan los 5 mayores (770 reclamos, alrededor de 1 km de radio cada uno). El silhouette prefiere k-medias, que reparte la ciudad sin encontrar focos.
- Defectos del archivo que condicionan todo: el catálogo de Limpieza cambió el 26/12/2017; la prevalencia sube de 33 % a 43 % entre 2017 y 2019; el 12 % de los reclamos se cierra a más de un año.

Las conclusiones no constituyen un diagnóstico sobre la gestión de la Intendencia de Montevideo.

## Reproducir

Requiere Python 3.12. Desde la carpeta del repo, en macOS o Linux:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python ejecutar_notebook.py
```

En Windows (PowerShell):

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -X utf8 ejecutar_notebook.py
```

El notebook extrae el CSV del ZIP la primera vez. La ejecución completa tarda unos 8 minutos; la mayor parte se va en las grillas de la sección 6 y en las predicciones de k vecinos sobre los 133 000 reclamos de 2019. Para abrirlo en VS Code o Jupyter, elegir como kernel el entorno `.venv` y ejecutar todas las celdas en orden.

## Entrega

Vence el jueves 5 de noviembre. Se entrega por Moodle el notebook ejecutado, con todas las salidas visibles; subir el repo a GitHub no reemplaza la entrega.

## Integrantes

- Marcos Devincenzi
- Santiago Alba
- Facundo Díaz
- Domingo Leiva
