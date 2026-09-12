# Chronos Forecast

MVP para cargar un CSV, preparar series temporales y generar pronósticos con
Chronos-2 mediante Flask.

## Requisitos locales

Use CPython 3.13 (64 bits). El intérprete actual del equipo es Python 3.14
distribuido por MSYS2/MinGW; Pandas y PyTorch no publican las distribuciones
binarias requeridas para esa combinación, por lo que no es apto para ejecutar
Chronos-2.

Una vez instalado CPython compatible, cree y active un entorno virtual en esta
carpeta e instale las dependencias:

```powershell
py -3.13 -m venv .venv313
.\.venv313\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

La siguiente fase añadirá una prueba independiente que descargará el modelo
`amazon/chronos-2` desde Hugging Face al ejecutarse. Sus pesos no se guardan en
el repositorio.
