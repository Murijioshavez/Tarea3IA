# Chronos Forecast

Aplicación Flask que carga un CSV, prepara la serie y genera pronósticos con
**Chronos-2** (`amazon/chronos-2`), incluyendo validación temporal contra datos no vistos.

## Puesta en marcha

Se requiere **CPython 3.13 (64 bits)**. PyTorch todavía no publica ruedas binarias para
3.14+, así que ese intérprete no sirve.

```powershell
py -3.13 -m venv .venv313
.\.venv313\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Si no tienes Python 3.13 instalado, [`uv`](https://docs.astral.sh/uv/) lo descarga solo:

```powershell
uv venv --python 3.13 .venv313
uv pip install --python .venv313\Scripts\python.exe -r requirements.txt
```

Los pesos del modelo (~120 MB) se descargan de Hugging Face la primera vez que se genera
un pronóstico y quedan en `.cache/huggingface/`. No se versionan.

## Ejecutar

```powershell
.\.venv313\Scripts\python.exe app.py
```

Abre <http://127.0.0.1:5000>. Variables de entorno opcionales: `FLASK_HOST`, `FLASK_PORT`
y `FLASK_DEBUG=1` (el modo debug expone una consola interactiva, así que está apagado por
defecto).

## Dataset de ejemplo

El dataset de Kaggle se descarga de forma reproducible, sin bajarlo a mano:

```powershell
.\.venv313\Scripts\python.exe scripts\fetch_dataset.py
```

Esto escribe `data/car_purchasing.csv` (ignorado por git). El archivo original **no tiene
columna de fecha**, y Chronos-2 necesita un índice temporal regular, así que el script
añade uno diario sobre el orden de las filas y conserva únicamente las columnas
financieras. Nombre, correo y país se descartan: los dos primeros son únicos por fila y
`country` tiene 211 valores distintos en 500 filas, así que ninguno aporta señal.

En la interfaz selecciona:

| Campo | Valor |
| --- | --- |
| Columna temporal | `fecha` |
| Variable a pronosticar | `car purchase amount` |
| Covariables | `age`, `annual Salary`, `net worth` (`credit card debt` y `gender` aportan muy poco) |
| Horizonte | `50` |

## Cómo funciona

1. **Carga** (`POST /api/upload`) — guarda el CSV y devuelve una vista previa. Los archivos
   con más de 24 h se eliminan automáticamente.
2. **Preparación** (`services/data_processing.py`) — valida columnas, convierte tipos,
   ordena por fecha e infiere la frecuencia. Si los timestamps tienen huecos, se usa el
   intervalo dominante en vez de rechazar el archivo.
3. **Pronóstico** (`services/forecasting.py`) — aparta las últimas `horizon` observaciones
   como validación, predice ese tramo usando solo lo anterior, y luego pronostica el futuro
   con todo el historial.

### Covariables

Las covariables son las variables explicativas que acompañan al objetivo. Sin ellas el
modelo solo ve la columna objetivo y no puede aprovechar ninguna relación entre columnas.

El tratamiento difiere según el tramo, y esto es deliberado:

- **En validación** las covariables del tramo reservado ya se observaron, así que se pasan
  como covariables *futuras conocidas* (`future_df`).
- **En el pronóstico futuro** no se conocen (no sabes el salario de un cliente que todavía
  no llega), así que solo se usan como covariables *pasadas*.

Por eso el pronóstico más allá del CSV es más conservador que las métricas de validación:
no se le da al modelo información que en la práctica no tendrías.

### Métricas

Cada variable se evalúa contra el tramo no visto:

| Métrica | Qué responde |
| --- | --- |
| **MAE** | Error promedio, en las unidades de los datos |
| **RMSE** | Como MAE pero penaliza más los errores grandes; siempre ≥ MAE |
| **MAPE** | Error como porcentaje del valor real (`mape_excluded` indica cuántos ceros se omitieron) |
| **R²** | ¿Supera a predecir el promedio? **≤ 0 significa que el modelo no aporta nada** |
| **MASE** | ¿Supera a repetir el último valor? `< 1` sí |
| **Cobertura** | % de valores reales dentro del intervalo 10–90 %. Debería rondar el 80 % |

Ninguna métrica se lee sola. Un MAE bajo con R² negativo significa que el modelo no está
aprendiendo nada; una cobertura del 98 % frente a un 80 % nominal significa que los
intervalos son tan anchos que no sirven para decidir.

## Verificación

```powershell
.\.venv313\Scripts\python.exe scripts\verify_chronos.py   # inferencia mínima, sin Flask
.\.venv313\Scripts\python.exe scripts\verify_api.py       # upload + forecast end-to-end
```

## Estructura

```
app.py                        Aplicación Flask y manejo de errores
routes/api.py                 Endpoints JSON: upload, preview, forecast
services/data_processing.py   Validación del CSV y construcción del input de Chronos-2
services/forecasting.py       Inferencia, validación temporal y métricas
templates/ static/            Interfaz: carga, configuración, gráfica y tabla
scripts/fetch_dataset.py      Descarga reproducible del dataset de Kaggle
scripts/verify_*.py           Pruebas de humo
```

## Limitaciones conocidas

- **Calibración de intervalos.** Con covariables la cobertura llega a ~96–98 % cuando el
  intervalo nominal es 80 %: las bandas son demasiado anchas. Están medidas pero no
  corregidas.
- **Una sola ventana de validación.** Las métricas salen de un único tramo reservado, así
  que tienen bastante ruido. Un backtest con varias ventanas daría cifras más estables.
- **Sin covariables categóricas.** Solo se aceptan columnas numéricas.
- **El CSV crudo de Kaggle no sirve tal cual.** `car_purchasing.csv` y `sales data file.csv`
  no tienen columna de fechas, así que el pronóstico los rechaza. Usa
  `scripts/fetch_dataset.py`, que añade el índice temporal. La codificación sí se resuelve
  sola: se intenta UTF-8, CP1252 y Latin-1 en ese orden.
- **Sin imputación.** Cualquier valor faltante o no numérico en las columnas seleccionadas
  rechaza el archivo en lugar de rellenarlo.
