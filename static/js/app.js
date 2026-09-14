const state = { datasetId: null, chart: null, timestampColumn: null };
const colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706"];
const uploadForm = document.querySelector("#upload-form");
const forecastForm = document.querySelector("#forecast-form");
const fileInput = document.querySelector("#csv-file");
const statusElement = document.querySelector("#status");
const uploadButton = uploadForm.querySelector("button");
const forecastButton = forecastForm.querySelector("button");

function showStatus(message, isError = false) {
    statusElement.textContent = message;
    statusElement.classList.toggle("error", isError);
}

function setLoading(button, loading, label) {
    button.disabled = loading;
    button.dataset.label ??= button.textContent;
    button.textContent = loading ? label : button.dataset.label;
}

async function readResponse(response) {
    const body = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(body.error || "Ocurrió un error inesperado.");
    return body;
}

function appendCell(row, value) {
    const cell = document.createElement("td");
    cell.textContent = value ?? "";
    row.append(cell);
}

function renderPreview(data) {
    const table = document.querySelector("#preview-table");
    const header = table.querySelector("thead");
    const body = table.querySelector("tbody");
    header.replaceChildren();
    body.replaceChildren();
    const headerRow = document.createElement("tr");
    data.columns.forEach(({ name }) => {
        const cell = document.createElement("th");
        cell.textContent = name;
        headerRow.append(cell);
    });
    header.append(headerRow);
    data.preview.forEach((record) => {
        const row = document.createElement("tr");
        data.columns.forEach(({ name }) => appendCell(row, record[name]));
        body.append(row);
    });
}

function fillSelectors(columns, timestampCandidates) {
    const timestampSelect = document.querySelector("#timestamp-column");
    const targetSelect = document.querySelector("#target-columns");
    const covariateSelect = document.querySelector("#covariate-columns");
    timestampSelect.replaceChildren();
    targetSelect.replaceChildren();
    covariateSelect.replaceChildren();

    // Only columns the backend accepted as a time axis belong in the timestamp list;
    // offering the rest just leads to a rejected forecast two steps later.
    const usable = new Set(timestampCandidates);
    columns.forEach(({ name, dtype }) => {
        if (usable.has(name)) timestampSelect.add(new Option(`${name} (${dtype})`, name));
        targetSelect.add(new Option(`${name} (${dtype})`, name));
        if (/int|float|double|number/i.test(dtype)) covariateSelect.add(new Option(`${name} (${dtype})`, name));
    });

    const likelyTimestamp = timestampCandidates.find((name) => /time|date|fecha/i.test(name));
    timestampSelect.value = likelyTimestamp ?? timestampCandidates[0] ?? "";
    const firstNumeric = columns.find(({ dtype }) => /int|float|double|number/i.test(dtype));
    if (firstNumeric) targetSelect.value = firstNumeric.name;
    syncSelectors();
    return usable.size > 0;
}

/** A column cannot be the time axis, a target and a covariate at once; show that in the UI. */
function syncSelectors() {
    const timestampColumn = document.querySelector("#timestamp-column").value;
    const targetSelect = document.querySelector("#target-columns");
    const targetColumns = new Set([...targetSelect.selectedOptions].map((option) => option.value));

    [...targetSelect.options].forEach((option) => {
        option.disabled = option.value === timestampColumn;
        if (option.disabled) option.selected = false;
    });
    const covariateSelect = document.querySelector("#covariate-columns");
    [...covariateSelect.options].forEach((option) => {
        option.disabled = option.value === timestampColumn || targetColumns.has(option.value);
        if (option.disabled) option.selected = false;
    });

    // Ctrl+clicking an already-selected entry silently clears it, so echo the current
    // selection back instead of leaving the user to guess what the list box holds.
    describeSelection("#target-help", targetSelect, "Ninguna seleccionada — elige al menos una.", true);
    describeSelection("#covariate-help", covariateSelect, "Ninguna seleccionada (opcional).", false);
}

function describeSelection(helpSelector, select, emptyMessage, required) {
    const chosen = [...select.selectedOptions].map((option) => option.value);
    const help = document.querySelector(helpSelector);
    help.textContent = chosen.length ? `Seleccionadas (${chosen.length}): ${chosen.join(", ")}` : emptyMessage;
    help.classList.toggle("empty-selection", !chosen.length && required);
}

function formatNumber(value) {
    return new Intl.NumberFormat("es", { maximumFractionDigits: 3 }).format(value);
}

function formatTimestamp(value) {
    const text = String(value ?? "");
    return text.endsWith("T00:00:00") ? text.slice(0, 10) : text.replace("T", " ");
}

function renderForecastTable(forecast) {
    const table = document.querySelector("#forecast-table");
    const headerRow = document.createElement("tr");
    ["Timestamp", "Variable", "Predicción", "Límite inferior", "Mediana", "Límite superior"].forEach((title) => {
        const cell = document.createElement("th");
        cell.textContent = title;
        headerRow.append(cell);
    });
    table.querySelector("thead").replaceChildren(headerRow);
    const body = table.querySelector("tbody");
    body.replaceChildren();
    forecast.forEach((row) => {
        const tableRow = document.createElement("tr");
        appendCell(tableRow, formatTimestamp(row.timestamp));
        appendCell(tableRow, row.target);
        appendCell(tableRow, formatNumber(row.prediction));
        appendCell(tableRow, formatNumber(row.lower));
        appendCell(tableRow, formatNumber(row.median));
        appendCell(tableRow, formatNumber(row.upper));
        body.append(tableRow);
    });
}

function renderMetrics(validation) {
    document.querySelector("#validation-summary").textContent = `Entrenamiento: ${validation.train_observations} observaciones · Validación: ${validation.validation_observations} observaciones no vistas.`;
    const grid = document.querySelector("#metrics-grid");
    grid.replaceChildren();
    Object.entries(validation.metrics).forEach(([target, values]) => {
        const card = document.createElement("article");
        card.className = "metric-card";
        const heading = document.createElement("h4");
        heading.textContent = target;
        const metricValues = document.createElement("div");
        metricValues.className = "metric-values";
        [
            ["MAE", formatNumber(values.mae)],
            ["RMSE", formatNumber(values.rmse)],
            ["MAPE", values.mape === null ? "No aplica" : `${formatNumber(values.mape)} %`],
            ["R²", values.r2 === null ? "No aplica" : formatNumber(values.r2)],
            ["MASE", values.mase === null ? "No aplica" : formatNumber(values.mase)],
            ["Cobertura", `${formatNumber(values.coverage)} % de ${formatNumber(values.nominal_coverage)} %`],
        ].forEach(([label, value]) => {
            const metric = document.createElement("span");
            metric.append(label);
            const number = document.createElement("strong");
            number.textContent = value;
            metric.append(number);
            metricValues.append(metric);
        });
        card.append(heading, metricValues);
        grid.append(card);
    });
}

function renderChart(result) {
    if (!window.Chart) throw new Error("No se pudo cargar la librería de gráficas. Comprueba tu conexión e inténtalo de nuevo.");
    if (state.chart) state.chart.destroy();
    // Historical, validation and forecast timestamps all arrive as the same ISO string from
    // the backend. Truncating only some of them silently breaks every lookup below, which is
    // what previously left the validation overlay empty.
    const labels = [...new Set([
        ...result.historical.map((row) => row[state.timestampColumn]),
        ...result.forecast.map((row) => row.timestamp),
    ])].sort();
    const datasets = [];
    result.metadata.target_columns.forEach((target, index) => {
        const color = colors[index % colors.length];
        const byTimestamp = (rows, property) => new Map(rows.map((row) => [row.timestamp, row[property]]));
        const historical = new Map(result.historical.map((row) => [row[state.timestampColumn], row[target]]));
        const targetForecast = result.forecast.filter((row) => row.target === target);
        const lower = byTimestamp(targetForecast, "lower");
        const upper = byTimestamp(targetForecast, "upper");
        const predictions = byTimestamp(targetForecast, "prediction");
        const validationPredictions = byTimestamp(result.validation.predictions.filter((row) => row.target === target), "prediction");
        datasets.push({ label: `${target} · histórico`, data: labels.map((label) => historical.get(label) ?? null), borderColor: color, borderWidth: 2, pointRadius: 0, tension: .2 });
        datasets.push({ label: `${target} · límite inferior`, data: labels.map((label) => lower.get(label) ?? null), borderColor: "transparent", borderWidth: 0, pointRadius: 0, fill: false });
        datasets.push({ label: `${target} · intervalo 10–90 %`, data: labels.map((label) => upper.get(label) ?? null), borderColor: "transparent", backgroundColor: `${color}26`, borderWidth: 0, pointRadius: 0, fill: "-1" });
        datasets.push({ label: `${target} · forecast`, data: labels.map((label) => predictions.get(label) ?? null), borderColor: color, borderDash: [6, 4], borderWidth: 2, pointRadius: 2, tension: .2 });
        datasets.push({ label: `${target} · validación estimada`, data: labels.map((label) => validationPredictions.get(label) ?? null), borderColor: color, borderDash: [2, 3], borderWidth: 2, pointRadius: 3, tension: .2 });
    });
    state.chart = new Chart(document.querySelector("#forecast-chart"), {
        type: "line", data: { labels, datasets },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: { mode: "index", intersect: false },
            scales: {
                // Long series produce hundreds of labels, so thin them out and drop the
                // time-of-day suffix unless the series actually needs it.
                x: { ticks: { autoSkip: true, maxTicksLimit: 12, maxRotation: 0, callback(value) { return formatTimestamp(this.getLabelForValue(value)); } } },
                y: { title: { display: true, text: "Valor" } },
            },
            plugins: {
                legend: { labels: { filter: (item) => !item.text.includes("límite inferior") } },
                tooltip: { callbacks: { title: (items) => formatTimestamp(items[0]?.label) } },
            },
        },
    });
}

document.querySelector("#timestamp-column").addEventListener("change", syncSelectors);
document.querySelector("#target-columns").addEventListener("change", syncSelectors);
document.querySelector("#covariate-columns").addEventListener("change", syncSelectors);

fileInput.addEventListener("change", () => {
    document.querySelector("#selected-file").textContent = fileInput.files[0]?.name || "Ningún archivo seleccionado";
});

uploadForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    if (!fileInput.files[0]) return;
    setLoading(uploadButton, true, "Cargando…");
    showStatus("Leyendo y validando el CSV…");
    try {
        const formData = new FormData();
        formData.append("file", fileInput.files[0]);
        const data = await readResponse(await fetch("/api/upload", { method: "POST", body: formData }));
        state.datasetId = data.dataset_id;
        renderPreview(data);
        const hasTimeAxis = fillSelectors(data.columns, data.timestamp_candidates ?? []);
        document.querySelector("#dataset-summary").textContent = `${data.row_count} filas · ${data.columns.length} columnas`;
        document.querySelector("#dataset-section").classList.remove("hidden");
        document.querySelector("#results-section").classList.add("hidden");

        const warning = document.querySelector("#no-time-axis");
        warning.hidden = hasTimeAxis;
        document.querySelector("#forecast-form button[type=submit]").disabled = !hasTimeAxis;
        if (!hasTimeAxis) {
            showStatus("Este CSV no tiene ninguna columna de fechas, así que no es una serie temporal.", true);
            return;
        }
        showStatus("Dataset cargado. Configura las columnas y el horizonte.");
    } catch (error) { showStatus(error.message, true); }
    finally { setLoading(uploadButton, false); }
});

forecastForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const targetColumns = [...document.querySelector("#target-columns").selectedOptions].map((option) => option.value);
    state.timestampColumn = document.querySelector("#timestamp-column").value;
    if (!targetColumns.length) {
        showStatus("Selecciona al menos una variable a pronosticar antes de generar el forecast.", true);
        document.querySelector("#target-columns").focus();
        return;
    }
    const excluded = new Set([...targetColumns, state.timestampColumn]);
    const covariateColumns = [...document.querySelector("#covariate-columns").selectedOptions]
        .map((option) => option.value)
        .filter((name) => !excluded.has(name));
    setLoading(forecastButton, true, "Generando…");
    showStatus("Chronos-2 está generando el forecast…");
    try {
        const result = await readResponse(await fetch("/api/forecast", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ dataset_id: state.datasetId, timestamp_column: state.timestampColumn, target_columns: targetColumns, covariate_columns: covariateColumns, horizon: Number(document.querySelector("#horizon").value) }) }));
        renderChart(result);
        renderForecastTable(result.forecast);
        renderMetrics(result.validation);
        document.querySelector("#forecast-summary").textContent = `${result.metadata.horizon} periodos · frecuencia ${result.metadata.frequency}`;
        document.querySelector("#results-section").classList.remove("hidden");
        showStatus("Forecast generado correctamente.");
    } catch (error) { showStatus(error.message, true); }
    finally { setLoading(forecastButton, false); }
});
