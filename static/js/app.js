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

function fillSelectors(columns) {
    const timestampSelect = document.querySelector("#timestamp-column");
    const targetSelect = document.querySelector("#target-columns");
    timestampSelect.replaceChildren();
    targetSelect.replaceChildren();
    columns.forEach(({ name, dtype }) => {
        timestampSelect.add(new Option(`${name} (${dtype})`, name));
        targetSelect.add(new Option(`${name} (${dtype})`, name));
    });
    const likelyTimestamp = columns.find(({ name }) => /time|date|fecha/i.test(name));
    if (likelyTimestamp) timestampSelect.value = likelyTimestamp.name;
    const firstNumeric = columns.find(({ dtype }) => /int|float|double|number/i.test(dtype));
    if (firstNumeric) targetSelect.value = firstNumeric.name;
}

function formatNumber(value) {
    return new Intl.NumberFormat("es", { maximumFractionDigits: 3 }).format(value);
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
        appendCell(tableRow, row.timestamp);
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
        [["MAE", formatNumber(values.mae)], ["RMSE", formatNumber(values.rmse)], ["MAPE", values.mape === null ? "No aplica" : `${formatNumber(values.mape)} %`]].forEach(([label, value]) => {
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
    const labels = [...new Set([
        ...result.historical.map((row) => row[state.timestampColumn]),
        ...result.forecast.map((row) => row.timestamp.split("T")[0]),
    ])];
    const datasets = [];
    result.metadata.target_columns.forEach((target, index) => {
        const color = colors[index % colors.length];
        const byDate = (rows, property) => new Map(rows.map((row) => [row.timestamp.split("T")[0], row[property]]));
        const historical = new Map(result.historical.map((row) => [row[state.timestampColumn], row[target]]));
        const targetForecast = result.forecast.filter((row) => row.target === target);
        const lower = byDate(targetForecast, "lower");
        const upper = byDate(targetForecast, "upper");
        const predictions = byDate(targetForecast, "prediction");
        const validationPredictions = byDate(result.validation.predictions.filter((row) => row.target === target), "prediction");
        datasets.push({ label: `${target} · histórico`, data: labels.map((label) => historical.get(label) ?? null), borderColor: color, borderWidth: 2, pointRadius: 0, tension: .2 });
        datasets.push({ label: `${target} · límite inferior`, data: labels.map((label) => lower.get(label) ?? null), borderColor: "transparent", borderWidth: 0, pointRadius: 0, fill: false });
        datasets.push({ label: `${target} · intervalo 10–90 %`, data: labels.map((label) => upper.get(label) ?? null), borderColor: "transparent", backgroundColor: `${color}26`, borderWidth: 0, pointRadius: 0, fill: "-1" });
        datasets.push({ label: `${target} · forecast`, data: labels.map((label) => predictions.get(label) ?? null), borderColor: color, borderDash: [6, 4], borderWidth: 2, pointRadius: 2, tension: .2 });
        datasets.push({ label: `${target} · validación estimada`, data: labels.map((label) => validationPredictions.get(label) ?? null), borderColor: color, borderDash: [2, 3], borderWidth: 2, pointRadius: 3, tension: .2 });
    });
    state.chart = new Chart(document.querySelector("#forecast-chart"), {
        type: "line", data: { labels, datasets },
        options: { responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false }, scales: { y: { title: { display: true, text: "Valor" } } }, plugins: { legend: { labels: { filter: (item) => !item.text.includes("límite inferior") } } } },
    });
}

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
        fillSelectors(data.columns);
        document.querySelector("#dataset-summary").textContent = `${data.row_count} filas · ${data.columns.length} columnas`;
        document.querySelector("#dataset-section").classList.remove("hidden");
        document.querySelector("#results-section").classList.add("hidden");
        showStatus("Dataset cargado. Configura las columnas y el horizonte.");
    } catch (error) { showStatus(error.message, true); }
    finally { setLoading(uploadButton, false); }
});

forecastForm.addEventListener("submit", async (event) => {
    event.preventDefault();
    const targetColumns = [...document.querySelector("#target-columns").selectedOptions].map((option) => option.value);
    state.timestampColumn = document.querySelector("#timestamp-column").value;
    setLoading(forecastButton, true, "Generando…");
    showStatus("Chronos-2 está generando el forecast…");
    try {
        const result = await readResponse(await fetch("/api/forecast", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ dataset_id: state.datasetId, timestamp_column: state.timestampColumn, target_columns: targetColumns, horizon: Number(document.querySelector("#horizon").value) }) }));
        renderChart(result);
        renderForecastTable(result.forecast);
        renderMetrics(result.validation);
        document.querySelector("#forecast-summary").textContent = `${result.metadata.horizon} periodos · frecuencia ${result.metadata.frequency}`;
        document.querySelector("#results-section").classList.remove("hidden");
        showStatus("Forecast generado correctamente.");
    } catch (error) { showStatus(error.message, true); }
    finally { setLoading(forecastButton, false); }
});
