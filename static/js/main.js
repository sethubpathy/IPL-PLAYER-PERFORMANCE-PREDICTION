/**
 * IPL Player Performance Prediction - Client-Side Interactive Scripts
 * Handles Chart.js instances, dynamic search filtering, and drawer expansion.
 */

document.addEventListener('DOMContentLoaded', function () {
    // Initialize common tooltips if bootstrap is available
    if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
        var tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }

    // Toggle match feature breakdown rows
    document.querySelectorAll('.btn-toggle-features').forEach(function (btn) {
        btn.addEventListener('click', function () {
            var targetId = this.getAttribute('data-target');
            var targetRow = document.getElementById(targetId);
            if (targetRow) {
                if (targetRow.classList.contains('d-none')) {
                    targetRow.classList.remove('d-none');
                    this.innerHTML = '<i class="bi bi-chevron-up"></i> Hide 12 Features';
                } else {
                    targetRow.classList.add('d-none');
                    this.innerHTML = '<i class="bi bi-sliders"></i> View 12 Features';
                }
            }
        });
    });

    // Client-side quick filter for players table
    var playerSearchInput = document.getElementById('tablePlayerSearch');
    if (playerSearchInput) {
        playerSearchInput.addEventListener('keyup', function () {
            var val = this.value.toLowerCase().trim();
            var rows = document.querySelectorAll('#playersCatalogTable tbody tr');
            rows.forEach(function (row) {
                var playerName = row.getAttribute('data-player-name') || '';
                if (playerName.toLowerCase().includes(val)) {
                    row.style.display = '';
                } else {
                    row.style.display = 'none';
                }
            });
        });
    }
});

/**
 * Render Feature Importance Horizontal Bar Chart
 */
function renderFeatureImportanceChart(canvasId, featuresData) {
    var ctx = document.getElementById(canvasId);
    if (!ctx) return;

    var labels = featuresData.map(function (f) { return f.name; });
    var values = featuresData.map(function (f) { return f.importance; });

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Importance Weight (%)',
                data: values,
                backgroundColor: function (context) {
                    var chart = context.chart;
                    var ctx = chart.ctx;
                    var gradient = ctx.createLinearGradient(0, 0, chart.width, 0);
                    gradient.addColorStop(0, '#00e5ff');
                    gradient.addColorStop(1, '#ffb703');
                    return gradient;
                },
                borderRadius: 6,
                borderSkipped: false
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: function (ctx) {
                            return ' ' + ctx.raw + '% of model prediction weight';
                        }
                    }
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8', callback: function(val) { return val + '%'; } }
                },
                y: {
                    grid: { display: false },
                    ticks: { color: '#f8fafc', font: { weight: '600' } }
                }
            }
        }
    });
}

/**
 * Render Actual vs Predicted Timeline Line Chart
 */
function renderPredictionsTimelineChart(canvasId, matchesData) {
    var ctx = document.getElementById(canvasId);
    if (!ctx) return;

    var labels = matchesData.map(function (m, idx) {
        return (m.date || ('M' + (idx + 1))) + ' (' + m.season + ')';
    });
    var actuals = matchesData.map(function (m) { return m.actual_performance; });
    var predicteds = matchesData.map(function (m) { return m.predicted_performance; });

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Actual Score (Runs + 20*Wkts)',
                    data: actuals,
                    borderColor: '#ffb703',
                    backgroundColor: 'rgba(255, 183, 3, 0.1)',
                    pointBackgroundColor: '#ffb703',
                    pointBorderColor: '#070b14',
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    tension: 0.25,
                    fill: true
                },
                {
                    label: 'Random Forest Predicted',
                    data: predicteds,
                    borderColor: '#00e5ff',
                    backgroundColor: 'rgba(0, 229, 255, 0.05)',
                    pointBackgroundColor: '#00e5ff',
                    pointBorderColor: '#070b14',
                    pointRadius: 5,
                    pointHoverRadius: 7,
                    borderDash: [5, 5],
                    tension: 0.25,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                mode: 'index',
                intersect: false
            },
            plugins: {
                legend: {
                    labels: { color: '#f8fafc', font: { size: 12, weight: '600' } }
                },
                tooltip: {
                    backgroundColor: 'rgba(14, 22, 38, 0.95)',
                    titleColor: '#00e5ff',
                    bodyColor: '#f8fafc',
                    borderColor: 'rgba(0, 229, 255, 0.3)',
                    borderWidth: 1,
                    padding: 12
                }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    ticks: { color: '#94a3b8', maxRotation: 45, minRotation: 30 }
                },
                y: {
                    title: { display: true, text: 'Performance Score', color: '#94a3b8' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#f8fafc' }
                }
            }
        }
    });
}

/**
 * Render Absolute Error Distribution Histogram
 */
function renderErrorDistributionChart(canvasId, distData) {
    var ctx = document.getElementById(canvasId);
    if (!ctx) return;

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: distData.labels,
            datasets: [{
                label: 'Matches in Error Range',
                data: distData.counts,
                backgroundColor: [
                    '#06d6a0', // 0-5 (very good)
                    '#48cae4', // 5-10
                    '#0077b6', // 10-20
                    '#ffb703', // 20-30
                    '#fb8500', // 30-40
                    '#ef476f', // 40-50
                    '#d90429'  // 50+
                ],
                borderRadius: 6
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { display: false }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Absolute Error Range (|Actual - Predicted|)', color: '#94a3b8' },
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    ticks: { color: '#f8fafc' }
                },
                y: {
                    title: { display: true, text: 'Number of Match Predictions', color: '#94a3b8' },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#94a3b8' }
                }
            }
        }
    });
}

/**
 * Render Season Career Progression Chart
 */
function renderSeasonProgressChart(canvasId, seasonsData) {
    var ctx = document.getElementById(canvasId);
    if (!ctx) return;

    var labels = seasonsData.map(function (s) { return s.season; });
    var runs = seasonsData.map(function (s) { return s.runs; });
    var wickets = seasonsData.map(function (s) { return s.wickets; });
    var avgPerf = seasonsData.map(function (s) { return s.avg_performance; });

    new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    type: 'bar',
                    label: 'Runs',
                    data: runs,
                    backgroundColor: 'rgba(255, 183, 3, 0.65)',
                    borderRadius: 4,
                    yAxisID: 'y'
                },
                {
                    type: 'bar',
                    label: 'Wickets',
                    data: wickets,
                    backgroundColor: 'rgba(0, 229, 255, 0.65)',
                    borderRadius: 4,
                    yAxisID: 'y'
                },
                {
                    type: 'line',
                    label: 'Avg Performance Score',
                    data: avgPerf,
                    borderColor: '#06d6a0',
                    backgroundColor: 'transparent',
                    pointBackgroundColor: '#06d6a0',
                    pointRadius: 4,
                    tension: 0.3,
                    yAxisID: 'y1'
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: { labels: { color: '#f8fafc' } }
            },
            scales: {
                x: {
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    ticks: { color: '#94a3b8' }
                },
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                    title: { display: true, text: 'Runs / Wickets', color: '#94a3b8' },
                    grid: { color: 'rgba(255, 255, 255, 0.04)' },
                    ticks: { color: '#f8fafc' }
                },
                y1: {
                    type: 'linear',
                    display: true,
                    position: 'right',
                    title: { display: true, text: 'Avg Score', color: '#06d6a0' },
                    grid: { drawOnChartArea: false },
                    ticks: { color: '#06d6a0' }
                }
            }
        }
    });
}
