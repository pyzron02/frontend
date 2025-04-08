// Enhanced Dashboard JavaScript

document.addEventListener('DOMContentLoaded', function() {
    // Initialize the tabbed interface for metrics
    initializeMetricsTabs();
    
    // Initialize expandable sections
    initializeExpandableSections();
    
    // Initialize tooltips
    initializeTooltips();
    
    // Load charts if Chart.js is available
    if (typeof Chart !== 'undefined') {
        loadCharts();
    }
    
    // Check for running backtests every 5 seconds
    if (document.querySelectorAll('.running').length > 0) {
        setInterval(checkBacktestStatus, 5000);
    }
    
    // Set up cancel buttons
    document.querySelectorAll('.cancel-btn').forEach(button => {
        button.addEventListener('click', function() {
            const jobId = this.getAttribute('data-job-id');
            cancelBacktest(jobId);
        });
    });
    
    // Auto-dismiss flash messages after 5 seconds
    setTimeout(function() {
        const alerts = document.querySelectorAll('.alert-container .alert');
        alerts.forEach(alert => {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        });
    }, 5000);
});

// Initialize the tabbed interface for metrics
function initializeMetricsTabs() {
    document.querySelectorAll('.metrics-tabs .nav-link').forEach(tab => {
        tab.addEventListener('click', function(e) {
            e.preventDefault();
            
            // Get the target content ID from the data attribute
            const targetId = this.getAttribute('data-bs-target');
            
            // Remove active class from all tabs and content
            document.querySelectorAll('.metrics-tabs .nav-link').forEach(t => {
                t.classList.remove('active');
            });
            
            document.querySelectorAll('.metrics-tabs .tab-pane').forEach(content => {
                content.classList.remove('active', 'show');
            });
            
            // Add active class to the clicked tab and its content
            this.classList.add('active');
            document.querySelector(targetId).classList.add('active', 'show');
        });
    });
}

// Initialize expandable sections
function initializeExpandableSections() {
    document.querySelectorAll('.expandable-header').forEach(header => {
        header.addEventListener('click', function() {
            const content = this.nextElementSibling;
            const icon = this.querySelector('.expand-icon');
            
            // Toggle the visibility of the content
            if (content.style.display === 'none' || !content.style.display) {
                content.style.display = 'block';
                if (icon) icon.textContent = '−'; // Minus sign
            } else {
                content.style.display = 'none';
                if (icon) icon.textContent = '+'; // Plus sign
            }
        });
    });
}

// Initialize tooltips
function initializeTooltips() {
    if (typeof bootstrap !== 'undefined' && bootstrap.Tooltip) {
        const tooltipTriggerList = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
        tooltipTriggerList.map(function (tooltipTriggerEl) {
            return new bootstrap.Tooltip(tooltipTriggerEl);
        });
    }
}

// Load charts for the dashboard
function loadCharts() {
    // Find all equity chart canvas elements
    document.querySelectorAll('[id^="equity-chart-"]').forEach(function(canvas) {
        try {
            // Try to parse data attributes safely
            let labels = [];
            let values = [];
            
            try {
                // Add safety checks for JSON.parse
                if (canvas.hasAttribute('data-labels')) {
                    try {
                        const labelsData = canvas.getAttribute('data-labels');
                        if (labelsData && labelsData.trim() !== '') {
                            labels = JSON.parse(labelsData);
                        }
                    } catch (parseError) {
                        console.error(`Error parsing labels for ${canvas.id}:`, parseError);
                    }
                }
                
                if (canvas.hasAttribute('data-values')) {
                    try {
                        const valuesData = canvas.getAttribute('data-values');
                        if (valuesData && valuesData.trim() !== '') {
                            values = JSON.parse(valuesData);
                            
                            // Ensure all values are actually numbers
                            values = values.map(v => {
                                const num = parseFloat(v);
                                return isNaN(num) ? 0 : num;
                            });
                        }
                    } catch (parseError) {
                        console.error(`Error parsing values for ${canvas.id}:`, parseError);
                    }
                }
            } catch (e) {
                console.error('Error accessing data attributes:', e);
                return;
            }
            
            // Only create chart if we have valid data
            if (Array.isArray(labels) && Array.isArray(values) && labels.length > 0 && values.length > 0) {
                // Create the chart with error handling
                try {
                    new Chart(canvas, {
                        type: 'line',
                        data: {
                            labels: labels,
                            datasets: [{
                                label: 'Strategy',
                                data: values,
                                borderColor: '#007bff',
                                backgroundColor: 'rgba(0, 123, 255, 0.1)',
                                tension: 0.1,
                                fill: true
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                y: {
                                    beginAtZero: false,
                                    title: {
                                        display: true,
                                        text: 'Portfolio Value'
                                    }
                                },
                                x: {
                                    title: {
                                        display: true,
                                        text: 'Date'
                                    }
                                }
                            }
                        }
                    });
                } catch (chartError) {
                    console.error(`Error creating chart for ${canvas.id}:`, chartError);
                    canvas.parentNode.innerHTML = '<div class="alert alert-danger">Error creating chart</div>';
                }
            } else {
                console.warn('Insufficient data for chart:', canvas.id);
                canvas.parentNode.innerHTML = '<div class="alert alert-warning">Insufficient data for chart</div>';
            }
        } catch (e) {
            console.error('Error creating chart:', e);
            canvas.parentNode.innerHTML = '<div class="alert alert-danger">Error creating chart</div>';
        }
    });
    
    // Find all monthly chart canvas elements
    document.querySelectorAll('[id^="monthly-chart-"]').forEach(function(canvas) {
        try {
            // Try to parse data attributes safely
            let labels = [];
            let values = [];
            
            try {
                // Add safety checks for JSON.parse
                if (canvas.hasAttribute('data-labels')) {
                    try {
                        const labelsData = canvas.getAttribute('data-labels');
                        if (labelsData && labelsData.trim() !== '') {
                            labels = JSON.parse(labelsData);
                        }
                    } catch (parseError) {
                        console.error(`Error parsing labels for ${canvas.id}:`, parseError);
                    }
                }
                
                if (canvas.hasAttribute('data-values')) {
                    try {
                        const valuesData = canvas.getAttribute('data-values');
                        if (valuesData && valuesData.trim() !== '') {
                            values = JSON.parse(valuesData);
                            
                            // Ensure all values are actually numbers
                            values = values.map(v => {
                                const num = parseFloat(v);
                                return isNaN(num) ? 0 : num;
                            });
                        }
                    } catch (parseError) {
                        console.error(`Error parsing values for ${canvas.id}:`, parseError);
                    }
                }
            } catch (e) {
                console.error('Error accessing data attributes:', e);
                return;
            }
            
            // Only create chart if we have valid data
            if (Array.isArray(labels) && Array.isArray(values) && labels.length > 0 && values.length > 0) {
                // Create the chart with error handling
                try {
                    new Chart(canvas, {
                        type: 'bar',
                        data: {
                            labels: labels,
                            datasets: [{
                                label: 'Monthly PnL',
                                data: values,
                                backgroundColor: values.map(value => value >= 0 ? 'rgba(40, 167, 69, 0.5)' : 'rgba(220, 53, 69, 0.5)'),
                                borderColor: values.map(value => value >= 0 ? '#28a745' : '#dc3545'),
                                borderWidth: 1
                            }]
                        },
                        options: {
                            responsive: true,
                            maintainAspectRatio: false,
                            scales: {
                                y: {
                                    title: {
                                        display: true,
                                        text: 'Profit/Loss'
                                    }
                                },
                                x: {
                                    title: {
                                        display: true,
                                        text: 'Month'
                                    }
                                }
                            }
                        }
                    });
                } catch (chartError) {
                    console.error(`Error creating chart for ${canvas.id}:`, chartError);
                    canvas.parentNode.innerHTML = '<div class="alert alert-danger">Error creating chart</div>';
                }
            } else {
                console.warn('Insufficient data for chart:', canvas.id);
                canvas.parentNode.innerHTML = '<div class="alert alert-warning">Insufficient data for chart</div>';
            }
        } catch (e) {
            console.error('Error creating chart:', e);
            canvas.parentNode.innerHTML = '<div class="alert alert-danger">Error creating chart</div>';
        }
    });
}

// Backend status checking functions
function checkBacktestStatus() {
    const statusUpdates = document.getElementById('status-updates');
    statusUpdates.classList.remove('d-none');
    
    fetch('/backtest_status')
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                if (data.status.length === 0) {
                    // No running backtests, reload the page
                    window.location.reload();
                } else {
                    // Some backtests are still running, check if any have finished
                    const completed = data.status.filter(job => job.status !== 'Running');
                    if (completed.length > 0) {
                        // Some backtests have finished, reload the page
                        window.location.reload();
                    }
                }
            }
            statusUpdates.classList.add('d-none');
        })
        .catch(error => {
            console.error('Error:', error);
            statusUpdates.classList.add('d-none');
        });
}

function cancelBacktest(jobId) {
    if (confirm("Are you sure you want to cancel this backtest?")) {
        fetch(`/cancel_backtest/${jobId}`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            }
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                window.location.reload();
            } else {
                alert('Error: ' + data.error);
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('Error: ' + error);
        });
    }
} 