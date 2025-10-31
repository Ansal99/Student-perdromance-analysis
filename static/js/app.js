const API_BASE_URL = 'http://localhost:5000/api';

document.addEventListener('DOMContentLoaded', function() {
    // Fetch health status and update banner
    fetchHealthBanner();
    // Initialize chat functionality
    const chatForm = document.getElementById('chat-form');
    const chatInput = document.getElementById('chat-input');
    const chatMessages = document.getElementById('chat-messages');
    
    if (chatForm) {
        chatForm.addEventListener('submit', async function(e) {
            e.preventDefault();
            const message = chatInput.value.trim();
            if (!message) return;
            
            // Add user message to chat
            addMessage(message, 'user');
            chatInput.value = '';
            
            // Show thinking indicator
            const thinkingMsg = addThinkingIndicator();
            
            try {
                const response = await fetch('/api/chat', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ message })
                });

                // Try to parse JSON response (server may return an error payload)
                let data = null;
                try {
                    data = await response.json();
                } catch (parseErr) {
                    console.error('Failed to parse JSON from /api/chat:', parseErr);
                    thinkingMsg.remove();
                    addMessage('Sorry, could not understand server response. Check the server logs.', 'assistant');
                    return;
                }

                // Remove thinking indicator
                thinkingMsg.remove();

                // If HTTP status not ok, show server-provided error message when available
                if (!response.ok) {
                    const serverErr = (data && data.error) ? data.error : `Server error (status ${response.status})`;
                    addMessage(`Error: ${serverErr}`, 'assistant');
                    return;
                }

                // Normal flow: server replied with JSON
                if (data && data.success) {
                    addMessage(data.response, 'assistant');
                } else {
                    const serverErr = (data && data.error) ? data.error : 'Sorry, I encountered an error. Please try again.';
                    addMessage(serverErr, 'assistant');
                }
            } catch (error) {
                console.error('Chat network/error:', error);
                thinkingMsg.remove();
                addMessage('Network error while contacting AI service: ' + (error.message || error), 'assistant');
            }
        });
    }
    
    function addMessage(content, type) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${type}`;
        messageDiv.textContent = content;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return messageDiv;
    }
    
    function addThinkingIndicator() {
        const thinkingDiv = document.createElement('div');
        thinkingDiv.className = 'message assistant thinking';
        thinkingDiv.innerHTML = `
            Thinking
            <div class="thinking-dots">
                <div class="thinking-dot"></div>
                <div class="thinking-dot"></div>
                <div class="thinking-dot"></div>
            </div>
        `;
        chatMessages.appendChild(thinkingDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return thinkingDiv;
    }
    initializeTabs();
    initializePredictionForm();
    initializeChartControls();
    initializeModelInfo();
    initializeRecords();
    initializeInputTracking();
});

async function fetchHealthBanner() {
    const banner = document.getElementById('health-banner');
    if (!banner) return;

    try {
        const res = await fetch('/api/health');
        const data = await res.json();

        if (data && data.success) {
            const parts = [];
            if (data.openai_configured) parts.push('OpenAI: configured');
            else parts.push('OpenAI: not configured');

            parts.push(`Supabase: ${data.supabase_initialized ? 'connected' : 'not connected'}`);

            banner.textContent = parts.join(' • ');
            banner.style.display = 'block';
            banner.className = 'health-banner ' + (data.openai_configured ? 'ok' : 'warn');
        } else {
            banner.textContent = 'Health check failed';
            banner.style.display = 'block';
            banner.className = 'health-banner warn';
        }
    } catch (err) {
        console.error('Health check failed', err);
        banner.textContent = 'Health check failed (network)';
        banner.style.display = 'block';
        banner.className = 'health-banner warn';
    }
}

function initializeTabs() {
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabContents = document.querySelectorAll('.tab-content');

    tabButtons.forEach(button => {
        button.addEventListener('click', () => {
            const tabName = button.getAttribute('data-tab');

            tabButtons.forEach(btn => btn.classList.remove('active'));
            tabContents.forEach(content => content.classList.remove('active'));

            button.classList.add('active');
            document.getElementById(`${tabName}-tab`).classList.add('active');
        });
    });
}

function initializeInputTracking() {
    const attendanceInput = document.getElementById('attendance');
    const studyHoursInput = document.getElementById('study_hours');
    const previousScoreInput = document.getElementById('previous_score');

    attendanceInput.addEventListener('input', (e) => {
        document.getElementById('attendance-value').textContent = e.target.value || '0';
    });

    studyHoursInput.addEventListener('input', (e) => {
        document.getElementById('study-hours-value').textContent = e.target.value || '0';
    });

    previousScoreInput.addEventListener('input', (e) => {
        document.getElementById('previous-score-value').textContent = e.target.value || '0';
    });
}

function initializePredictionForm() {
    const form = document.getElementById('prediction-form');
    const btnText = form.querySelector('.btn-text');
    const btnLoader = form.querySelector('.btn-loader');
    const resultDiv = document.getElementById('prediction-result');

    form.addEventListener('submit', async (e) => {
        e.preventDefault();

        btnText.style.display = 'none';
        btnLoader.style.display = 'block';
        form.querySelector('button').disabled = true;

        const formData = {
            student_name: document.getElementById('student_name').value,
            attendance: parseFloat(document.getElementById('attendance').value),
            study_hours: parseFloat(document.getElementById('study_hours').value),
            previous_score: parseFloat(document.getElementById('previous_score').value)
        };

        try {
            const response = await fetch(`${API_BASE_URL}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(formData)
            });

            const data = await response.json();

            if (data.success) {
                displayPredictionResult(data);
                resultDiv.style.display = 'block';
                resultDiv.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            } else {
                alert('Error: ' + data.error);
            }
        } catch (error) {
            alert('Network error: ' + error.message);
        } finally {
            btnText.style.display = 'inline';
            btnLoader.style.display = 'none';
            form.querySelector('button').disabled = false;
        }
    });
}

function displayPredictionResult(data) {
    document.getElementById('predicted-score').textContent = data.prediction;

    const riskBadge = document.getElementById('risk-badge');
    riskBadge.className = `risk-badge ${data.risk_color}`;
    document.getElementById('risk-level').textContent = data.risk_level;

    const recommendationsList = document.getElementById('recommendations-list');
    recommendationsList.innerHTML = '';
    data.recommendations.forEach(rec => {
        const li = document.createElement('li');
        li.textContent = rec;
        recommendationsList.appendChild(li);
    });
}

function initializeChartControls() {
    const chartButtons = document.querySelectorAll('.chart-btn');
    const chartLoader = document.getElementById('chart-loader');
    const chartImage = document.getElementById('chart-image');

    chartButtons.forEach(button => {
        button.addEventListener('click', async () => {
            const chartType = button.getAttribute('data-chart');

            chartButtons.forEach(btn => btn.classList.remove('active'));
            button.classList.add('active');

            chartLoader.style.display = 'flex';
            chartImage.style.display = 'none';

            try {
                const response = await fetch(`${API_BASE_URL}/visualize`, {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json'
                    },
                    body: JSON.stringify({ chart_type: chartType })
                });

                const data = await response.json();

                if (data.success) {
                    chartImage.src = data.image;
                    chartImage.style.display = 'block';
                    chartLoader.style.display = 'none';
                } else {
                    alert('Error loading chart: ' + data.error);
                    chartLoader.style.display = 'none';
                }
            } catch (error) {
                alert('Network error: ' + error.message);
                chartLoader.style.display = 'none';
            }
        });
    });

    chartButtons[0].click();
}

function initializeModelInfo() {
    const loadButton = document.getElementById('load-model-info');
    const modelInfoDiv = document.getElementById('model-info');

    loadButton.addEventListener('click', async () => {
        loadButton.disabled = true;
        loadButton.textContent = 'Loading...';

        try {
            const response = await fetch(`${API_BASE_URL}/model-info`);
            const data = await response.json();

            if (data.success) {
                document.getElementById('r2-score').textContent = data.r2_score;
                document.getElementById('rmse').textContent = data.rmse;

                document.getElementById('coef-attendance').textContent = data.coefficients.attendance;
                document.getElementById('coef-study-hours').textContent = data.coefficients.study_hours;
                document.getElementById('coef-previous-score').textContent = data.coefficients.previous_score;
                document.getElementById('coef-intercept').textContent = data.coefficients.intercept;

                document.getElementById('eq-attendance').textContent = data.coefficients.attendance;
                document.getElementById('eq-study-hours').textContent = data.coefficients.study_hours;
                document.getElementById('eq-previous-score').textContent = data.coefficients.previous_score;
                document.getElementById('eq-intercept').textContent = data.coefficients.intercept;

                modelInfoDiv.style.opacity = '1';
                loadButton.textContent = 'Refresh Model Information';
            } else {
                alert('Error loading model info: ' + data.error);
                loadButton.textContent = 'Load Model Information';
            }
        } catch (error) {
            alert('Network error: ' + error.message);
            loadButton.textContent = 'Load Model Information';
        } finally {
            loadButton.disabled = false;
        }
    });
}

function initializeRecords() {
    const loadButton = document.getElementById('load-records');
    const tableContainer = document.getElementById('records-table-container');
    const tbody = document.getElementById('records-tbody');

    loadButton.addEventListener('click', async () => {
        loadButton.disabled = true;
        loadButton.textContent = 'Loading...';

        try {
            const response = await fetch(`${API_BASE_URL}/students`);
            const data = await response.json();

            if (data.success) {
                tbody.innerHTML = '';

                if (data.students.length === 0) {
                    tbody.innerHTML = '<tr><td colspan="7" style="text-align: center; padding: 2rem; color: #64748b;">No records found. Make a prediction first!</td></tr>';
                } else {
                    data.students.forEach(student => {
                        const row = document.createElement('tr');

                        const riskColorClass = student.risk_level === 'Low Risk' ? 'success' :
                                              student.risk_level === 'Medium Risk' ? 'warning' : 'danger';

                        const date = new Date(student.created_at).toLocaleDateString('en-US', {
                            year: 'numeric',
                            month: 'short',
                            day: 'numeric'
                        });

                        row.innerHTML = `
                            <td>${student.student_name}</td>
                            <td>${student.attendance}%</td>
                            <td>${student.study_hours} hrs</td>
                            <td>${student.previous_score}</td>
                            <td><strong>${student.predicted_score}</strong></td>
                            <td><span class="risk-cell ${riskColorClass}">${student.risk_level}</span></td>
                            <td>${date}</td>
                        `;

                        tbody.appendChild(row);
                    });
                }

                tableContainer.style.display = 'block';
                loadButton.textContent = 'Refresh Records';
            } else {
                alert('Error loading records: ' + data.error);
                loadButton.textContent = 'Load Recent Records';
            }
        } catch (error) {
            alert('Network error: ' + error.message);
            loadButton.textContent = 'Load Recent Records';
        } finally {
            loadButton.disabled = false;
        }
    });
}
