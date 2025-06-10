let selectedFeatures = [];

// Load selected features from backend
async function loadSelectedFeatures() {
    try {
        const response = await fetch('/selected_features');
        const data = await response.json();
        selectedFeatures = data.features;
        createFormFields();
    } catch (error) {
        console.error('Error loading selected features:', error);
    }
}

function createFormFields() {
    const formFields = document.getElementById('formFields');
    const fieldsPerSection = Math.ceil(selectedFeatures.length / 3);

    let html = '<div class="form-grid">';

    for (let section = 0; section < 3; section++) {
        const startIdx = section * fieldsPerSection;
        const endIdx = Math.min(startIdx + fieldsPerSection, selectedFeatures.length);
        const sectionFeatures = selectedFeatures.slice(startIdx, endIdx);

        if (sectionFeatures.length === 0) continue;

        html += `
            <div class="form-section">
                <h3>📏 Features Group ${section + 1}</h3>
        `;

        sectionFeatures.forEach(feature => {
            const displayName = feature.split(' ').map(word =>
                word.charAt(0).toUpperCase() + word.slice(1)
            ).join(' ');

            html += `
                <div class="form-group">
                    <label>${displayName}:</label>
                    <input type="number" step="any" name="${feature}" required>
                </div>
            `;
        });

        html += '</div>';
    }

    html += '</div>';
    formFields.innerHTML = html;
}

async function loadSampleData() {
    try {
        showLoading(true);
        const response = await fetch('/sample_data');
        const data = await response.json();

        // Fill form with sample data
        const form = document.getElementById('predictionForm');
        Object.keys(data.features).forEach(key => {
            const input = form.querySelector(`input[name="${key}"]`);
            if (input) {
                input.value = parseFloat(data.features[key]).toFixed(4);
            }
        });

        showLoading(false);
        alert(`✅ Sample data loaded successfully!\n\nActual diagnosis: ${data.actual_diagnosis}`);
    } catch (error) {
        showLoading(false);
        alert('❌ Error loading sample data: ' + error.message);
    }
}

function clearForm() {
    document.getElementById('predictionForm').reset();
    document.getElementById('result').style.display = 'none';
}

function showSelectedFeatures() {
    const featureList = selectedFeatures.map((feature, index) =>
        `${index + 1}. ${feature}`
    ).join('\n');
    alert(`📋 Selected Features (15 total):\n\n${featureList}`);
}

function showLoading(show) {
    document.getElementById('loading').style.display = show ? 'block' : 'none';
}

function showResult(result) {
    const resultDiv = document.getElementById('result');
    const className = result.prediction.toLowerCase();

    resultDiv.className = `result ${className}`;
    resultDiv.innerHTML = `
        <h3>🔬 ${result.model_used} Result: ${result.prediction}</h3>
        <div class="result-stats">
            <div class="stat-card">
                <div class="stat-value">${result.confidence}%</div>
                <div class="stat-label">Confidence</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${result.probability_benign}%</div>
                <div class="stat-label">Benign Probability</div>
            </div>
            <div class="stat-card">
                <div class="stat-value">${result.probability_malignant}%</div>
                <div class="stat-label">Malignant Probability</div>
            </div>
        </div>
        <p style="margin-top: 15px; font-style: italic;">
            ${result.prediction === 'Benign' ? 
                '✅ The analysis suggests this is likely a benign (non-cancerous) tumor.' : 
                '⚠️ The analysis suggests this may be a malignant (cancerous) tumor. Please consult with a healthcare professional immediately.'}
        </p>
    `;
    resultDiv.style.display = 'block';
}

document.getElementById('predictionForm').onsubmit = async function(e) {
    e.preventDefault();

    showLoading(true);
    document.getElementById('result').style.display = 'none';

    const formData = new FormData(e.target);
    const data = {};

    for (let [key, value] of formData.entries()) {
        data[key] = parseFloat(value) || 0;
    }

    // Get selected model
    const selectedModel = document.querySelector('input[name="model"]:checked').value;
    data.model_type = selectedModel;

    try {
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(data)
        });

        const result = await response.json();

        showLoading(false);

        if (result.error) {
            alert('❌ Error: ' + result.error);
        } else {
            showResult(result);
        }
    } catch (error) {
        showLoading(false);
        alert('❌ Error making prediction: ' + error.message);
    }
};

// Initialize the form when page loads
document.addEventListener('DOMContentLoaded', function() {
    loadSelectedFeatures();
});