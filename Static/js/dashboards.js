async function getPrediction() {
    const symbol = document.getElementById('searchInput').value;

    if (!symbol) {
        alert('Please enter a company symbol!');
        return;
    }

    try {
        // Send a POST request to the Flask route
        const response = await fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ symbol: symbol }),
        });

        const data = await response.json();

        if (data.status === 'success') {
            // Update the range
            const predictedRange = data.predicted_range;
            document.getElementById('resultsContainer').innerHTML = `
                <h4>Predicted Range: ${predictedRange}</h4>
                <canvas id="stockChart"></canvas>
            `;

            // Plot the graph
            const ctx = document.getElementById('stockChart').getContext('2d');
            new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.plot_data.dates,
                    datasets: [{
                        label: 'Actual Prices',
                        data: data.plot_data.actual_prices,
                        borderColor: 'blue',
                        fill: false,
                    }, {
                        label: 'Predicted Price',
                        data: [data.plot_data.predicted_price],
                        borderColor: 'green',
                        fill: false,
                    }]
                },
                options: {
                    responsive: true,
                    maintainAspectRatio: false,
                },
            });
        } else {
            alert(data.message);
        }
    } catch (error) {
        console.error('Error fetching prediction:', error);
        alert('Failed to fetch prediction. Please try again.');
    }
}
