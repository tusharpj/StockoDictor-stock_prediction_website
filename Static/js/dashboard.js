async function getPrediction() {
    const company = document.getElementById("searchInput").value; 
  
    try {
      const response = await fetch('http://127.0.0.1:5000/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json'
        },
        body: JSON.stringify({ company: company }) 
      });
  
      const result = await response.json(); 
      const resultsContainer = document.getElementById('resultsContainer');
      resultsContainer.innerHTML = ''; // Clear previous results
  
      // Check if there's an error in the response
      if (result.error) {
        resultsContainer.innerHTML = `<p class="text-danger">${result.error}</p>`;
        return;
      }
      
      // Display the prediction
      const predictionElement = document.createElement('p');
      predictionElement.textContent = `Company: ${result.company}, Prediction: ${result.prediction}`;
      resultsContainer.appendChild(predictionElement);
  
      // Create and display the chart
      displayChart(result.plot_data);
  
    } catch (error) {
      console.error('Error:', error);
      document.getElementById("resultsContainer").innerText = "Error fetching prediction!";
    }
  }
  
  function displayChart(plotData) {
      const canvas = document.getElementById('stockChart');
      const ctx = canvas.getContext('2d');
  
      // Check if there's an existing chart and destroy it
      let existingChart = Chart.getChart(ctx);
      if (existingChart) {
          existingChart.destroy();
      }
  
      new Chart(ctx, {
          type: 'line',
          data: {
              labels: plotData.dates,
              datasets: [{
                  label: 'Actual Prices',
                  data: plotData.actual_prices,
                  borderColor: 'blue',
                  fill: false
              }, {
                  label: 'Predicted Price',
                  data: Array(plotData.dates.length - 1).fill(null).concat([plotData.predicted_price]),
                  borderColor: 'red',
                  fill: false
              }]
          },
          options: {
              responsive: true,
              title: {
                  display: true,
                  text: 'Stock Price Prediction'
              },
              scales: {
                  x: {
                      display: true,
                      title: {
                          display: true,
                          text: 'Date'
                      }
                  },
                  y: {
                      display: true,
                      title: {
                          display: true,
                          text: 'Price'
                      }
                  }
              }
          }
      });
  }