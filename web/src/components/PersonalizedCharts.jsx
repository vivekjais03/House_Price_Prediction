import React, { useState, useEffect } from 'react';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement,
} from 'chart.js';
import { Bar, Line, Scatter, Doughnut } from 'react-chartjs-2';

ChartJS.register(
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  BarElement,
  Title,
  Tooltip,
  Legend,
  ArcElement
);

const PersonalizedCharts = ({ theme, userData, onAnalysisComplete }) => {
  const [analysisData, setAnalysisData] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    if (userData && Object.values(userData).every(val => val !== '')) {
      fetchPersonalizedAnalysis();
    }
  }, [userData]);

  const fetchPersonalizedAnalysis = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await fetch('https://house-price-prediction-sgev.onrender.com/personalized-analytics', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(userData)
      });

      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }

      const data = await response.json();
      setAnalysisData(data);
      
      // Notify parent component that analysis is complete
      if (onAnalysisComplete) {
        onAnalysisComplete(data);
      }
    } catch (err) {
      setError(err.message);
      console.error('Error fetching personalized analysis:', err);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="charts-container">
        <div className="loading">
          <div>🔍 Analyzing your house data...</div>
          <div style={{ fontSize: '14px', marginTop: '8px', color: 'var(--muted)' }}>
            Finding similar houses and calculating price sensitivity
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="charts-container">
        <div className="error">
          <div>Error analyzing your data: {error}</div>
          <button 
            onClick={fetchPersonalizedAnalysis} 
            className="btn btn-primary"
            style={{ marginTop: '16px' }}
          >
            Retry Analysis
          </button>
        </div>
      </div>
    );
  }

  if (!analysisData) {
    return (
      <div className="charts-container">
        <div className="info-message">
          <div>📊 Enter your house details above to see personalized analytics</div>
          <div style={{ fontSize: '14px', marginTop: '8px', color: 'var(--muted)' }}>
            Charts will show how your house compares to similar properties
          </div>
        </div>
      </div>
    );
  }

  const chartOptions = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
      legend: {
        labels: {
          color: theme === 'dark' ? '#e5e7eb' : '#0f172a',
        },
      },
    },
    scales: {
      x: {
        ticks: {
          color: theme === 'dark' ? '#e5e7eb' : '#0f172a',
        },
        grid: {
          color: theme === 'dark' ? '#374151' : '#e2e8f0',
        },
      },
      y: {
        ticks: {
          color: theme === 'dark' ? '#e5e7eb' : '#0f172a',
        },
        grid: {
          color: theme === 'dark' ? '#374151' : '#e2e8f0',
        },
      },
    },
  };

  return (
    <div className="charts-container">
      <h2 className="charts-title">🏠 Your House: Personalized Analysis</h2>
      
      <div className="charts-grid">
        {/* Your Price vs Similar Houses */}
        {analysisData.similar_houses.prices.length > 0 && (
          <div className="chart-card">
            <h3>Your Price vs Similar Houses</h3>
            <p>How your predicted price compares to similar properties</p>
            
            <div className="price-summary">
              <div className="price-item">
                <span className="price-label">Your Predicted Price:</span>
                <span className="price-value">${analysisData.user_prediction.predicted_price_usd.toLocaleString()}</span>
              </div>
              <div className="price-item">
                <span className="price-label">Similar Houses Found:</span>
                <span className="price-value">{analysisData.similar_houses.count}</span>
              </div>
              <div className="price-item">
                <span className="price-label">Price Percentile:</span>
                <span className="price-value">{analysisData.user_prediction.percentile}%</span>
              </div>
            </div>

            <div className="chart-wrapper">
              <Bar
                data={{
                  labels: ['Your House', 'Similar Houses (Min)', 'Similar Houses (Avg)', 'Similar Houses (Max)'],
                  datasets: [{
                    label: 'Price (USD)',
                    data: [
                      analysisData.user_prediction.predicted_price_usd,
                      analysisData.similar_houses.price_range.min,
                      analysisData.similar_houses.price_range.average,
                      analysisData.similar_houses.price_range.max
                    ],
                    backgroundColor: [
                      'rgba(34, 211, 238, 0.8)',  // Your house - cyan
                      'rgba(255, 99, 132, 0.6)',   // Min - pink
                      'rgba(75, 192, 192, 0.6)',   // Avg - teal
                      'rgba(255, 159, 64, 0.6)'    // Max - orange
                    ],
                    borderColor: [
                      'rgba(34, 211, 238, 1)',
                      'rgba(255, 99, 132, 1)',
                      'rgba(75, 192, 192, 1)',
                      'rgba(255, 159, 64, 1)'
                    ],
                    borderWidth: 2,
                  }]
                }}
                options={{
                  ...chartOptions,
                  plugins: {
                    ...chartOptions.plugins,
                    title: {
                      display: true,
                      text: 'Price Comparison',
                      color: theme === 'dark' ? '#e5e7eb' : '#0f172a',
                    },
                  },
                }}
              />
            </div>
          </div>
        )}

        {/* Feature Comparison Chart */}
        <div className="chart-card">
          <h3>How Your Features Compare</h3>
          <p>Your house features vs. dataset averages</p>
          
          <div className="chart-wrapper">
            <Bar
              data={{
                labels: Object.keys(analysisData.feature_comparison),
                datasets: [
                  {
                    label: 'Your Values',
                    data: Object.values(analysisData.feature_comparison).map(f => f.user_value),
                    backgroundColor: 'rgba(34, 211, 238, 0.8)',
                    borderColor: 'rgba(34, 211, 238, 1)',
                    borderWidth: 2,
                  },
                  {
                    label: 'Dataset Average',
                    data: Object.values(analysisData.feature_comparison).map(f => f.dataset_average),
                    backgroundColor: 'rgba(255, 99, 132, 0.6)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 2,
                  }
                ]
              }}
              options={{
                ...chartOptions,
                plugins: {
                  ...chartOptions.plugins,
                  title: {
                    display: true,
                    text: 'Feature Comparison',
                    color: theme === 'dark' ? '#e5e7eb' : '#0f172a',
                  },
                },
              }}
            />
          </div>
        </div>

        {/* Price Sensitivity Analysis */}
        <div className="chart-card">
          <h3>Price Sensitivity Analysis</h3>
          <p>How much each feature affects your house price</p>
          
          <div className="chart-wrapper">
            <Bar
              data={{
                labels: Object.keys(analysisData.price_sensitivity),
                datasets: [{
                  label: 'Price Change (%)',
                  data: Object.values(analysisData.price_sensitivity).map(f => f.price_change_percent),
                  backgroundColor: Object.values(analysisData.price_sensitivity).map(f => {
                    if (f.sensitivity === 'high') return 'rgba(255, 99, 132, 0.8)';
                    if (f.sensitivity === 'medium') return 'rgba(255, 159, 64, 0.8)';
                    return 'rgba(75, 192, 192, 0.8)';
                  }),
                  borderColor: Object.values(analysisData.price_sensitivity).map(f => {
                    if (f.sensitivity === 'high') return 'rgba(255, 99, 132, 1)';
                    if (f.sensitivity === 'medium') return 'rgba(255, 159, 64, 1)';
                    return 'rgba(75, 192, 192, 1)';
                  }),
                  borderWidth: 2,
                }]
              }}
              options={{
                ...chartOptions,
                plugins: {
                  ...chartOptions.plugins,
                  title: {
                    display: true,
                    text: 'Price Sensitivity (10% feature change)',
                    color: theme === 'dark' ? '#e5e7eb' : '#0f172a',
                  },
                },
              }}
            />
          </div>
          
          <div className="sensitivity-legend">
            <div className="legend-item">
              <span className="legend-color high"></span>
              <span>High Sensitivity (&gt;5% price change)</span>
            </div>
            <div className="legend-item">
              <span className="legend-color medium"></span>
              <span>Medium Sensitivity (2-5% price change)</span>
            </div>
            <div className="legend-item">
              <span className="legend-color low"></span>
              <span>Low Sensitivity (&lt;2% price change)</span>
            </div>
          </div>
        </div>

        {/* Similar Houses Price Distribution */}
        {analysisData.similar_houses.prices.length > 0 && (
          <div className="chart-card">
            <h3>Similar Houses Price Distribution</h3>
            <p>Price range of houses with similar features to yours</p>
            
            <div className="chart-wrapper">
              <Bar
                data={{
                  labels: Array.from({ length: 10 }, (_, i) => {
                    const minPrice = i * (analysisData.similar_houses.price_range.max - analysisData.similar_houses.price_range.min) / 10 + analysisData.similar_houses.price_range.min;
                    const maxPrice = (i + 1) * (analysisData.similar_houses.price_range.max - analysisData.similar_houses.price_range.min) / 10 + analysisData.similar_houses.price_range.min;
                    return `$${(minPrice / 1000).toFixed(0)}k - $${(maxPrice / 1000).toFixed(0)}k`;
                  }),
                  datasets: [{
                    label: 'Number of Houses',
                    data: Array.from({ length: 10 }, (_, i) => {
                      const minPrice = i * (analysisData.similar_houses.price_range.max - analysisData.similar_houses.price_range.min) / 10 + analysisData.similar_houses.price_range.min;
                      const maxPrice = (i + 1) * (analysisData.similar_houses.price_range.max - analysisData.similar_houses.price_range.min) / 10 + analysisData.similar_houses.price_range.min;
                      return analysisData.similar_houses.prices.filter(
                        price => price >= minPrice && price < maxPrice
                      ).length;
                    }),
                    backgroundColor: 'rgba(75, 192, 192, 0.6)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 1,
                  }]
                }}
                options={{
                  ...chartOptions,
                  plugins: {
                    ...chartOptions.plugins,
                    title: {
                      display: true,
                      text: 'Similar Houses Price Distribution',
                      color: theme === 'dark' ? '#e5e7eb' : '#0f172a',
                    },
                  },
                }}
              />
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default PersonalizedCharts; 
