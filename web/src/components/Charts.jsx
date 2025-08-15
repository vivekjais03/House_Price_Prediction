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
import { Bar, Scatter } from 'react-chartjs-2';

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

const Charts = ({ theme }) => {
  const [chartData, setChartData] = useState({
    featureImportance: null,
    priceDistribution: null,
    featureVsPrice: null,
    modelPerformance: null,
    sampleData: null
  });
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchChartData();
  }, []);

  const fetchChartData = async () => {
    try {
      setLoading(true);
      setError(null);

      const endpoints = [
        'feature-importance',
        'price-distribution',
        'feature-vs-price',
        'model-performance',
        'sample-data'
      ];

      const results = await Promise.all(
        endpoints.map(async (endpoint) => {
          try {
            const res = await fetch(`https://house-price-prediction-sgev.onrender.com/${endpoint}`);
            if (!res.ok) {
              console.warn(`Failed to fetch ${endpoint}:`, res.status);
              return { error: `HTTP ${res.status}` };
            }
            return await res.json();
          } catch (err) {
            console.warn(`Error fetching ${endpoint}:`, err.message);
            return { error: err.message };
          }
        })
      );

      setChartData({
        featureImportance: results[0]?.error ? null : results[0],
        priceDistribution: results[1]?.error ? null : results[1],
        featureVsPrice: results[2]?.error ? null : results[2],
        modelPerformance: results[3]?.error ? null : results[3],
        sampleData: results[4]?.error ? null : results[4]
      });
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="charts-container">
        <div className="loading">Loading charts...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="charts-container">
        <div className="error">
          <div>Error loading charts: {error}</div>
          <button 
            onClick={fetchChartData} 
            className="btn btn-primary"
            style={{ marginTop: '16px' }}
          >
            Retry
          </button>
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
      <h2 className="charts-title">Data Analysis & Visualizations</h2>
      
      <div className="charts-grid">
        {/* Feature Importance Chart */}
        {chartData.featureImportance?.labels && chartData.featureImportance?.data && (
          <div className="chart-card">
            <h3>Feature Importance</h3>
            <p>Which factors most affect house prices?</p>
            <div className="chart-wrapper">
              <Bar
                data={{
                  labels: chartData.featureImportance.labels,
                  datasets: [{
                    label: 'Importance Score',
                    data: chartData.featureImportance.data,
                    backgroundColor: 'rgba(34, 211, 238, 0.8)',
                    borderColor: 'rgba(34, 211, 238, 1)',
                    borderWidth: 1,
                  }]
                }}
                options={{
                  ...chartOptions,
                  plugins: {
                    ...chartOptions.plugins,
                    title: {
                      display: true,
                      text: 'Feature Importance Ranking',
                      color: theme === 'dark' ? '#e5e7eb' : '#0f172a',
                    },
                  },
                }}
              />
            </div>
          </div>
        )}

        {/* Price Distribution Chart */}
        {chartData.priceDistribution?.prices && chartData.priceDistribution.prices.length > 0 && (
          <div className="chart-card">
            <h3>Price Distribution</h3>
            <p>Distribution of house prices in the dataset</p>
            <div className="chart-wrapper">
              {(() => {
                const prices = chartData.priceDistribution.prices;
                const maxPrice = Math.max(...prices);
                const binSize = 500000; // $0.5M bins
                const numBins = Math.ceil(maxPrice / binSize);
                const labels = Array.from({ length: numBins }, (_, i) => 
                  `$${(i * 0.5).toFixed(1)}M - $${((i + 1) * 0.5).toFixed(1)}M`
                );
                const counts = Array.from({ length: numBins }, (_, i) => {
                  const minPrice = i * binSize;
                  const maxPriceRange = (i + 1) * binSize;
                  return prices.filter(price => price >= minPrice && price < maxPriceRange).length;
                });

                return (
                  <Bar
                    data={{
                      labels,
                      datasets: [{
                        label: 'Number of Houses',
                        data: counts,
                        backgroundColor: 'rgba(255, 99, 132, 0.8)',
                        borderColor: 'rgba(255, 99, 132, 1)',
                        borderWidth: 1,
                      }]
                    }}
                    options={{
                      ...chartOptions,
                      plugins: {
                        ...chartOptions.plugins,
                        title: {
                          display: true,
                          text: 'House Price Distribution',
                          color: theme === 'dark' ? '#e5e7eb' : '#0f172a',
                        },
                      },
                    }}
                  />
                );
              })()}
            </div>
          </div>
        )}

        {/* Model Performance Metrics */}
        {chartData.modelPerformance?.metrics && (
          <div className="chart-card">
            <h3>Model Performance</h3>
            <p>How well does our model predict house prices?</p>
            <div className="metrics-grid">
              <div className="metric">
                <span className="metric-label">R² Score</span>
                <span className="metric-value">
                  {(chartData.modelPerformance.metrics?.r2_score ?? 0 * 100).toFixed(1)}%
                </span>
              </div>
              <div className="metric">
                <span className="metric-label">MAE</span>
                <span className="metric-value">
                  ${(chartData.modelPerformance.metrics?.mae ?? 0 * 100000).toFixed(0)}
                </span>
              </div>
              <div className="metric">
                <span className="metric-label">RMSE</span>
                <span className="metric-value">
                  ${(chartData.modelPerformance.metrics?.rmse ?? 0 * 100000).toFixed(0)}
                </span>
              </div>
            </div>
            {chartData.modelPerformance?.predictions_vs_actual?.actual && (
              <div className="chart-wrapper">
                <Scatter
                  data={{
                    datasets: [
                      {
                        label: 'Actual vs Predicted',
                        data: chartData.modelPerformance.predictions_vs_actual.actual.map((actual, i) => ({
                          x: actual,
                          y: chartData.modelPerformance.predictions_vs_actual.predicted[i]
                        })),
                        backgroundColor: 'rgba(75, 192, 192, 0.6)',
                        borderColor: 'rgba(75, 192, 192, 1)',
                      }
                    ]
                  }}
                  options={{
                    ...chartOptions,
                    plugins: {
                      ...chartOptions.plugins,
                      title: {
                        display: true,
                        text: 'Actual vs Predicted Prices',
                        color: theme === 'dark' ? '#e5e7eb' : '#0f172a',
                      },
                    },
                    scales: {
                      x: {
                        title: {
                          display: true,
                          text: 'Actual Price (USD)',
                          color: theme === 'dark' ? '#e5e7eb' : '#0f172a',
                        },
                        ...chartOptions.scales.x,
                      },
                      y: {
                        title: {
                          display: true,
                          text: 'Predicted Price (USD)',
                          color: theme === 'dark' ? '#e5e7eb' : '#0f172a',
                        },
                        ...chartOptions.scales.y,
                      },
                    },
                  }}
                />
              </div>
            )}
          </div>
        )}

        {/* Feature vs Price Scatter Plots */}
        {chartData.featureVsPrice && Object.keys(chartData.featureVsPrice).length > 0 && (
          <div className="chart-card">
            <h3>Feature vs Price Relationships</h3>
            <p>How individual features relate to house prices</p>
            <div className="feature-charts">
              {Object.entries(chartData.featureVsPrice).map(([feature, data]) => (
                <div key={feature} className="feature-chart">
                  <h4>{feature}</h4>
                  <div className="chart-wrapper small">
                    <Scatter
                      data={{
                        datasets: [{
                          label: `${feature} vs Price`,
                          data: data?.x?.map((x, i) => ({ x, y: data?.y?.[i] })) ?? [],
                          backgroundColor: 'rgba(255, 159, 64, 0.6)',
                          borderColor: 'rgba(255, 159, 64, 1)',
                        }]
                      }}
                      options={{
                        ...chartOptions,
                        plugins: {
                          ...chartOptions.plugins,
                          legend: { display: false },
                        },
                        scales: {
                          x: {
                            title: {
                              display: true,
                              text: feature,
                              color: theme === 'dark' ? '#e5e7eb' : '#0f172a',
                            },
                            ...chartOptions.scales.x,
                          },
                          y: {
                            title: {
                              display: true,
                              text: 'Price (USD)',
                              color: theme === 'dark' ? '#e5e7eb' : '#0f172a',
                            },
                            ...chartOptions.scales.y,
                          },
                        },
                      }}
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default Charts;
