import React, { useState, useEffect } from 'react';
import Charts from './components/Charts';
import PersonalizedCharts from './components/PersonalizedCharts';
import './styles.css';

const API_URL = 'https://house-price-prediction-sgev.onrender.com'

const initialValues = {
	MedInc: '',
	HouseAge: '',
	AveRooms: '',
	AveBedrms: '',
	Population: '',
	AveOccup: '',
	Latitude: '',
	Longitude: ''
}

const defaultRates = {
	USD: 1,
	INR: 83,
	EUR: 0.92,
	GBP: 0.78,
	JPY: 146,
	AED: 3.67,
}

const currencySymbols = {
	USD: '$',
	INR: '₹',
	EUR: '€',
	GBP: '£',
	JPY: '¥',
	AED: 'AED ',
}

function App() {
	const [values, setValues] = useState(initialValues)
	const [loading, setLoading] = useState(false)
	const [error, setError] = useState('')
	const [result, setResult] = useState(null)
	const [currency, setCurrency] = useState('INR')
	const [rates, setRates] = useState(defaultRates)
	const [fxLive, setFxLive] = useState(false)
	const [indianUnits, setIndianUnits] = useState(true) // Lakh/Crore display when INR
	const [country, setCountry] = useState('IN') // IN or US-CA
	const [theme, setTheme] = useState('light')
	const [activeTab, setActiveTab] = useState('prediction');

	useEffect(() => {
		// load font
		const link = document.createElement('link')
		link.href = 'https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700;800&display=swap'
		link.rel = 'stylesheet'
		document.head.appendChild(link)
		return () => { document.head.removeChild(link) }
	}, [])

	useEffect(() => {
		document.documentElement.setAttribute('data-theme', theme)
	}, [theme])

	useEffect(() => {
		// Fetch latest FX rates with base USD
		(async () => {
			try {
				const res = await fetch('https://api.exchangerate.host/latest?base=USD')
				if (!res.ok) return
				const data = await res.json()
				if (data && data.rates) {
					setRates(prev => ({
						...prev,
						USD: 1,
						INR: data.rates.INR ?? prev.INR,
						EUR: data.rates.EUR ?? prev.EUR,
						GBP: data.rates.GBP ?? prev.GBP,
						JPY: data.rates.JPY ?? prev.JPY,
						AED: data.rates.AED ?? prev.AED,
					}))
					setFxLive(true)
				}
			} catch (_) {
				// ignore; keep defaults
			}
		})()
	}, [])

	useEffect(() => {
		// Auto-enable Indian units when INR selected
		setIndianUnits(currency === 'INR')
	}, [currency])

	function handleChange(e) {
		const { name, value } = e.target
		setValues(v => ({ ...v, [name]: value }))
	}

	async function handleSubmit(e) {
		e.preventDefault()
		setError('')
		setResult(null)
		setLoading(true)
		try {
			// Validate all fields are numeric
			const payload = {}
			for (const key of Object.keys(values)) {
				const num = Number(values[key])
				if (Number.isNaN(num)) {
					throw new Error(`${key} must be a number`)
				}
				payload[key] = num
			}

			const res = await fetch(`${API_URL}/predict`, {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(payload)
			})
			
			if (!res.ok) {
				const text = await res.text()
				throw new Error(`API error: ${res.status} ${text}`)
			}
			
			const data = await res.json()
			setResult(data)
		} catch (err) {
			setError(err.message || 'Something went wrong')
		} finally {
			setLoading(false)
		}
	}

	function resetForm() {
		setValues(initialValues)
		setResult(null)
		setError('')
	}

	const selectedSymbol = currencySymbols[currency] || ''
	const usdAmount = result?.predicted_price_usd ?? 0
	const convertedAmount = usdAmount * (rates[currency] ?? 1)
	const exchangeRate = rates[currency] ?? 1

	// Formatting helpers
	const nfUS = new Intl.NumberFormat('en-US')
	const nfIN = new Intl.NumberFormat('en-IN')

	function formatWithUnits(amount, curr) {
		// amount is in selected currency (not USD)
		if (curr === 'INR') {
			if (!indianUnits) return `${currencySymbols.INR}${nfIN.format(Math.round(amount))}`
			const crore = 10000000
			const lakh = 100000
			if (amount >= crore) return `${currencySymbols.INR}${(amount / crore).toFixed(2)} Cr`
			if (amount >= lakh) return `${currencySymbols.INR}${(amount / lakh).toFixed(2)} Lakh`
			return `${currencySymbols.INR}${nfIN.format(Math.round(amount))}`
		}
		// International: show Millions if >= 1,000,000
		if (amount >= 1_000_000) return `${currencySymbols[curr] ?? ''}${(amount / 1_000_000).toFixed(2)} M`
		return `${currencySymbols[curr] ?? ''}${nfUS.format(Math.round(amount))}`
	}

	// Dynamic helper for MedInc: dataset unit is USD in tens of thousands
	const medIncVal = Number(values.MedInc)
	const medIncUSD = !Number.isNaN(medIncVal) ? medIncVal * 10000 : 0
	const medIncINR = medIncUSD * (rates.INR ?? 83)

	// Dynamic placeholders and help by country
	const latHelp = country === 'IN' ? 'Degrees (India approx. 6–37)' : 'Degrees (California approx. 32–42)'
	const lonHelp = country === 'IN' ? 'Degrees (India approx. 68–97)' : 'Degrees (California approx. -124 to -114)'
	const latPlaceholder = country === 'IN' ? 'e.g., 28.61' : 'e.g., 37.88'
	const lonPlaceholder = country === 'IN' ? 'e.g., 77.21' : 'e.g., -122.23'

	return (
		<div className="container" style={{ width: '100%', maxWidth: '100%' }}>
			<div className="topbar">
				<h1 className="title">House Price Prediction</h1>
				<div className="topbar-controls">
					<div className="tab-buttons">
						<button 
							type="button"
							className={`tab-button ${activeTab === 'prediction' ? 'active' : ''}`}
							onClick={() => setActiveTab('prediction')}
						>
							📊 Prediction
						</button>
						<button 
							type="button"
							className={`tab-button ${activeTab === 'charts' ? 'active' : ''}`}
							onClick={() => setActiveTab('charts')}
						>
							📈 Analytics
						</button>
						<button 
							type="button"
							className={`tab-button ${activeTab === 'personalized' ? 'active' : ''}`}
							onClick={() => setActiveTab('personalized')}
						>
							🏠 Your House
						</button>
					</div>
					<button type="button" className="theme-toggle" onClick={() => setTheme(t => (t === 'light' ? 'dark' : 'light'))}>
						{theme === 'light' ? '🌙 Dark' : '☀️ Light'}
					</button>
				</div>
			</div>

			{activeTab === 'prediction' && (
				<>


					<div className="form-container">
						<h2 className="form-title">House Price Prediction</h2>
						<p className="form-subtitle">
							Enter your house details below to get an accurate price prediction
						</p>
						
						<form onSubmit={handleSubmit}>
							<div className="form-grid">
								<div className="form-group">
									<label htmlFor="MedInc">
										Median Income (USD × 10,000)
									</label>
									<input
										type="number"
										id="MedInc"
										step="0.1"
										value={values.MedInc}
										onChange={(e) => setValues({...values, MedInc: e.target.value})}
										placeholder="e.g., 8.3"
										required
									/>
									<div className="helper-text">
										Median household income in the block area (in tens of thousands USD)
									</div>
								</div>

								<div className="form-group">
									<label htmlFor="HouseAge">
										House Age (Years)
									</label>
									<input
										type="number"
										id="HouseAge"
										value={values.HouseAge}
										onChange={(e) => setValues({...values, HouseAge: e.target.value})}
										placeholder="e.g., 25"
										required
									/>
									<div className="helper-text">
										Age of the house in years
									</div>
								</div>

								<div className="form-group">
									<label htmlFor="AveRooms">
										 Average Rooms
									</label>
									<input
										type="number"
										id="AveRooms"
										step="0.1"
										value={values.AveRooms}
										onChange={(e) => setValues({...values, AveRooms: e.target.value})}
										placeholder="e.g., 6.0"
										required
									/>
									<div className="helper-text">
										Average number of rooms per household
									</div>
								</div>

								<div className="form-group">
									<label htmlFor="AveBedrms">
										Average Bedrooms
									</label>
									<input
										type="number"
										id="AveBedrms"
										step="0.1"
										value={values.AveBedrms}
										onChange={(e) => setValues({...values, AveBedrms: e.target.value})}
										placeholder="e.g., 3.0"
										required
									/>
									<div className="helper-text">
										Average number of bedrooms per household
									</div>
								</div>

								<div className="form-group">
									<label htmlFor="Population">
										Local Population (Block Area)
									</label>
									<input
										type="number"
										id="Population"
										value={values.Population}
										onChange={(e) => setValues({...values, Population: e.target.value})}
										placeholder="e.g., 1500"
										required
									/>
									<div className="helper-text">
										Total population in the block area
									</div>
								</div>

								<div className="form-group">
									<label htmlFor="AveOccup">
										Average Occupancy
									</label>
									<input
										type="number"
										id="AveOccup"
										step="0.1"
										value={values.AveOccup}
										onChange={(e) => setValues({...values, AveOccup: e.target.value})}
										placeholder="e.g., 3.0"
										required
									/>
									<div className="helper-text">
										Average number of household members
									</div>
								</div>

								<div className="form-group">
									<label htmlFor="Latitude">
										Latitude
									</label>
									<input
										type="number"
										id="Latitude"
										step="0.0001"
										value={values.Latitude}
										onChange={(e) => setValues({...values, Latitude: e.target.value})}
										placeholder="e.g., 37.7749"
										required
									/>
									<div className="helper-text">
										Geographic latitude coordinate
									</div>
								</div>

								<div className="form-group">
									<label htmlFor="Longitude">
										Longitude
									</label>
									<input
										type="number"
										id="Longitude"
										step="0.0001"
										value={values.Longitude}
										onChange={(e) => setValues({...values, Longitude: e.target.value})}
										placeholder="e.g., -122.4194"
										required
									/>
									<div className="helper-text">
										Geographic longitude coordinate
									</div>
								</div>

								<div className="form-group">
									<label htmlFor="currency">
										Currency
									</label>
									<select
										id="currency"
										value={currency}
										onChange={(e) => setCurrency(e.target.value)}
									>
										<option value="USD">USD ($)</option>
										<option value="INR">INR (₹)</option>
										<option value="EUR">EUR (€)</option>
										<option value="GBP">GBP (£)</option>
										<option value="JPY">JPY (¥)</option>
									</select>
									<div className="helper-text">
										Select your preferred currency for price display
									</div>
								</div>

								<div className="form-group">
									<label htmlFor="country">
										Country / Region
									</label>
									<select
										id="country"
										value={country}
										onChange={(e) => setCountry(e.target.value)}
									>
										<option value="US">United States</option>
										<option value="IN">India</option>
										<option value="EU">Europe</option>
										<option value="UK">United Kingdom</option>
										<option value="JP">Japan</option>
										<option value="Other">Other</option>
									</select>
									<div className="helper-text">
										Select your region for better context
									</div>
								</div>
							</div>

							<button type="submit" className="submit-btn" disabled={loading}>
								{loading ? '🔄 Predicting...' : '🚀 Predict House Price'}
							</button>
						</form>
					</div>

					{error && (
						<div className="error">
							❌ Error: {error}
							<button onClick={() => setError(null)}>Dismiss</button>
						</div>
					)}

					{result && (
						<div className="prediction-result">
							<h3 className="prediction-title">🎯 Price Prediction Result</h3>
							<div className="predicted-price">
								{formatWithUnits(convertedAmount, currency)}
							</div>
							
							<div className="price-breakdown">
								<div className="price-item">
									<div className="price-label">Predicted Price ({currency})</div>
									<div className="price-value">{selectedSymbol}{currency === 'INR' ? nfIN.format(Math.round(convertedAmount)) : nfUS.format(Math.round(convertedAmount))}</div>
								</div>
								<div className="price-item">
									<div className="price-label">Base Price (USD)</div>
									<div className="price-value">${nfUS.format(Math.round(usdAmount))}</div>
								</div>
								<div className="price-item">
									<div className="price-label">Raw Prediction (100k units)</div>
									<div className="price-value">{result.predicted_price_100k}</div>
								</div>
								<div className="price-item">
									<div className="price-label">Exchange Rate</div>
									<div className="price-value">{fxLive ? 'Live' : 'Default'} ({exchangeRate.toFixed(4)})</div>
								</div>
							</div>
						</div>
					)}


				</>
			)}

			{activeTab === 'charts' && (
				<Charts theme={theme} />
			)}

			{activeTab === 'personalized' && (
				<PersonalizedCharts 
					theme={theme} 
					userData={values}
					onAnalysisComplete={(data) => {
						// Optional: You can use this data for other purposes
						console.log('Personalized analysis complete:', data);
					}}
				/>
			)}
		</div>
	)
}

export default App
