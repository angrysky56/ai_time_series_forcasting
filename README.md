# 🤖 AI-Assisted Multi-Model Time Series Forecasting Platform

An advanced, production-ready time series forecasting system that combines traditional statistical models, modern neural networks, AI-driven parameter optimization, and comprehensive market context analysis to deliver sophisticated financial predictions with LLM-generated narrative insights.

![Platform Demo](image-2.png)

## 🚀 Key Features

### **Multi-Model Forecasting Engine**
- **Traditional Models**: Prophet, ARIMA, ETS with advanced parameter tuning
- **Neural Networks**: LSTM, RNN, GRU, N-BEATS, Transformer, TCN, TFT, NLinear
- **Unified Framework**: All models integrated via Darts framework for consistent handling
- **Auto-Selection**: AI-powered model selection using walk-forward validation

### **AI-Driven Optimization**
- **Smart Indicator Selection**: 25+ technical indicators with automatic parameter optimization
- **Market Regime Analysis**: Adaptive indicator selection based on market conditions
- **Optuna Integration**: Hyperparameter optimization for maximum prediction accuracy
- **Data Characteristics Analysis**: Trend strength, volatility, momentum, and cyclicality assessment

### **Multi-Timeframe Coherent Analysis**
- **Synchronized Forecasting**: Monthly, weekly, daily, and hourly predictions with consistency scoring
- **Cross-Timeframe Validation**: Ensures logical alignment between different time horizons
- **Regime-Aware Predictions**: Adapts forecasting approach based on market conditions
- **Volatility Assessment**: Dynamic confidence intervals across timeframes

### **Comprehensive Market Context**
- **Traditional Markets**: S&P 500, VIX, DXY, sector ETFs, treasury yields
- **Crypto Markets**: BTC/ETH dominance, correlation analysis, cross-exchange data
- **Fundamental Analysis**: P/E ratios, analyst ratings, financial statements, sector performance
- **Risk Assessment**: Market regime detection, correlation matrices, opportunity identification

### **Advanced Backtesting & Validation**
- **Walk-Forward Validation**: Gold standard time series cross-validation
- **Model Comparison**: Automated performance comparison across multiple models
- **Rolling/Expanding Windows**: Flexible validation approaches for different market conditions
- **Comprehensive Metrics**: MAE, RMSE, MAPE with confidence intervals and fold analysis

### **LLM-Powered Analysis**
- **Narrative Generation**: AI-generated market analysis and trading insights
- **Multi-Period Synthesis**: Coherent analysis across all timeframes
- **Risk & Opportunity Assessment**: Contextual interpretation of technical and fundamental signals
- **Dynamic Adaptation**: Analysis adapts to current market regime and conditions

## 🏗️ Architecture

```
📦 AI Time Series Forecasting Platform
├── 🧠 Core Forecasting Engine
│   ├── Multi-model support (Prophet, ARIMA, ETS, Neural Networks)
│   ├── Unified prediction interface via Darts
│   └── Frequency-aware time series handling
├── 🎯 AI Parameter Optimizer
│   ├── Automatic indicator selection (25+ indicators)
│   ├── Optuna-based hyperparameter tuning
│   └── Market regime adaptive configuration
├── 📊 Multi-Period Forecaster
│   ├── Coherent cross-timeframe analysis
│   ├── Consistency scoring algorithms
│   └── Synchronized prediction generation
├── 🌐 Market Context Analyzer
│   ├── Traditional & crypto market indicators
│   ├── Fundamental analysis integration
│   └── Risk/opportunity assessment
├── 🔬 Advanced Backtesting
│   ├── Walk-forward validation
│   ├── Model comparison framework
│   └── Performance analytics
├── 🤖 LLM Integration
│   ├── Multi-period narrative generation
│   ├── Market context interpretation
│   └── Trading insights synthesis
└── 🖥️ Streamlit Interface
    ├── Interactive multi-period visualization
    ├── Real-time model comparison
    └── Comprehensive analytics dashboard
```

## 📁 Project Structure

```
├── src/
│   ├── forecasting.py              # Multi-model forecasting engine
│   ├── multi_period_forecaster.py  # Cross-timeframe coherent analysis
│   ├── ai_parameter_optimizer.py   # AI-driven indicator optimization
│   ├── market_indicators.py        # Market context & fundamental analysis
│   ├── backtesting.py              # Walk-forward validation & model comparison
│   ├── llm_integration.py          # AI narrative generation
│   ├── technical_analysis.py       # Technical indicator calculations
│   ├── data_fetcher.py             # Multi-source data acquisition
│   └── utils/
│       └── chart_renderer.py       # Advanced visualization system
├── tests/                          # Comprehensive test suite
├── notebooks/                      # Research & development notebooks
├── data/                          # Local data storage
├── app.py                         # Main Streamlit application
├── pyproject.toml                 # UV project configuration
└── README.md                      # This file
```

## 🛠️ Installation & Setup

### Prerequisites
- **Python 3.12+** (tested with 3.12)
- **UV Package Manager** (recommended for dependency management)
- **Optional**: LM Studio for local LLM integration

### Quick Start

```bash
# Clone the repository
git clone https://github.com/angrysky56/ai_time_series_forecasting
cd ai_time_series_forecasting

# Create and activate virtual environment with UV
uv venv --python 3.12
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
uv pip install -e .

# Launch the application
streamlit run app.py
```

### Dependencies Highlights
- **Forecasting**: `prophet`, `darts`, `statsmodels`, `neuralprophet`
- **ML/Optimization**: `optuna`, `scikit-learn`, `xgboost`, `lightgbm`
- **Neural Networks**: `torch`, `pytorch-lightning`, `tensorflow`
- **Data Sources**: `yfinance`, `ccxt`, `pycoingecko`
- **Technical Analysis**: `talipp` (25+ indicators)
- **Visualization**: `plotly`, `streamlit`

## 🎮 Usage Guide

### Single-Period Analysis
1. **Select Data Source**: Choose between stocks, crypto, or auto-detection
2. **Configure Parameters**: Set timeframe, data range, and forecast periods
3. **Model Selection**: Choose specific model or enable auto-selection with walk-forward validation
4. **AI Optimization**: Enable AI-driven indicator selection and parameter tuning
5. **Generate Analysis**: Get comprehensive forecasts with market context and LLM insights

### Multi-Period Coherent Analysis
1. **Enable Multi-Period Mode**: Switch to coherent cross-timeframe analysis
2. **Automatic Data Fetching**: System fetches optimized data for all timeframes
3. **Consistency Analysis**: Review alignment between monthly, weekly, daily, and hourly predictions
4. **Integrated Insights**: Get unified analysis across all time horizons
5. **Trading Recommendations**: Receive regime-aware trading suggestions

### Advanced Features
- **Walk-Forward Validation**: Enable for robust model performance assessment
- **Market Context**: Automatic correlation analysis with broader market indicators
- **AI Indicator Selection**: Let AI choose optimal technical indicators based on market regime
- **Fundamental Integration**: Automatic P/E, sector, and analyst data incorporation
- **Risk Assessment**: Comprehensive risk/opportunity analysis

## 🔧 Configuration

### Model Selection Strategy
- **Conservative**: Prophet + ARIMA (stable, interpretable)
- **Aggressive**: Neural Networks (LSTM, Transformer, N-BEATS)
- **Adaptive**: Auto-selection with walk-forward validation
- **Ensemble**: Multiple models with performance-weighted combination

### Data Sources
- **Primary**: Yahoo Finance (comprehensive stock/crypto coverage)
- **Crypto Extensions**: CCXT for exchange-specific data
- **Market Context**: Automatic traditional market indicator integration
- **Fundamentals**: yfinance-based financial statement analysis

### Timeframe Optimization
- **Intraday**: 1m, 5m, 15m, 30m, 1h, 4h
- **Daily+**: 1d, 1w, 1mo with seasonal adjustments
- **Frequency Handling**: Automatic pandas frequency mapping
- **Cross-Timeframe**: Intelligent consistency scoring

## 📊 Performance Metrics

### Backtesting Standards
- **Walk-Forward Validation**: Industry-standard time series validation
- **Multiple Metrics**: MAE, RMSE, MAPE with statistical confidence
- **Fold Analysis**: Performance stability across different market conditions
- **Model Comparison**: Automated best-model selection

### Market Context Integration
- **Correlation Analysis**: Dynamic correlation with market indicators
- **Regime Detection**: Bull/bear/volatile/sideways market identification
- **Risk Assessment**: Volatility, drawdown, and stability metrics
- **Opportunity Scoring**: Risk-adjusted return potential

## 🤖 AI & LLM Integration

### Local LLM Setup (Optional)
1. Install [LM Studio](https://lmstudio.ai/)
2. Download a compatible model (Llama, Mistral, etc.)
3. Start local server on default port
4. Enable AI analysis in the application

### Narrative Generation
- **Technical Analysis**: AI interpretation of indicator signals
- **Market Context**: Synthesis of broader market conditions
- **Risk Assessment**: Natural language risk/opportunity analysis
- **Trading Insights**: Actionable recommendations based on multi-factor analysis

## 🔬 Research & Development

### Experimental Features
- **Ensemble Methods**: Multi-model combination strategies
- **Online Learning**: Real-time model adaptation
- **Regime Switching**: Dynamic model selection based on market conditions
- **Advanced Risk Models**: Value-at-Risk and Expected Shortfall integration

### Performance Benchmarks
- **Traditional Models**: Prophet (stable baseline), ARIMA (trend-following), ETS (seasonality)
- **Neural Networks**: Transformer (attention-based), N-BEATS (interpretable), TCN (efficient)
- **Comparison Metrics**: Sharpe ratio, maximum drawdown, hit rate, profit factor

## 🛣️ Roadmap

### Phase 1: Core Enhancements ✅
- Multi-model integration
- AI parameter optimization
- Multi-timeframe analysis
- Advanced backtesting

### Phase 2: Intelligence Layer (In Progress)
- Ensemble forecasting
- Dynamic model selection
- Real-time adaptation
- Enhanced risk management

### Phase 3: Production Features (Planned)
- API endpoints
- Database integration
- Real-time monitoring
- Portfolio optimization

## 🤝 Contributing

### Development Setup
```bash
# Install development dependencies
uv pip install -e ".[dev]"

# Run tests
pytest tests/

# Code formatting
ruff format .
ruff check .
```

### Code Standards
- **Style**: PEP 8 with descriptive naming
- **Testing**: Comprehensive unit and integration tests
- **Documentation**: Docstrings for all public functions
- **Type Hints**: Full type annotation coverage

## 📜 License

MIT License - see LICENSE file for details.

## 🙏 Acknowledgments

- **Prophet**: Facebook's time series forecasting library
- **Darts**: Unified time series forecasting framework
- **Optuna**: Hyperparameter optimization framework
- **Talipp**: Comprehensive technical analysis library
- **Streamlit**: Interactive web application framework

---

**Built with ❤️ for the quantitative finance community**

For questions, issues, or contributions, please open an issue on GitHub.
