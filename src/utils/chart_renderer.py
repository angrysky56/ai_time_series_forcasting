"""Chart rendering utilities with error boundaries and improved reliability.

This module extracts complex chart rendering logic from the main app to improve 
maintainability and provide robust error handling for plotly charts.

Following Meta-Cognitive AI Coding Protocol:
- Target: Extract complex chart logic from monolithic app.py
- Scope: Minimum viable abstraction for chart rendering
- Leverage: Use existing plotly patterns and data structures
- Preserve: Maintain all advanced chart features and customization
"""

import logging
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from typing import Dict, Any, Optional, List, Union

logger = logging.getLogger(__name__)


class ChartRenderError(Exception):
    """Custom exception for chart rendering errors."""
    pass


class StreamlitChartRenderer:
    """Handles safe rendering of complex plotly charts in Streamlit.
    
    Provides error boundaries, data validation, and fallback mechanisms
    for chart rendering operations.
    """
    
    def __init__(self):
        self.default_config = {
            'displaylogo': False,
            'modeBarButtonsToRemove': ['pan2d', 'lasso2d'],
            'scrollZoom': True
        }
    
    def render_safe_chart(
        self, 
        fig: go.Figure, 
        title: str = "Chart",
        use_container_width: bool = True,
        theme: str = "streamlit",
        config: Optional[Dict[str, Any]] = None
    ) -> bool:
        """Safely render a plotly chart with error boundaries.
        
        Args:
            fig: Plotly figure object
            title: Chart title for error messages
            use_container_width: Whether to use container width
            theme: Chart theme ('streamlit' or None)
            config: Additional plotly config options
            
        Returns:
            bool: True if successful, False if error occurred
        """
        try:
            if fig is None:
                st.error(f"❌ {title}: No data available for chart rendering")
                return False
            
            # Validate figure has data
            if not fig.data:
                st.warning(f"⚠️ {title}: Chart has no data traces")
                return False
            
            # Merge config with defaults
            chart_config = {**self.default_config}
            if config:
                chart_config.update(config)
            
            # Render with error boundary
            st.plotly_chart(
                fig, 
                use_container_width=use_container_width,
                theme=theme,
                config=chart_config
            )
            return True
            
        except Exception as e:
            error_msg = f"❌ {title} rendering failed: {str(e)}"
            logger.error(error_msg, exc_info=True)
            st.error(error_msg)
            st.info("💡 Try refreshing the page or adjusting the data range")
            return False
    
    def create_multi_period_chart(
        self,
        multi_data: Dict[str, pd.DataFrame],
        forecasts: Dict[str, pd.DataFrame],
        symbol: str
    ) -> Optional[go.Figure]:
        """Create multi-period forecast chart with error handling.
        
        Args:
            multi_data: Historical data by timeframe
            forecasts: Forecast data by timeframe
            symbol: Trading symbol
            
        Returns:
            Plotly figure or None if error
        """
        try:
            if not multi_data or not forecasts:
                raise ChartRenderError("No data available for multi-period chart")
            
            # Create subplot layout
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=('Monthly Outlook', 'Weekly Outlook', 'Daily Outlook', 'Hourly Outlook'),
                shared_xaxes=False,
                vertical_spacing=0.08,
                horizontal_spacing=0.1
            )
            
            positions = [(1,1), (1,2), (2,1), (2,2)]
            colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
            periods = ['monthly', 'weekly', 'daily', 'hourly']
            
            for i, period in enumerate(periods):
                if i >= 4:  # Safety check
                    break
                    
                row, col = positions[i]
                color = colors[i]
                
                # Skip if data not available
                if period not in multi_data or period not in forecasts:
                    continue
                    
                historical_data = multi_data[period]
                forecast_df = forecasts[period]
                
                if historical_data.empty or forecast_df.empty:
                    continue
                
                # Add historical data (last 50 points for clarity)
                hist_slice = historical_data.tail(50)
                fig.add_trace(
                    go.Scatter(
                        x=hist_slice['timestamp'],
                        y=hist_slice['close'],
                        mode='lines',
                        name=f'{period.title()} Historical',
                        line=dict(color=color, width=1),
                        opacity=0.7,
                        legendgroup=period
                    ),
                    row=row, col=col
                )
                
                # Add forecast data
                self._add_forecast_trace(fig, forecast_df, historical_data, period, color, row, col)
            
            # Update layout
            fig.update_layout(
                title=f"{symbol} - Multi-Period Forecast Analysis",
                height=800,
                showlegend=False,
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            return fig
            
        except Exception as e:
            logger.error(f"Multi-period chart creation failed: {e}", exc_info=True)
            raise ChartRenderError(f"Failed to create multi-period chart: {str(e)}")
    
    def create_single_period_chart(
        self,
        data: pd.DataFrame,
        forecast: pd.DataFrame,
        symbol: str,
        model_name: str,
        timeframe: str,
        technical_indicators: Optional[Dict[str, Any]] = None,
        show_rsi: bool = False
    ) -> Optional[go.Figure]:
        """Create single period forecast chart with technical indicators.
        
        Args:
            data: Historical price data
            forecast: Forecast data
            symbol: Trading symbol
            model_name: Forecasting model name
            timeframe: Data timeframe
            technical_indicators: Technical indicator settings
            show_rsi: Whether to show RSI subplot
            
        Returns:
            Plotly figure or None if error
        """
        try:
            if data is None or data.empty:
                raise ChartRenderError("No historical data available")
            
            if forecast is None or forecast.empty:
                raise ChartRenderError("No forecast data available")
            
            # Create subplot layout
            rows = 2 if show_rsi else 1
            fig = make_subplots(
                rows=rows,
                cols=1,
                shared_xaxes=True,
                vertical_spacing=0.05,
                row_heights=[0.7, 0.3] if show_rsi else [1.0],
                subplot_titles=[f'{symbol} - {model_name} Forecast ({timeframe})', 'RSI'] if show_rsi else [f'{symbol} - {model_name} Forecast ({timeframe})']
            )
            
            # Add candlestick chart
            fig.add_trace(
                go.Candlestick(
                    x=data['timestamp'],
                    open=data['open'],
                    high=data['high'],
                    low=data['low'],
                    close=data['close'],
                    name='Price',
                    increasing_line_color='#00cc96',
                    decreasing_line_color='#ef553b'
                ),
                row=1, col=1
            )
            
            # Add forecast line
            fig.add_trace(
                go.Scatter(
                    x=forecast['ds'],
                    y=forecast['yhat'],
                    mode='lines',
                    name=f'{model_name} Forecast',
                    line=dict(color='#ff7f0e', width=3)
                ),
                row=1, col=1
            )
            
            # Add confidence intervals if available
            self._add_confidence_intervals(fig, forecast, data, row=1, col=1)
            
            # Add technical indicators
            if technical_indicators:
                self._add_technical_indicators(fig, data, technical_indicators, row=1, col=1)
            
            # Add RSI subplot if requested
            if show_rsi and 'RSI' in data.columns:
                self._add_rsi_subplot(fig, data, row=2, col=1)
            
            # Update layout
            fig.update_layout(
                height=700 if show_rsi else 600,
                xaxis_rangeslider_visible=False,
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                ),
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)'
            )
            
            fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
            if show_rsi:
                fig.update_xaxes(showticklabels=False, row=1, col=1)
            
            return fig
            
        except Exception as e:
            logger.error(f"Single period chart creation failed: {e}", exc_info=True)
            raise ChartRenderError(f"Failed to create chart: {str(e)}")
    
    def _add_forecast_trace(
        self, 
        fig: go.Figure, 
        forecast_df: pd.DataFrame, 
        historical_data: pd.DataFrame,
        period: str,
        color: str,
        row: int,
        col: int
    ):
        """Add forecast trace with proper data handling."""
        try:
            # Get future points only - fix timezone mismatch issue
            hist_timestamps = pd.to_datetime(historical_data['timestamp'])
            # Convert to timezone-naive UTC if timezone-aware
            if hist_timestamps.dt.tz is not None:
                hist_timestamps = hist_timestamps.dt.tz_convert('UTC').dt.tz_localize(None)
            last_hist_ts = hist_timestamps.max()
            
            fdf = forecast_df.copy()
            fdf['ds'] = pd.to_datetime(fdf['ds'], errors='coerce')
            # Ensure forecast timestamps are timezone-naive UTC
            if fdf['ds'].dt.tz is not None:
                fdf['ds'] = fdf['ds'].dt.tz_convert('UTC').dt.tz_localize(None)
            
            # Now both timestamps are timezone-naive UTC - safe to compare
            future_fdf = fdf[fdf['ds'] > last_hist_ts]
            
            if future_fdf.empty:
                future_fdf = fdf.tail(1)  # Fallback
            
            fig.add_trace(
                go.Scatter(
                    x=future_fdf['ds'],
                    y=future_fdf['yhat'],
                    mode='lines',
                    name=f'{period.title()} Forecast',
                    line=dict(color=color, width=3),
                    legendgroup=period
                ),
                row=row, col=col
            )
            
            # Add confidence intervals if available
            if 'yhat_lower' in future_fdf.columns and future_fdf['yhat_lower'].notna().any():
                fig.add_trace(
                    go.Scatter(
                        x=future_fdf['ds'],
                        y=future_fdf['yhat_upper'],
                        mode='lines',
                        line=dict(width=0),
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=row, col=col
                )
                fig.add_trace(
                    go.Scatter(
                        x=future_fdf['ds'],
                        y=future_fdf['yhat_lower'],
                        mode='lines',
                        line=dict(width=0),
                        fill='tonexty',
                        fillcolor=f'rgba({self._hex_to_rgb(color)}, 0.2)',
                        showlegend=False,
                        hoverinfo='skip'
                    ),
                    row=row, col=col
                )
        except Exception as e:
            logger.warning(f"Failed to add forecast trace for {period}: {e}")
    
    def _add_confidence_intervals(
        self, 
        fig: go.Figure, 
        forecast: pd.DataFrame, 
        data: pd.DataFrame,
        row: int = 1,
        col: int = 1
    ):
        """Add confidence intervals to forecast."""
        try:
            if 'yhat_lower' not in forecast.columns or forecast['yhat_lower'].isna().all():
                return
            
            # Show confidence intervals for future points only - fix timezone mismatch
            hist_timestamps = pd.to_datetime(data['timestamp'])
            # Convert to timezone-naive UTC if timezone-aware
            if hist_timestamps.dt.tz is not None:
                hist_timestamps = hist_timestamps.dt.tz_convert('UTC').dt.tz_localize(None)
            last_hist_ts = hist_timestamps.max()
            
            forecast_ts = pd.to_datetime(forecast['ds'])
            # Ensure forecast timestamps are timezone-naive UTC
            if forecast_ts.dt.tz is not None:
                forecast_ts = forecast_ts.dt.tz_convert('UTC').dt.tz_localize(None)
            
            # Now both timestamps are timezone-naive UTC - safe to compare
            future_mask = forecast_ts > last_hist_ts
            future_points = forecast[future_mask]
            
            if future_points.empty:
                future_points = forecast.tail(1)
            
            fig.add_trace(
                go.Scatter(
                    x=future_points['ds'],
                    y=future_points['yhat_upper'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo='skip'
                ),
                row=row, col=col
            )
            fig.add_trace(
                go.Scatter(
                    x=future_points['ds'],
                    y=future_points['yhat_lower'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(255, 127, 14, 0.2)',
                    showlegend=False,
                    name='Confidence Interval'
                ),
                row=row, col=col
            )
        except Exception as e:
            logger.warning(f"Failed to add confidence intervals: {e}")
    
    def _add_technical_indicators(
        self, 
        fig: go.Figure, 
        data: pd.DataFrame, 
        indicators: Dict[str, Any],
        row: int = 1,
        col: int = 1
    ):
        """Add technical indicators to chart."""
        try:
            for indicator, config in indicators.items():
                if indicator.startswith('SMA_') and indicator in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data['timestamp'],
                            y=data[indicator],
                            mode='lines',
                            name=indicator.replace('_', ' '),
                            line=dict(color='blue', width=1)
                        ),
                        row=row, col=col
                    )
                elif indicator.startswith('EMA_') and indicator in data.columns:
                    fig.add_trace(
                        go.Scatter(
                            x=data['timestamp'],
                            y=data[indicator],
                            mode='lines',
                            name=indicator.replace('_', ' '),
                            line=dict(color='green', width=1)
                        ),
                        row=row, col=col
                    )
        except Exception as e:
            logger.warning(f"Failed to add technical indicators: {e}")
    
    def _add_rsi_subplot(
        self, 
        fig: go.Figure, 
        data: pd.DataFrame,
        row: int = 2,
        col: int = 1
    ):
        """Add RSI subplot with proper boundaries."""
        try:
            if 'RSI' not in data.columns:
                return
            
            fig.add_trace(
                go.Scatter(
                    x=data['timestamp'],
                    y=data['RSI'],
                    mode='lines',
                    name='RSI',
                    line=dict(color='purple', width=1)
                ),
                row=row, col=col
            )
            
            # Add RSI levels
            x_range = [data['timestamp'].iloc[0], data['timestamp'].iloc[-1]]
            
            # Overbought line (70)
            fig.add_shape(
                type="line",
                x0=x_range[0], x1=x_range[1],
                y0=70, y1=70,
                line=dict(dash="dash", color="red", width=1),
                row=row, col=col
            )
            
            # Oversold line (30)
            fig.add_shape(
                type="line",
                x0=x_range[0], x1=x_range[1],
                y0=30, y1=30,
                line=dict(dash="dash", color="green", width=1),
                row=row, col=col
            )
            
            fig.update_yaxes(title_text="RSI", range=[0, 100], row=row, col=col)
            
        except Exception as e:
            logger.warning(f"Failed to add RSI subplot: {e}")
    
    def _hex_to_rgb(self, hex_color: str) -> str:
        """Convert hex color to RGB string for plotly."""
        try:
            hex_color = hex_color.lstrip('#')
            rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            return f"{rgb[0]}, {rgb[1]}, {rgb[2]}"
        except:
            return "128, 128, 128"  # Default gray
