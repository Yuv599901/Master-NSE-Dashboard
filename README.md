# 🇮🇳 NSE AI Master Fundamental & Technical Dashboard

A mobile-optimized, full-featured financial analytics dashboard built with **Streamlit**, **Plotly**, and **yfinance**. It combines automated fundamental scraping from **Screener.in**, Stan Weinstein **Stage-2 trend templates**, **ATR position sizing**, and AI-driven trade action verdicts tailored for Indian stock markets (NSE/BSE).

---

## 🚀 Key Features

* **🧠 AI Action Engine & Scorecard:** Generates composite health scores out of 100 with actionable Buy/Hold/Exit guidance based on moving averages, RSI, and ROE.
* **🔒 Screener.in Automated Scraper:** Securely authenticates and parses consolidated P&L, balance sheets, cash flow statements, quarterly promoter shareholding patterns, and pros/cons directly from Screener.in.
* **📈 Advanced Technical Charts:** Multi-pane interactive Plotly charts featuring Candlesticks, Exponential/Simple Moving Averages (EMA21, SMA50, SMA200), Volume, RSI (14), and MACD.
* **🎯 Stage-2 Trend Template:** Evaluates 5 key institutional uptrend filters.
* **🛡️ ATR Position Sizing Calculator:** Volatility-based stop-loss and multi-R risk-reward target calculations based on custom trading capital and risk parameters.
* **📱 Mobile-Optimized Layout:** Responsive CSS styling and stacked metric cards designed for seamless viewing on smartphone and tablet touchscreens.

---

## 🛠️ Project Directory Structure

```text
nse-master-dashboard/
├── app.py              # Main Streamlit application script
├── requirements.txt    # Python dependencies
└── README.md           # Project documentation
