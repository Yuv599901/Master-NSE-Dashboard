from datetime import datetime
import io
import urllib.parse
import xml.etree.ElementTree as ET
import BeautifulSoup as bs4
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import requests
import streamlit as st
import yfinance as yf

# ---------------------------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="NSE AI Master Fundamental & Technical Dashboard",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling for Metric Cards & Aesthetic Polish
st.markdown("""
<style>
    .metric-card { background-color: #1e2530; padding: 15px; border-radius: 10px; border: 1px solid #2d3748; text-align: center; }
    .pro-box { background-color: rgba(0, 204, 150, 0.1); padding: 12px; border-left: 4px solid #00CC96; border-radius: 4px; margin-bottom: 8px;}
    .con-box { background-color: rgba(239, 85, 59, 0.1); padding: 12px; border-left: 4px solid #EF553B; border-radius: 4px; margin-bottom: 8px;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# 1. Technical Analysis Helpers & Indicators
# ---------------------------------------------------------------------------
def nse_tick_size(price: float) -> float:
    if price < 250:
        return 0.01
    elif price < 1000:
        return 0.05
    elif price < 5000:
        return 0.10
    elif price < 10000:
        return 0.50
    elif price < 20000:
        return 1.00
    return 5.00


def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close = df["Close"]
    high = df["High"]
    low = df["Low"]

    df["SMA20"] = close.rolling(20).mean()
    df["SMA50"] = close.rolling(50).mean()
    df["SMA150"] = close.rolling(150).mean()
    df["SMA200"] = close.rolling(200).mean()
    df["EMA21"] = close.ewm(span=21, adjust=False).mean()

    std20 = close.rolling(20).std()
    df["BB_Upper"] = df["SMA20"] + 2 * std20
    df["BB_Lower"] = df["SMA20"] - 2 * std20

    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    df["RSI14"] = 100 - (100 / (1 + rs))

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["MACD"] = ema12 - ema26
    df["MACD_Signal"] = df["MACD"].ewm(span=9, adjust=False).mean()
    df["MACD_Hist"] = df["MACD"] - df["MACD_Signal"]

    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    df["ATR14"] = tr.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    df["Vol_SMA50"] = df["Volume"].rolling(50).mean()
    return df


def evaluate_stage2_template(df: pd.DataFrame) -> dict:
    last = df.iloc[-1]
    close = last["Close"]
    s50, s150, s200 = last["SMA50"], last["SMA150"], last["SMA200"]
    s200_prev = df["SMA200"].iloc[-22] if len(df) > 220 else np.nan

    yr = df.tail(252)
    hi52 = yr["High"].max()
    lo52 = yr["Low"].min()

    checks = {
        "Price > SMA 50, 150 & 200": bool(close > s50 and close > s150 and close > s200),
        "SMA 50 > SMA 150 > SMA 200": bool(s50 > s150 > s200),
        "SMA 200 Trending Up (1M)": bool(s200 > s200_prev) if not np.isnan(s200_prev) else False,
        "Within 25% of 52W High": bool((close / hi52 - 1) * 100 >= -25.0),
        "At least 30% above 52W Low": bool((close / lo52 - 1) * 100 >= 30.0),
    }
    return {"checks": checks, "score": sum(checks.values()), "total": len(checks), "hi52": hi52, "lo52": lo52}


def find_pivot_levels(df: pd.DataFrame, lookback: int = 20):
    recent = df.tail(lookback)
    resistance = recent["High"].max()
    support = recent["Low"].min()
    pivot = (recent["High"].iloc[-1] + recent["Low"].iloc[-1] + recent["Close"].iloc[-1]) / 3
    r1 = 2 * pivot - recent["Low"].iloc[-1]
    s1 = 2 * pivot - recent["High"].iloc[-1]
    return {"Resistance (20D)": resistance, "R1 Classic": r1, "Pivot": pivot, "S1 Classic": s1, "Support (20D)": support}


# ---------------------------------------------------------------------------
# 2. Advanced Screener.in Scraper Module (Promoters, Pros/Cons, P&L, CF)
# ---------------------------------------------------------------------------
class ScreenerScraper:
    def __init__(self, username: str, password: str, ticker: str):
        self.username = username
        self.password = password
        self.ticker = ticker.upper().strip().replace(".NS", "").replace(".BO", "")
        self.session = requests.Session()
        self.logged_in = False
        self.error_message = None
        self._authenticate()

    def _authenticate(self):
        if not self.username or not self.password:
            self.error_message = "Credentials missing (Username or Password empty)."
            return
        try:
            login_url = "https://www.screener.in/login/"
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
                "Referer": login_url,
            }
            resp = self.session.get(login_url, headers=headers, timeout=10)
            if resp.status_code != 200:
                self.error_message = f"Failed to reach Screener login page. HTTP {resp.status_code}"
                return

            soup = BeautifulSoup(resp.text, "html.parser")
            csrf_input = soup.find("input", {"name": "csrfmiddlewaretoken"})
            csrf_token = csrf_input["value"] if csrf_input else self.session.cookies.get("csrftoken", "")

            if not csrf_token:
                self.error_message = "Could not extract CSRF token from Screener login form."
                return

            payload = {
                "csrfmiddlewaretoken": csrf_token,
                "username": self.username,
                "password": self.password,
            }
            post_headers = {
                **headers,
                "Content-Type": "application/x-www-form-urlencoded",
                "Origin": "https://www.screener.in",
            }

            post_resp = self.session.post(login_url, data=payload, headers=post_headers, timeout=10, allow_redirects=True)
            if post_resp.status_code == 200 and ("logout" in post_resp.text.lower() or "account" in post_resp.text.lower()):
                self.logged_in = True
                self.error_message = None
            else:
                self.error_message = f"Login rejected (HTTP {post_resp.status_code}). Check your email and password."
        except Exception as e:
            self.error_message = f"Authentication Exception: {str(e)}"
            self.logged_in = False

    def _parse_table_html(self, html_fragment: str) -> pd.DataFrame:
        df = pd.read_html(io.StringIO(html_fragment))[0]
        if str(df.columns[0]).startswith("Unnamed"):
            df = df.rename(columns={df.columns[0]: "Particulars"})
        df["Particulars"] = df["Particulars"].astype(str).str.replace(r"\s*\+$", "", regex=True).str.strip()
        return df

    def scrape_company_page(self) -> dict:
        parsed_data = {
            "ratios": {},
            "growth_tables": {},
            "pros": [],
            "cons": [],
            "profit_loss": pd.DataFrame(),
            "cash_flows": pd.DataFrame(),
            "balance_sheet": pd.DataFrame(),
            "shareholding": pd.DataFrame(),
            "documents": [],
            "source": f"Error: {self.error_message}",
        }
        if not self.logged_in:
            return parsed_data

        try:
            headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            url_cons = f"https://www.screener.in/company/{self.ticker}/consolidated/"
            resp = self.session.get(url_cons, headers=headers, timeout=10)
            if resp.status_code != 200 or "cash-flow" not in resp.text:
                url_std = f"https://www.screener.in/company/{self.ticker}/"
                resp = self.session.get(url_std, headers=headers, timeout=10)

            if resp.status_code != 200:
                self.error_message = f"Company page returned HTTP {resp.status_code}"
                parsed_data["source"] = self.error_message
                return parsed_data

            soup = BeautifulSoup(resp.text, "html.parser")

            # 1. Top Ratios
            ratios = {}
            top_ratios_ul = soup.find("ul", {"id": "top-ratios"})
            if top_ratios_ul:
                for li in top_ratios_ul.find_all("li"):
                    name_elem = li.find("span", {"class": "name"})
                    val_elem = li.find("span", {"class": "value"}) or li.find("span", {"class": "number"})
                    if name_elem and val_elem:
                        ratios[name_elem.get_text(strip=True)] = " ".join(val_elem.get_text().split())

            # 2. Pros and Cons Extraction
            pros, cons = [], []
            analysis_div = soup.find("div", {"class": "analysis-includes"}) or soup.find("div", {"id": "analysis"})
            if analysis_div:
                for ul in analysis_div.find_all("ul", {"class": "pros"}):
                    for li in ul.find_all("li"):
                        pros.append(li.get_text(strip=True))
                for ul in analysis_div.find_all("ul", {"class": "cons"}):
                    for li in ul.find_all("li"):
                        cons.append(li.get_text(strip=True))

            # If specific class wasn't hit, parse general text blocks
            if not pros and not cons:
                for div in soup.find_all("div", {"class": "card"}):
                    text = div.get_text()
                    if "Pros" in text and "Cons" in text:
                        for li in div.find_all("li"):
                            txt = li.get_text(strip=True)
                            if any(w in txt.lower() for w in ["good", "growth", "reduced", "low", "consistent"]):
                                pros.append(txt)
                            else:
                                cons.append(txt)

            # 3. Growth Tables
            growth_tables = {}
            for tbl in soup.find_all("table", {"class": "ranges-table"}):
                th = tbl.find("th")
                title = th.get_text(strip=True) if th else "Growth Metric"
                rows = []
                for tr in tbl.find_all("tr")[1:]:
                    tds = [td.get_text(strip=True) for td in tr.find_all("td")]
                    if len(tds) == 2:
                        rows.append({"Period": tds[0].rstrip(":"), "Value": tds[1]})
                if rows:
                    growth_tables[title] = pd.DataFrame(rows)

            # 4. Profit & Loss
            pl_sec = soup.find("section", {"id": "profit-loss"})
            pl_df = self._parse_table_html(str(pl_sec.find("table"))) if (pl_sec and pl_sec.find("table")) else pd.DataFrame()

            # 5. Balance Sheet
            bs_sec = soup.find("section", {"id": "balance-sheet"})
            bs_df = self._parse_table_html(str(bs_sec.find("table"))) if (bs_sec and bs_sec.find("table")) else pd.DataFrame()

            # 6. Cash Flows
            cf_sec = soup.find("section", {"id": "cash-flow"})
            if cf_sec and cf_sec.find("table"):
                cf_df = self._parse_table_html(str(cf_sec.find("table")))
            else:
                all_tables = pd.read_html(io.StringIO(resp.text))
                cf_df = all_tables[7] if len(all_tables) > 7 else pd.DataFrame()

            # 7. Shareholding Pattern (Usually Table Index around 10 or section id 'shareholding')
            sh_sec = soup.find("section", {"id": "shareholding"})
            sh_df = self._parse_table_html(str(sh_sec.find("table"))) if (sh_sec and sh_sec.find("table")) else pd.DataFrame()
            if sh_df.empty:
                # Fallback scan all tables for promoter row
                all_tables = pd.read_html(io.StringIO(resp.text))
                for t in all_tables:
                    if any("Promoters" in str(c) for c in t.values):
                        sh_df = t
                        break

            parsed_data["ratios"] = ratios
            parsed_data["growth_tables"] = growth_tables
            parsed_data["pros"] = pros
            parsed_data["cons"] = cons
            parsed_data["profit_loss"] = pl_df
            parsed_data["balance_sheet"] = bs_df
            parsed_data["cash_flows"] = cf_df
            parsed_data["shareholding"] = sh_df
            parsed_data["source"] = f"Screener.in ({resp.url}) ✅"
            self.error_message = None

        except Exception as e:
            self.error_message = f"Parsing Error on Company Page: {str(e)}"
            parsed_data["source"] = self.error_message

        return parsed_data

    def scrape_concalls_rss(self) -> list:
        query = urllib.parse.quote(f"{self.ticker} stock exchange filing concall result")
        url = f"https://news.google.com/rss/search?q={query}&hl=en-IN&gl=IN&ceid=IN:en"
        announcements = []
        try:
            resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=6)
            if resp.status_code == 200:
                root = ET.fromstring(resp.content)
                for item in root.findall(".//item")[:10]:
                    announcements.append({
                        "Title": item.findtext("title", "No Title"),
                        "Source": item.findtext("source", "Exchange / News"),
                        "Date": item.findtext("pubDate", datetime.now().strftime("%Y-%m-%d")),
                        "Link": item.findtext("link", "#"),
                    })
        except Exception:
            pass
        return announcements


def fetch_stock_data(symbol: str, period: str = "2y", interval: str = "1d"):
    ticker_str = symbol.upper().strip()
    if not ticker_str.endswith(".NS") and not ticker_str.endswith(".BO") and not ticker_str.startswith("^"):
        ticker_str = f"{ticker_str}.NS"

    ticker = yf.Ticker(ticker_str)
    df = ticker.history(period=period, interval=interval)
    if df.empty:
        return None, None, []

    info = {}
    try:
        info = ticker.info or {}
    except Exception:
        pass

    yf_news = []
    try:
        yf_news = ticker.news or []
    except Exception:
        pass

    return df, info, yf_news


# ---------------------------------------------------------------------------
# 3. AI Scoring Engine & Buy/Hold/Exit Evaluator
# ---------------------------------------------------------------------------
def calculate_ai_score_and_recommendation(df, info, scraped_data) -> dict:
    score = 50  # Baseline neutral score out of 100
    reasons = []

    # Technical momentum weighting (RSI, Moving Averages)
    last = df.iloc[-1]
    close = last["Close"]
    sma50 = last["SMA50"]
    sma200 = last["SMA200"]
    rsi = last["RSI14"]

    if close > sma50 > sma200:
        score += 15
        reasons.append("Bullish Trend: Price is above both 50-day and 200-day moving averages.")
    elif close < sma50 < sma200:
        score -= 15
        reasons.append("Bearish Trend: Price is below key moving averages.")

    if 45 <= rsi <= 65:
        score += 10
        reasons.append(f"Healthy RSI momentum ({rsi:.1f}), neither overbought nor oversold.")
    elif rsi > 75:
        score -= 10
        reasons.append(f"Overbought warning: RSI is high at {rsi:.1f}.")

    # Fundamental weighting (ROE / P/E / Promoters)
    ratios = scraped_data.get("ratios", {})
    roe_str = ratios.get("Return on equity", "") or ratios.get("ROE", "")
    if roe_str:
        try:
            roe_val = float(roe_str.replace("%", "").strip())
            if roe_val > 15:
                score += 15
                reasons.append(f"Strong Return on Equity (ROE > 15% at {roe_val}%).")
            elif roe_val < 8:
                score -= 10
                reasons.append(f"Low Return on Equity ({roe_val}%).")
        except:
            pass

    # Recommendation mapping
    if score >= 75:
        rec = "STRONG BUY 🟢"
        action = "Accumulate on dips; favorable technical structure and robust fundamentals."
    elif score >= 60:
        rec = "BUY / ACCUMULATE 🟢"
        action = "Good growth potential; suitable for gradual positioning."
    elif score >= 45:
        rec = "HOLD 🟡"
        action = "Neutral sentiment. Maintain existing holdings and monitor support zones."
    elif score >= 30:
        rec = "REDUCE / TAKE PROFITS 🟠"
        action = "Momentum slowing down or valuation stretched; consider trimming allocation."
    else:
        rec = "EXIT / SELL 🔴"
        action = "Deteriorating technical trend or fundamental weakness. Protect capital."

    return {"score": min(max(score, 0), 100), "recommendation": rec, "action_plan": action, "reasons": reasons}


# ---------------------------------------------------------------------------
# Sidebar Setup & User Input Triggers
# ---------------------------------------------------------------------------
st.sidebar.title("🇮🇳 NSE Master Suite")
symbol_input = st.sidebar.text_input("Enter Any NSE Ticker", value="TCS", help="e.g., TCS, RELIANCE, INFY")

st.sidebar.markdown("---")
st.sidebar.subheader("🔐 Screener.in Credentials")
st.sidebar.caption("Provide credentials to pull verified live financial data from Screener.in.")
screener_user = st.sidebar.text_input("Screener Email / ID", value="", type="default")
screener_pass = st.sidebar.text_input("Screener Password", value="", type="password")

run_scrape = st.sidebar.button("🚀 Run Live Scraper & Fetch Data")

period = st.sidebar.selectbox("Lookback Period", ["6mo", "1y", "2y", "5y"], index=2)
interval = st.sidebar.selectbox("Candle Interval", ["1d", "1wk", "1h", "15m"], index=0)

st.sidebar.markdown("---")
st.sidebar.subheader("Position & ATR Risk Sizer")
capital = st.sidebar.number_input("Trading Capital (₹)", min_value=10000, value=500000, step=25000)
risk_pct = st.sidebar.slider("Risk per Trade (%)", min_value=0.25, max_value=3.0, value=1.0, step=0.25)
atr_mult = st.sidebar.slider("ATR Stop Multiplier", min_value=1.0, max_value=4.0, value=2.0, step=0.25)

if run_scrape:
    if not screener_user or not screener_pass:
        st.sidebar.error("⚠️ Please enter both Screener email and password.")
    else:
        with st.spinner(f"Authenticating and scraping Screener.in for {symbol_input.upper()}..."):
            scraper = ScreenerScraper(screener_user, screener_pass, symbol_input)
            scraped_data = scraper.scrape_company_page()
            st.session_state["scraper"] = scraper
            st.session_state["scraped_data"] = scraped_data
            st.session_state["scraped_ticker"] = symbol_input.upper().strip()

if (
    "scraped_data" in st.session_state
    and st.session_state.get("scraped_ticker") == symbol_input.upper().strip()
):
    scraped_data = st.session_state["scraped_data"]
    scraper = st.session_state.get("scraper")
else:
    scraper = ScreenerScraper("", "", symbol_input)
    scraped_data = {
        "ratios": {},
        "growth_tables": {},
        "pros": [],
        "cons": [],
        "profit_loss": pd.DataFrame(),
        "cash_flows": pd.DataFrame(),
        "balance_sheet": pd.DataFrame(),
        "shareholding": pd.DataFrame(),
        "documents": [],
        "source": "Waiting for User Input (Click 'Run Live Scraper')",
    }

df_raw, info, yf_news = fetch_stock_data(symbol_input, period=period, interval=interval)

if df_raw is None or df_raw.empty:
    st.error(f"Could not fetch NSE technical data for '{symbol_input}'. Please check the ticker symbol.")
    st.stop()

df = compute_indicators(df_raw)
last = df.iloc[-1]
prev = df.iloc[-2] if len(df) > 1 else last

price = float(last["Close"])
chg = price - float(prev["Close"])
chg_pct = (chg / float(prev["Close"])) * 100
atr_val = float(last["ATR14"]) if not np.isnan(last["ATR14"]) else price * 0.02
rsi_val = float(last["RSI14"]) if not np.isnan(last["RSI14"]) else 50.0
vol_ratio = float(last["Volume"] / last["Vol_SMA50"]) if last["Vol_SMA50"] > 0 else 1.0
tick = nse_tick_size(price)

company_name = info.get("longName", symbol_input.upper())
sector = info.get("sector", "NSE Equity")

# Compute AI Score & Suggestion
ai_analysis = calculate_ai_score_and_recommendation(df, info, scraped_data)

# Header Banner
st.title(f"{company_name} ({symbol_input.upper()}.NS)")

if scraper and scraper.logged_in and not scraper.error_message:
    st.success(f"🔒 **Status**: {scraped_data['source']} for **{symbol_input.upper()}**")
elif scraper and scraper.error_message and run_scrape:
    st.error(f"❌ **Scraper Error**: {scraper.error_message}")
else:
    st.info(f"ℹ️ **Status**: {scraped_data['source']}")

st.caption(f"Sector: **{sector}** | Last Candle: **{df.index[-1].strftime('%Y-%m-%d %H:%M')}**")

# Top KPI Metric Strip + AI Scorecard
k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Last Price (₹)", f"₹{price:,.2f}", f"{chg:+.2f} ({chg_pct:+.2f}%)")
k2.metric("AI Score", f"{ai_analysis['score']} / 100", ai_analysis["recommendation"])
k3.metric("RSI (14)", f"{rsi_val:.1f}", "Neutral" if 40 <= rsi_val <= 60 else "Extreme")
k4.metric("ATR (14)", f"₹{atr_val:.2f}", f"{(atr_val / price) * 100:.2f}% vol")
target_mean = info.get("targetMeanPrice")
if target_mean:
    upside = (target_mean / price - 1) * 100
    k5.metric("Analyst Target", f"₹{target_mean:,.2f}", f"{upside:+.1f}% upside")
else:
    k5.metric("52W Range", f"₹{df['Low'].tail(252).min():,.0f} - ₹{df['High'].tail(252).max():,.0f}")

st.markdown("---")

# ---------------------------------------------------------------------------
# Multitab Navigation Dashboard Layout
# ---------------------------------------------------------------------------
tab_overview, tab_growth, tab_cf, tab_shareholding, tab_concalls, tab_chart, tab_tech_plan, tab_news = st.tabs([
    "🤖 AI Score & Verdict",
    "📈 Growth, Pros & Cons",
    "💰 Cash Flow & P&L",
    "🏛️ Promoter Shareholding",
    "📢 Concalls & Filings",
    "📊 Technical Chart",
    "🎯 Stage-2 & Risk Sizer",
    "📰 Live News Scanner"
])

# ---------------------------------------------------------------------------
# TAB 1: AI Score & Verdict (Actionable Buy/Hold/Exit Guidance)
# ---------------------------------------------------------------------------
with tab_overview:
    st.subheader(f"🧠 AI Action Engine & Trade Verdict for {symbol_input.upper()}")
    
    col_score, col_verdict = st.columns([1, 2])
    with col_score:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Composite AI Health Score</h3>
            <h1 style="color: #00CC96; font-size: 3rem;">{ai_analysis['score']}/100</h1>
            <h4>{ai_analysis['recommendation']}</h4>
        </div>
        """, unsafe_allow_html=True)

    with col_verdict:
        st.markdown(f"#### 📋 Action Strategy")
        st.info(ai_analysis["action_plan"])
        st.markdown(f"#### 🔍 Key Evaluation Rationale")
        for r in ai_analysis["reasons"]:
            st.markdown(f"- {r}")

# ---------------------------------------------------------------------------
# TAB 2: Growth, Pros & Cons
# ---------------------------------------------------------------------------
with tab_growth:
    st.subheader(f"Fundamental Ratios, Pros & Cons: {symbol_input.upper()}")
    
    if scraper and scraper.logged_in and scraped_data["ratios"]:
        # Top Ratios
        ratio_items = list(scraped_data["ratios"].items())
        cols = st.columns(min(4, len(ratio_items)))
        for idx, (k, v) in enumerate(ratio_items):
            cols[idx % 4].metric(k, v)

        st.markdown("---")
        
        # Pros & Cons Visual Display
        pc_col1, pc_col2 = st.columns(2)
        with pc_col1:
            st.markdown("### ✅ Strengths & Pros")
            if scraped_data["pros"]:
                for pro in scraped_data["pros"]:
                    st.markdown(f'<div class="pro-box">✓ {pro}</div>', unsafe_allow_html=True)
            else:
                st.markdown("No explicit pro tags parsed.")

        with pc_col2:
            st.markdown("### ⚠️ Weaknesses & Cons")
            if scraped_data["cons"]:
                for con in scraped_data["cons"]:
                    st.markdown(f'<div class="con-box">✕ {con}</div>', unsafe_allow_html=True)
            else:
                st.markdown("No explicit con tags parsed.")

        st.markdown("---")
        if scraped_data["growth_tables"]:
            st.subheader("Compounded Sales & Profit Growth")
            g_cols = st.columns(len(scraped_data["growth_tables"]))
            for idx, (title, g_df) in enumerate(scraped_data["growth_tables"].items()):
                with g_cols[idx]:
                    st.markdown(f"**{title}**")
                    st.dataframe(g_df, hide_index=True, use_container_width=True)
    else:
        st.info("💡 Enter your Screener.in credentials in the sidebar and click **🚀 Run Live Scraper & Fetch Data**.")

# ---------------------------------------------------------------------------
# TAB 3: Cash Flow Statements & P&L
# ---------------------------------------------------------------------------
with tab_cf:
    st.subheader(f"Financial Statements (P&L & Cash Flows): {symbol_input.upper()}")
    if scraper and scraper.logged_in:
        if not scraped_data["profit_loss"].empty:
            st.markdown("### 📊 Profit & Loss Statement (₹ Cr)")
            st.dataframe(scraped_data["profit_loss"], hide_index=True, use_container_width=True)

        if not scraped_data["cash_flows"].empty:
            st.markdown("### 💰 Cash Flow Statement (₹ Cr)")
            cf_df = scraped_data["cash_flows"]
            st.dataframe(cf_df, hide_index=True, use_container_width=True)

            op_row = cf_df[cf_df["Particulars"].str.contains("Operating Activity", case=False, na=False)]
            if not op_row.empty:
                st.markdown("---")
                st.subheader("Operating Cash Flow Trend")
                years = [c for c in cf_df.columns if c != "Particulars"]
                vals = pd.to_numeric(op_row.iloc[0][years].astype(str).str.replace(",", ""), errors="coerce")
                chart_df = pd.DataFrame({"Period": years, "Operating Cash Flow (₹ Cr)": vals.values}).dropna()
                fig_cf = px.bar(chart_df, x="Period", y="Operating Cash Flow (₹ Cr)", template="plotly_dark")
                st.plotly_chart(fig_cf, use_container_width=True)
    else:
        st.warning("⚠️ Authenticate via Screener in the sidebar to load balance sheets and cash flows.")

# ---------------------------------------------------------------------------
# TAB 4: Promoter Shareholding
# ---------------------------------------------------------------------------
with tab_shareholding:
    st.subheader(f"🏛️ Shareholding Pattern (Promoters, FII, DII, Public): {symbol_input.upper()}")
    if scraper and scraper.logged_in and not scraped_data["shareholding"].empty:
        st.success("✅ Live Shareholding Table Extracted successfully!")
        st.dataframe(scraped_data["shareholding"], hide_index=True, use_container_width=True)
    else:
        st.info("💡 Run live scraper with credentials to view detailed quarterly promoter holdings, pledge ratios, and institutional changes.")

# ---------------------------------------------------------------------------
# TAB 5: Concalls & Filings
# ---------------------------------------------------------------------------
with tab_concalls:
    st.subheader(f"Corporate Filings & Concalls: {symbol_input.upper()}")
    col_doc1, col_doc2 = st.columns(2)

    with col_doc1:
        st.markdown("### 📄 Screener.in Documents & Concalls")
        if scraped_data["documents"]:
            for doc in scraped_data["documents"][:15]:
                st.markdown(f"- [{doc['Title']}]({doc['Link']})")
        else:
            st.info("Run scraper to pull direct PDF reports.")

    with col_doc2:
        st.markdown("### 🏛️ Exchange Filings & Earnings Call News")
        announcements = scraper.scrape_concalls_rss() if scraper else []
        for item in announcements:
            st.markdown(f"**[{item['Title']}]({item['Link']})**")
            st.caption(f"{item['Source']} • {item['Date']}")

# ---------------------------------------------------------------------------
# TAB 6: Technical Chart
# ---------------------------------------------------------------------------
with tab_chart:
    plot_df = df.tail(180)
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.04,
        row_heights=[0.60, 0.18, 0.22],
        subplot_titles=("Price, Moving Averages & Bollinger Bands", "Volume", "RSI (14) & MACD")
    )
    fig.add_trace(go.Candlestick(x=plot_df.index, open=plot_df["Open"], high=plot_df["High"], low=plot_df["Low"], close=plot_df["Close"], name="OHLC"), row=1, col=1)
    for col_name, color, dash in [("EMA21", "#00CC96", "solid"), ("SMA50", "#FFA15A", "solid"), ("SMA200", "#EF553B", "dash")]:
        fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df[col_name], line=dict(color=color, width=1.4, dash=dash), name=col_name), row=1, col=1)
    vol_colors = ["#00CC96" if c >= o else "#EF553B" for c, o in zip(plot_df["Close"], plot_df["Open"])]
    fig.add_trace(go.Bar(x=plot_df.index, y=plot_df["Volume"], marker_color=vol_colors, name="Volume", showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=plot_df.index, y=plot_df["RSI14"], line=dict(color="#AB63FA", width=1.5), name="RSI 14"), row=3, col=1)
    fig.add_hline(y=70, line_dash="dot", line_color="red", row=3, col=1)
    fig.add_hline(y=30, line_dash="dot", line_color="green", row=3, col=1)
    fig.update_layout(height=720, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=10, r=10, t=35, b=10))
    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------------------
# TAB 7: Stage-2 & Risk Sizer
# ---------------------------------------------------------------------------
with tab_tech_plan:
    c_left, c_right = st.columns(2)
    with c_left:
        st.subheader("Stage-2 Trend Template Check")
        tt = evaluate_stage2_template(df)
        st.progress(tt["score"] / tt["total"], text=f"Score: {tt['score']} / {tt['total']} Criteria Passed")
        for label, passed in tt["checks"].items():
            st.write(f"{'✅' if passed else '❌'} **{label}**")

        st.markdown("---")
        st.subheader("Key Support, Resistance & Pivots")
        pivots = find_pivot_levels(df)
        piv_df = pd.DataFrame([{"Level": k, "Price (₹)": f"₹{v:,.2f}", "Dist from CMP": f"{(v / price - 1) * 100:+.2f}%"} for k, v in pivots.items()])
        st.dataframe(piv_df, hide_index=True, use_container_width=True)

    with c_right:
        st.subheader("ATR Position Sizing & Stop Calculator")
        stop_price = round((price - atr_mult * atr_val) / tick) * tick
        risk_per_share = max(price - stop_price, tick)
        risk_budget = capital * (risk_pct / 100.0)
        qty = int(risk_budget // risk_per_share)
        pos_value = qty * price
        target_2r = round((price + 2 * risk_per_share) / tick) * tick
        target_3r = round((price + 3 * risk_per_share) / tick) * tick
        st.info(
            f"**Entry (CMP):** ₹{price:,.2f}  \n"
            f"**ATR Stop ({atr_mult}x ATR):** ₹{stop_price:,.2f} (-{(risk_per_share/price)*100:.2f}%)  \n"
            f"**Suggested Qty:** **{qty} shares** (Capital Used: ₹{pos_value:,.0f} | Max Risk: ₹{qty * risk_per_share:,.0f})  \n"
            f"**2R Target:** ₹{target_2r:,.2f} | **3R Target:** ₹{target_3r:,.2f}"
        )

# ---------------------------------------------------------------------------
# TAB 8: Live News Scanner
# ---------------------------------------------------------------------------
with tab_news:
    st.subheader(f"Live News Scanner for {symbol_input.upper()}")
    if yf_news:
        for item in yf_news[:10]:
            content = item.get("content", item)
            title = content.get("title", "Update")
            link = content.get("canonicalUrl", {}).get("url") or content.get("link", "#")
            st.markdown(f"**[{title}]({link})**")
    else:
        st.info("No direct Yahoo Finance news feed items returned.")
