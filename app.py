import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

st.set_page_config(page_title="N100 Financial Intelligence Platform", page_icon="📈", layout="wide")

@st.cache_data
def load_data():
    files_cfg = {
        "sectors": ("sectors.xlsx", 0),
        "peer_groups": ("peer_groups.xlsx", 0),
        "stock_prices": ("stock_prices.xlsx", 0),
        "market_cap": ("market_cap.xlsx", 0),
        "financial_ratios": ("financial_ratios.xlsx", 0),
        "companies": ("companies.xlsx", 1),
        "balancesheet": ("balancesheet.xlsx", 1),
        "profitandloss": ("profitandloss.xlsx", 1),
        "cashflow": ("cashflow.xlsx", 1),
        "analysis": ("analysis.xlsx", 1),
        "prosandcons": ("prosandcons.xlsx", 1),
        "documents": ("documents.xlsx", 1),
    }
    dfs = {}
    for k, (f, h) in files_cfg.items():
        if os.path.exists(f):
            df = pd.read_excel(f, header=h)
            df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]
            dfs[k] = df

    dfs["companies"] = dfs["companies"].rename(columns={"id": "company_id"})
    latest_mcap = dfs["market_cap"].sort_values("year").groupby("company_id").last().reset_index()
    latest_ratios = dfs["financial_ratios"].sort_values("year").groupby("company_id").last().reset_index()

    master = dfs["companies"].merge(
        dfs["sectors"][['company_id', 'broad_sector', 'sub_sector', 'index_weight_pct']], on="company_id", how="left"
    ).merge(
        latest_mcap[['company_id', 'market_cap_crore', 'pe_ratio', 'pb_ratio', 'ev_ebitda', 'dividend_yield_pct']], on="company_id", how="left"
    ).merge(
        latest_ratios[['company_id', 'net_profit_margin_pct', 'return_on_equity_pct', 'debt_to_equity', 'free_cash_flow_cr']], on="company_id", how="left"
    )
    return dfs, master

dfs, master_df = load_data()

st.title("📈 N100 Financial Intelligence Platform")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Market Overview", "Screener", "Stock Deep Dive", "DCF Valuation", "Peer Clustering & PCA"
])

# TAB 1: MARKET OVERVIEW
with tab1:
    st.header("Nifty 100 Landscape")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Companies Analyzed", len(master_df))
    c2.metric("Median P/E", f"{master_df['pe_ratio'].median():.2f}")
    c3.metric("Median ROE (%)", f"{master_df['return_on_equity_pct'].median():.2f}%")
    c4.metric("Total Market Cap (Cr)", f"₹{master_df['market_cap_crore'].sum():,.0f}")

    colA, colB = st.columns([1, 1])
    with colA:
        st.subheader("Sector Weight Distribution (%)")
        sec_weights = dfs["sectors"].groupby("broad_sector")["index_weight_pct"].sum().reset_index()
        fig_sec = px.pie(sec_weights, names="broad_sector", values="index_weight_pct", hole=0.45)
        st.plotly_chart(fig_sec, width="stretch")

    with colB:
        st.subheader("P/E Ratio vs. Return on Equity (ROE)")
        clean_p = master_df.dropna(subset=["pe_ratio", "return_on_equity_pct"])
        # Filter outliers for clean visualization
        clean_p = clean_p[(clean_p["pe_ratio"] > 0) & (clean_p["pe_ratio"] <= 120) & (clean_p["return_on_equity_pct"] >= -20) & (clean_p["return_on_equity_pct"] <= 100)]
        fig = px.scatter(clean_p, x="pe_ratio", y="return_on_equity_pct", color="broad_sector", size="market_cap_crore", hover_name="company_id", labels={"pe_ratio": "P/E Ratio", "return_on_equity_pct": "ROE (%)"})
        st.plotly_chart(fig, width="stretch")

# TAB 2: SCREENER
with tab2:
    st.header("Fundamental Screener")
    colA, colB, colC = st.columns(3)
    min_roe = colA.slider("Minimum ROE (%)", 0.0, 40.0, 15.0)
    max_pe = colB.slider("Maximum P/E", 5.0, 80.0, 35.0)
    max_de = colC.slider("Maximum Debt-to-Equity", 0.0, 5.0, 1.0)

    screened = master_df[
        (master_df["return_on_equity_pct"] >= min_roe) & 
        (master_df["pe_ratio"] > 0) & (master_df["pe_ratio"] <= max_pe) &
        (master_df["debt_to_equity"] <= max_de)
    ].sort_values("return_on_equity_pct", ascending=False)

    st.success(f"Matched {len(screened)} companies")
    st.dataframe(screened[["company_id", "company_name", "broad_sector", "pe_ratio", "return_on_equity_pct", "debt_to_equity", "market_cap_crore"]], width="stretch")

# TAB 3: STOCK DEEP DIVE
with tab3:
    st.header("Company Deep Dive")
    tkr = st.selectbox("Select Ticker", sorted(master_df["company_id"].unique()))
    c_info = master_df[master_df["company_id"] == tkr].iloc[0]
    st.write(f"**{c_info.get('company_name')}** | Sector: `{c_info.get('broad_sector')}` | Sub-sector: `{c_info.get('sub_sector')}`")
    st.write(c_info.get("about_company", ""))

    c_row1, c_row2, c_row3, c_row4 = st.columns(4)
    c_row1.metric("Market Cap", f"₹{c_info.get('market_cap_crore', 0):,.0f} Cr")
    c_row2.metric("P/E Ratio", f"{c_info.get('pe_ratio', np.nan):.2f}")
    c_row3.metric("Book Value", f"₹{c_info.get('book_value', np.nan)}")
    c_row4.metric("ROCE (%)", f"{c_info.get('roce_percentage', np.nan)}%")

    p_df = dfs["stock_prices"][dfs["stock_prices"]["company_id"] == tkr].sort_values("date")
    if not p_df.empty:
        p_df["date"] = pd.to_datetime(p_df["date"])
        p_df["10_ma"] = p_df["close_price"].rolling(10).mean()
        fig_p = px.line(p_df, x="date", y=["close_price", "10_ma"], title=f"{tkr} Price Performance & 10-Period MA")
        st.plotly_chart(fig_p, width="stretch")

    pros = dfs["prosandcons"][dfs["prosandcons"]["company_id"] == tkr]
    if not pros.empty:
        colP, colC = st.columns(2)
        with colP:
            st.markdown(f"🟢 **Pro:** {pros.iloc[0].get('pros')}")
        with colC:
            st.markdown(f"🔴 **Con:** {pros.iloc[0].get('cons')}")

# TAB 4: DCF VALUATION
with tab4:
    st.header("Intrinsic DCF Valuation")
    dcf_tkr = st.selectbox("Select Stock for DCF", sorted(master_df["company_id"].unique()), key="dcf_sel")
    cf_df = dfs["cashflow"][dfs["cashflow"]["company_id"] == dcf_tkr].sort_values("year")
    mcap_val = master_df[master_df["company_id"] == dcf_tkr].iloc[0]["market_cap_crore"]

    colW, colG, colT = st.columns(3)
    wacc = colW.slider("WACC (%)", 8.0, 16.0, 11.0, 0.5) / 100.0
    f_growth = colG.slider("FCF Growth (%)", 2.0, 25.0, 10.0, 0.5) / 100.0
    term_g = colT.slider("Terminal Growth (%)", 2.0, 6.0, 4.0, 0.25) / 100.0

    if not cf_df.empty:
        latest = cf_df.iloc[-1]
        base_fcf = latest.get("operating_activity", 0) + latest.get("investing_activity", 0)
        if base_fcf <= 0:
            base_fcf = abs(base_fcf) * 0.5

        proj = [base_fcf * ((1 + f_growth) ** i) for i in range(1, 6)]
        disc = [c / ((1 + wacc) ** i) for i, c in enumerate(proj, 1)]
        tv = (proj[-1] * (1 + term_g)) / (wacc - term_g)
        pv_tv = tv / ((1 + wacc) ** 5)
        ev = sum(disc) + pv_tv
        upside = ((ev - mcap_val) / mcap_val) * 100

        d1, d2, d3 = st.columns(3)
        d1.metric("Base FCF", f"₹{base_fcf:,.1f} Cr")
        d2.metric("DCF Intrinsic EV", f"₹{ev:,.1f} Cr")
        d3.metric("Implied Upside", f"{upside:+.1f}%")

# TAB 5: PEER CLUSTERING & PCA
with tab5:
    st.header("Unsupervised Fundamental Clustering")
    feats = ["pe_ratio", "pb_ratio", "return_on_equity_pct", "debt_to_equity"]
    c_df = master_df.dropna(subset=feats).copy()
    c_df = c_df[(c_df["return_on_equity_pct"] >= -20) & (c_df["return_on_equity_pct"] <= 100) & (c_df["pe_ratio"] <= 120)]

    k_val = st.slider("Number of Clusters (k)", 2, 6, 4)
    X = StandardScaler().fit_transform(c_df[feats])
    c_df["cluster"] = KMeans(n_clusters=k_val, random_state=42, n_init=10).fit_predict(X).astype(str)
    coords = PCA(n_components=2).fit_transform(X)
    c_df["pca1"] = coords[:, 0]
    c_df["pca2"] = coords[:, 1]

    fig_cl = px.scatter(c_df, x="pca1", y="pca2", color="cluster", hover_name="company_id", title="2D PCA Fundamental Cohorts", labels={"pca1": "Principal Component 1", "pca2": "Principal Component 2"})
    st.plotly_chart(fig_cl, width="stretch")
    st.write("Cohort Medians:")
    st.dataframe(c_df.groupby("cluster")[feats].median().round(2), width="stretch")
