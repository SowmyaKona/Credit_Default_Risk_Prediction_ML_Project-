import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Credit Risk Predictor", page_icon="💳", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
.stApp { background: #0d1117; color: #e6edf3; }
#MainMenu, footer, header { visibility: hidden; }

.hero {
    background: linear-gradient(135deg, #161b22 0%, #1c2333 60%, #0d1117 100%);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 36px 40px 28px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(88,166,255,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title { font-size: 2.2rem; font-weight: 700; color: #e6edf3; letter-spacing: -0.5px; margin: 0 0 6px; }
.hero-title span { color: #58a6ff; }
.hero-subtitle { font-size: 0.95rem; color: #8b949e; margin: 0; }

.info-box {
    background: rgba(56,139,253,0.08);
    border: 1px solid rgba(56,139,253,0.25);
    border-radius: 8px;
    padding: 12px 16px;
    font-size: 0.83rem;
    color: #79c0ff;
    margin-bottom: 24px;
    line-height: 1.5;
}

.section-card {
    background: #161b22;
    border: 1px solid #30363d;
    border-radius: 12px;
    padding: 24px 28px;
    margin-bottom: 16px;
}
.section-title {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 1.2px;
    text-transform: uppercase;
    color: #58a6ff;
    margin: 0 0 16px;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-title::after { content: ''; flex: 1; height: 1px; background: #21262d; }

div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div:first-child {
    background: #0d1117 !important;
    border-color: #30363d !important;
    border-radius: 8px !important;
    color: #e6edf3 !important;
}
div[data-baseweb="input"]:focus-within > div,
div[data-baseweb="select"]:focus-within > div:first-child {
    border-color: #58a6ff !important;
    box-shadow: 0 0 0 3px rgba(88,166,255,0.15) !important;
}
label[data-testid="stWidgetLabel"] p {
    font-size: 0.80rem !important;
    font-weight: 500 !important;
    color: #8b949e !important;
    text-transform: uppercase !important;
    letter-spacing: 0.5px !important;
}

div.stButton > button {
    background: #238636 !important;
    color: #fff !important;
    border: 1px solid #2ea043 !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 1rem !important;
    padding: 14px 0 !important;
    width: 100% !important;
    transition: all 0.2s !important;
}
div.stButton > button:hover {
    background: #2ea043 !important;
    box-shadow: 0 0 20px rgba(46,160,67,0.35) !important;
    transform: translateY(-1px) !important;
}

.result-high {
    background: rgba(248,81,73,0.08);
    border: 1px solid rgba(248,81,73,0.35);
    border-left: 4px solid #f85149;
    border-radius: 12px;
    padding: 24px 28px;
    text-align: center;
    margin-bottom: 16px;
}
.result-low {
    background: rgba(46,160,67,0.08);
    border: 1px solid rgba(46,160,67,0.35);
    border-left: 4px solid #2ea043;
    border-radius: 12px;
    padding: 24px 28px;
    text-align: center;
    margin-bottom: 16px;
}
.result-mid {
    background: rgba(210,153,34,0.08);
    border: 1px solid rgba(210,153,34,0.35);
    border-left: 4px solid #d2991f;
    border-radius: 12px;
    padding: 24px 28px;
    text-align: center;
    margin-bottom: 16px;
}
.result-label { font-size: 1.2rem; font-weight: 700; margin: 0 0 4px; }
.result-prob  { font-family: 'DM Mono', monospace; font-size: 3rem; font-weight: 500; letter-spacing: -2px; margin: 6px 0; }
.result-desc  { font-size: 0.82rem; color: #8b949e; margin: 0; }
.result-high .result-label, .result-high .result-prob { color: #f85149; }
.result-low  .result-label, .result-low  .result-prob { color: #3fb950; }
.result-mid  .result-label, .result-mid  .result-prob { color: #d2991f; }

.prob-bar-wrap { background: #21262d; border-radius: 99px; height: 8px; margin: 14px 0 4px; overflow: hidden; }
.prob-bar      { height: 8px; border-radius: 99px; transition: width 0.6s ease; }

.metric-row { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 12px; }
.metric-chip { background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 10px 16px; flex: 1; min-width: 110px; }
.metric-chip .m-label { font-size: 0.70rem; color: #6e7681; text-transform: uppercase; letter-spacing: 0.6px; }
.metric-chip .m-value { font-size: 1.05rem; font-weight: 600; color: #e6edf3; font-family: 'DM Mono', monospace; margin-top: 3px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return pickle.load(open("model.pkl", "rb"))

model = load_model()

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="hero-title">💳 Credit Risk <span>Predictor</span></div>
  <p class="hero-subtitle">Enter a customer's last 6 months of credit history to predict the probability of defaulting next month.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="info-box">
  ℹ️ <strong>How to fill this form:</strong>
  "Last Month" = your most recent month, "2 Months Ago" = the month before that, and so on.
  All 6 months are required — the model uses <strong>repayment status, bill amounts, and payment amounts</strong>
  together to predict whether the customer will default <strong>next month</strong>.
</div>
""", unsafe_allow_html=True)

# Relative month labels — no hardcoded dates
month_labels = ["Last Month", "2 Months Ago", "3 Months Ago", "4 Months Ago", "5 Months Ago", "6 Months Ago"]

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — Customer Profile
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">👤 &nbsp;Customer Profile</div>', unsafe_allow_html=True)

c1, c2, c3, c4, c5 = st.columns(5)
with c1:
    LIMIT_BAL = st.number_input("Credit Limit (NT$)", min_value=0, step=10000, value=200000)
with c2:
    SEX = st.selectbox("Gender", ["Female", "Male"])
with c3:
    EDUCATION = st.selectbox("Education", [1, 2, 3, 4],
        format_func=lambda x: {1:"Graduate", 2:"University", 3:"High School", 4:"Other"}[x])
with c4:
    MARRIAGE = st.selectbox("Marital Status", [1, 2, 3],
        format_func=lambda x: {1:"Married", 2:"Single", 3:"Other"}[x])
with c5:
    AGE = st.number_input("Age", min_value=18, max_value=100, step=1, value=30)

st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — Repayment Status
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">📅 &nbsp;Repayment Status — Last 6 Months</div>', unsafe_allow_html=True)
st.caption("**Scale:** -2 = No consumption · -1 = Paid in full · 0 = Minimum paid · 1 to 8 = Months of delay")

pay_keys = ["PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]
pay_cols = st.columns(6)
pay_vals = {}
for col, label, k in zip(pay_cols, month_labels, pay_keys):
    with col:
        pay_vals[k] = st.number_input(label, min_value=-2, max_value=9, step=1, value=0, key=f"pay_{k}")

st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — Bill Amounts
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">🧾 &nbsp;Bill Statement Amount — NT$ (Last 6 Months)</div>', unsafe_allow_html=True)

bill_keys = ["BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6"]
bill_cols = st.columns(6)
bill_vals = {}
for col, label, k in zip(bill_cols, month_labels, bill_keys):
    with col:
        bill_vals[k] = st.number_input(label, min_value=0, step=500, value=0, key=f"bill_{k}")

st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — Payment Amounts
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-card">', unsafe_allow_html=True)
st.markdown('<div class="section-title">💰 &nbsp;Previous Payment Amount — NT$ (Last 6 Months)</div>', unsafe_allow_html=True)

pamt_keys = ["PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]
pamt_cols = st.columns(6)
pamt_vals = {}
for col, label, k in zip(pamt_cols, month_labels, pamt_keys):
    with col:
        pamt_vals[k] = st.number_input(label, min_value=0, step=500, value=0, key=f"pamt_{k}")

st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════════════════════
_, btn_col, _ = st.columns([1.5, 2, 1.5])
with btn_col:
    predict_clicked = st.button("🔍  Predict Default Risk")

if predict_clicked:
    SEX_enc = 1 if SEX == "Male" else 2

    data = pd.DataFrame([[
        LIMIT_BAL, SEX_enc, EDUCATION, MARRIAGE, AGE,
        pay_vals["PAY_0"],  pay_vals["PAY_2"],  pay_vals["PAY_3"],
        pay_vals["PAY_4"],  pay_vals["PAY_5"],  pay_vals["PAY_6"],
        bill_vals["BILL_AMT1"], bill_vals["BILL_AMT2"], bill_vals["BILL_AMT3"],
        bill_vals["BILL_AMT4"], bill_vals["BILL_AMT5"], bill_vals["BILL_AMT6"],
        pamt_vals["PAY_AMT1"], pamt_vals["PAY_AMT2"], pamt_vals["PAY_AMT3"],
        pamt_vals["PAY_AMT4"], pamt_vals["PAY_AMT5"], pamt_vals["PAY_AMT6"],
    ]], columns=[
        'LIMIT_BAL','SEX','EDUCATION','MARRIAGE','AGE',
        'PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6',
        'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',
        'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6',
    ])

    # Feature engineering — exact notebook column order
    data['total_bill']    = data[['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']].sum(axis=1)
    data['avg_bill']      = data[['BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6']].mean(axis=1)
    data['bill_trend']    = data['BILL_AMT1'] - data['BILL_AMT6']
    data['util_rate']     = data['total_bill'] / (data['LIMIT_BAL'] + 1)
    data['total_pay']     = data[['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']].sum(axis=1)
    data['avg_pay']       = data[['PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6']].mean(axis=1)
    data['pay_ratio']     = data['total_pay'] / (data['total_bill'] + 1)
    data['avg_pay_delay'] = data[['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']].mean(axis=1)
    data['max_pay_delay'] = data[['PAY_0','PAY_2','PAY_3','PAY_4','PAY_5','PAY_6']].max(axis=1)

    # Log-transform — matches notebook preprocessing
    log_cols = [
        'PAY_AMT1','PAY_AMT2','PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6',
        'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6',
        'total_bill','avg_bill','total_pay','avg_pay','LIMIT_BAL'
    ]
    for col in log_cols:
        data[col] = np.log1p(data[col].clip(lower=0))

    prediction = model.predict(data)[0]
    prob       = model.predict_proba(data)[0][1]
    prob_pct   = prob * 100

    raw_total_bill = sum(bill_vals.values())
    raw_total_pay  = sum(pamt_vals.values())
    util_pct       = (raw_total_bill / (LIMIT_BAL + 1)) * 100
    max_delay      = max(pay_vals.values())
    pay_bill_ratio = raw_total_pay / (raw_total_bill + 1)

    st.markdown("---")
    st.markdown("### 📊 Prediction Result")

    # Three-tier result: low / medium / high
    if prob_pct < 40:
        bar_color = "#3fb950"
        card_class = "result-low"
        label = "✅ Low Risk — Unlikely to Default"
        advice = "This customer shows low default risk. Standard credit terms are appropriate."
    elif prob_pct < 65:
        bar_color = "#d2991f"
        card_class = "result-mid"
        label = "⚡ Medium Risk — Monitor Closely"
        advice = "Borderline case. Consider a credit review or reduced limit before approving."
    else:
        bar_color = "#f85149"
        card_class = "result-high"
        label = "⚠️ High Risk — Likely to Default"
        advice = "High default probability. Credit extension or limit increase is not recommended."

    st.markdown(f"""
    <div class="{card_class}">
      <div class="result-label">{label}</div>
      <div class="result-prob">{prob:.1%}</div>
      <div class="prob-bar-wrap">
        <div class="prob-bar" style="width:{prob_pct:.1f}%; background:{bar_color};"></div>
      </div>
      <div class="result-desc">{advice}</div>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-chip">
        <div class="m-label">Total Bills (6M)</div>
        <div class="m-value">NT$ {raw_total_bill:,.0f}</div>
      </div>
      <div class="metric-chip">
        <div class="m-label">Total Payments (6M)</div>
        <div class="m-value">NT$ {raw_total_pay:,.0f}</div>
      </div>
      <div class="metric-chip">
        <div class="m-label">Utilization Rate</div>
        <div class="m-value">{util_pct:.1f}%</div>
      </div>
      <div class="metric-chip">
        <div class="m-label">Max Pay Delay</div>
        <div class="m-value">{max_delay} mo</div>
      </div>
      <div class="metric-chip">
        <div class="m-label">Pay / Bill Ratio</div>
        <div class="m-value">{pay_bill_ratio:.2f}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)