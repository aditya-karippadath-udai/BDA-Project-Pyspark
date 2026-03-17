"""
DiabetesScan — Random Forest Diabetes Predictor
Inputs: HbA1c level, Age, BMI, Gender
Trains RF on-the-fly from diabetes_dataset.csv (no saved models)
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DiabetesScan",
    page_icon="🩺",
    layout="centered",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700;800&family=JetBrains+Mono:wght@300;400;500&display=swap');

:root {
    --bg:     #f5f0e8;
    --card:   #ffffff;
    --ink:    #1a1410;
    --muted:  #7a6f62;
    --teal:   #007a6e;
    --teal2:  #00a896;
    --amber:  #d47c0a;
    --red:    #c0392b;
    --green:  #1a7a3c;
    --border: #e0d8cc;
    --r:      14px;
}

html, body, [class*="css"] {
    background: var(--bg) !important;
    color: var(--ink) !important;
    font-family: 'Outfit', sans-serif !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 1.5rem !important; max-width: 720px !important; margin: 0 auto; }

/* Inputs */
.stSlider > div > div > div > div { background: var(--teal) !important; }
.stSelectbox > div > div {
    background: var(--card) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: 10px !important;
    color: var(--ink) !important;
}
label { color: var(--ink) !important; font-weight: 600 !important; font-size: 0.9rem !important; }

/* Button */
.stButton > button {
    width: 100% !important;
    background: var(--teal) !important;
    color: #fff !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1.05rem !important;
    letter-spacing: 0.03em !important;
    padding: 0.75rem !important;
    border: none !important;
    border-radius: 50px !important;
    box-shadow: 0 4px 20px rgba(0,122,110,0.3) !important;
    transition: all 0.15s !important;
    margin-top: 0.5rem !important;
}
.stButton > button:hover {
    background: var(--teal2) !important;
    box-shadow: 0 6px 28px rgba(0,168,150,0.4) !important;
    transform: translateY(-1px) !important;
}

/* Radio */
.stRadio > label { font-weight: 600 !important; }
.stRadio > div { gap: 1rem !important; }

/* Divider */
hr { border-color: var(--border) !important; margin: 1.2rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ── HEADER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; padding: 2rem 0 1.5rem;">
  <div style="display:inline-block; background:#007a6e; color:#fff;
              font-family:'JetBrains Mono',monospace; font-size:0.7rem;
              letter-spacing:0.14em; padding:4px 16px; border-radius:50px;
              margin-bottom:1rem;"> · LIVE TRAINING</div>
  <h1 style="font-family:'Outfit',sans-serif; font-size:2.8rem; font-weight:800;
             color:#1a1410; margin:0; letter-spacing:-0.03em; line-height:1.1;">
    Diabetes<span style="color:#007a6e;">Scan</span>
  </h1>
  <p style="color:#7a6f62; font-size:1rem; margin: 0.6rem 0 0; font-weight:300;">
    Enter 4 clinical values — get an instant prediction
  </p>
</div>
""", unsafe_allow_html=True)


# ── TRAIN MODEL (cached) ──────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def train_rf():
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    from imblearn.over_sampling import SMOTE

    df = pd.read_csv("diabetes_dataset.csv").dropna()
    df["gender_Male"] = (df["gender"] == "Male").astype(int)

    FEATURES = ["hbA1c_level", "age", "bmi", "gender_Male"]
    X = df[FEATURES].values
    y = df["diabetes"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    sm = SMOTE(random_state=42)
    X_tr_sm, y_tr_sm = sm.fit_resample(X_train, y_train)

    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr_sm)
    X_te_sc = scaler.transform(X_test)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=10, random_state=42, n_jobs=-1
    )
    rf.fit(X_tr_sc, y_tr_sm)

    acc = accuracy_score(y_test, rf.predict(X_te_sc))
    return rf, scaler, acc, X_tr_sc, y_tr_sm


# ── LOAD MODEL ────────────────────────────────────────────────────────────────
with st.spinner("🌲  Preparing model … please wait"):
    try:
        rf, scaler, train_acc, X_tr, y_tr = train_rf()
        model_ready = True
    except FileNotFoundError:
        model_ready = False

if not model_ready:
    st.error("❌  `diabetes_dataset.csv` not found. Make sure it is committed to the repo root.")
    st.stop()


# ── INPUT FORM ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#fff; border:1.5px solid #e0d8cc; border-radius:16px;
            padding:1.8rem 2rem; margin-bottom:1.2rem;
            box-shadow: 0 2px 16px rgba(0,0,0,0.06);">
  <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#7a6f62;
              letter-spacing:0.12em; margin-bottom:1rem;">PATIENT INPUTS</div>
""", unsafe_allow_html=True)

c1, c2 = st.columns(2)

with c1:
    hba1c = st.slider(
        "HbA1c Level (%)",
        min_value=3.5, max_value=9.0, value=5.7, step=0.1,
        help="Glycated haemoglobin. ≥6.5% = diabetic range."
    )
    bmi = st.slider(
        "BMI",
        min_value=10.0, max_value=60.0, value=27.0, step=0.1,
        help="Body Mass Index. Normal: 18.5–24.9"
    )

with c2:
    age = st.slider(
        "Age (years)",
        min_value=1, max_value=100, value=45,
        help="Patient age in years."
    )
    gender = st.radio(
        "Gender",
        options=["Female", "Male"],
        horizontal=True,
        help="Used to encode gender_Male feature."
    )

st.markdown("</div>", unsafe_allow_html=True)

predict_btn = st.button("🔬  Predict Diabetes Risk")


# ── PREDICTION ────────────────────────────────────────────────────────────────
if predict_btn:
    gender_male = 1 if gender == "Male" else 0
    X_input = np.array([[hba1c, age, bmi, gender_male]], dtype=np.float32)
    X_sc    = scaler.transform(X_input)

    prob  = rf.predict_proba(X_sc)[0][1]
    label = int(prob >= 0.5)

    if prob < 0.3:
        risk_label, risk_color, bg_color, border_color, icon = \
            "LOW RISK", "#1a7a3c", "#edf7f0", "#a8d8b8", "✅"
        desc = "Your values are within normal ranges. Maintain healthy habits."
    elif prob < 0.6:
        risk_label, risk_color, bg_color, border_color, icon = \
            "MODERATE RISK", "#d47c0a", "#fef9ec", "#f5d98a", "⚠️"
        desc = "Some elevated markers detected. Consider consulting a healthcare provider."
    else:
        risk_label, risk_color, bg_color, border_color, icon = \
            "HIGH RISK", "#c0392b", "#fdf0ef", "#f0a99e", "🔴"
        desc = "Multiple risk factors detected. Clinical evaluation is strongly recommended."

    # ── Result card ──
    st.markdown(f"""
    <div style="background:{bg_color}; border:2px solid {border_color};
                border-radius:16px; padding:1.8rem 2rem; margin-top:0.8rem;
                box-shadow: 0 4px 20px rgba(0,0,0,0.08);">
      <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem;
                  color:{risk_color}; letter-spacing:0.12em; margin-bottom:0.4rem;">
        PREDICTION RESULT
      </div>
      <div style="display:flex; align-items:center; gap:0.8rem; margin-bottom:0.5rem;">
        <span style="font-size:2rem;">{icon}</span>
        <span style="font-family:'Outfit',sans-serif; font-size:2rem; font-weight:800;
                     color:{risk_color}; letter-spacing:-0.02em;">{risk_label}</span>
      </div>
      <p style="color:#4a4035; font-size:0.92rem; margin:0 0 1.2rem;">{desc}</p>
      <div style="display:flex; gap:1rem; flex-wrap:wrap; margin-bottom:1rem;">
        <div style="flex:1; min-width:120px; background:rgba(255,255,255,0.7);
                    border-radius:10px; padding:0.8rem 1rem; text-align:center;">
          <div style="font-family:'JetBrains Mono',monospace; font-size:1.6rem;
                      font-weight:500; color:{risk_color};">{prob:.1%}</div>
          <div style="font-size:0.75rem; color:#7a6f62; margin-top:2px;">Risk Probability</div>
        </div>
        <div style="flex:1; min-width:120px; background:rgba(255,255,255,0.7);
                    border-radius:10px; padding:0.8rem 1rem; text-align:center;">
          <div style="font-family:'JetBrains Mono',monospace; font-size:1.6rem;
                      font-weight:500; color:#1a1410;">{'Yes' if label else 'No'}</div>
          <div style="font-size:0.75rem; color:#7a6f62; margin-top:2px;">Diabetes Predicted</div>
        </div>
      </div>
      <div style="background:rgba(255,255,255,0.5); border-radius:50px; height:8px;">
        <div style="background:{risk_color}; height:8px; border-radius:50px;
                    width:{prob*100:.1f}%; transition:width 0.4s;"></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Input vs thresholds ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#7a6f62;
                letter-spacing:0.1em; margin-bottom:0.5rem;">YOUR INPUTS vs CLINICAL THRESHOLDS</div>
    """, unsafe_allow_html=True)

    thresholds = {
        "HbA1c Level (%)": (hba1c, 6.5,  "≥6.5% = Diabetic range (ADA 2024)",  "%"),
        "Age (years)":     (age,   45,   "Risk increases significantly after 45", ""),
        "BMI":             (bmi,   30.0, "≥30 = Obese, elevated diabetes risk",   ""),
    }

    for label_t, (val, thresh, note, unit) in thresholds.items():
        above     = val >= thresh
        bar_color = "#c0392b" if above else "#007a6e"
        bar_pct   = min(val / (thresh * 1.5) * 100, 100)
        flag      = "⚠️" if above else "✓"
        st.markdown(f"""
        <div style="background:#fff; border:1.5px solid #e0d8cc; border-radius:10px;
                    padding:0.8rem 1.1rem; margin-bottom:0.5rem;">
          <div style="display:flex; justify-content:space-between; margin-bottom:5px;">
            <span style="font-size:0.85rem; font-weight:600; color:#1a1410;">{label_t}</span>
            <span style="font-family:'JetBrains Mono',monospace; font-size:0.85rem;
                         color:{bar_color}; font-weight:500;">{flag} {val}{unit}</span>
          </div>
          <div style="background:#e0d8cc; border-radius:50px; height:5px; margin-bottom:4px;">
            <div style="background:{bar_color}; height:5px; border-radius:50px;
                        width:{bar_pct:.0f}%;"></div>
          </div>
          <div style="font-size:0.72rem; color:#7a6f62;">{note}</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Feature importance chart ──
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-family:'JetBrains Mono',monospace; font-size:0.68rem; color:#7a6f62;
                letter-spacing:0.1em; margin-bottom:0.5rem;">FEATURE IMPORTANCE (RANDOM FOREST)</div>
    """, unsafe_allow_html=True)

    FEATURE_NAMES = ["HbA1c Level", "Age", "BMI", "Gender (Male)"]
    importances   = rf.feature_importances_
    order         = np.argsort(importances)[::-1]

    fig, ax = plt.subplots(figsize=(6, 2.8))
    fig.patch.set_facecolor('#ffffff')
    ax.set_facecolor('#f5f0e8')

    palette       = ['#007a6e', '#4a9e94', '#7abfba', '#aad5d1']
    sorted_colors = [palette[i] for i in range(len(importances))]

    bars = ax.barh(
        [FEATURE_NAMES[i] for i in order],
        importances[order],
        color=sorted_colors,
        height=0.5,
        edgecolor='none',
    )
    for bar, val in zip(bars, importances[order]):
        ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', color='#1a1410', fontsize=9,
                fontfamily='monospace')

    ax.set_xlabel('Importance', color='#7a6f62', fontsize=9)
    ax.tick_params(colors='#4a4035', labelsize=9)
    ax.spines[['top', 'right', 'bottom']].set_visible(False)
    ax.spines['left'].set_edgecolor('#e0d8cc')
    ax.set_xlim(0, max(importances) * 1.3)
    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Disclaimer ──
    st.markdown("""
    <div style="background:#f5f0e8; border:1px solid #e0d8cc; border-radius:10px;
                padding:0.8rem 1.1rem; margin-top:1rem; font-size:0.78rem; color:#7a6f62;">
      ⚕️ <strong style="color:#1a1410;">Medical Disclaimer:</strong>
      This tool is for educational and research purposes only and does not constitute
      medical advice. Always consult a qualified healthcare provider for diagnosis.
    </div>
    """, unsafe_allow_html=True)


# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; margin-top:2.5rem; padding-top:1rem;
            border-top:1px solid #e0d8cc; font-family:'JetBrains Mono',monospace;
            font-size:0.65rem; color:#c0b8ac; letter-spacing:0.08em;">
  DIABETESSCAN · TRAINED LIVE ON STREAMLIT CLOUD
</div>
""", unsafe_allow_html=True)
