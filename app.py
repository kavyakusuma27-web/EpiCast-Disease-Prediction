import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from datetime import datetime

st.set_page_config(page_title="EpiCast | Disease Intelligence", page_icon="🧬", layout="wide")

# ---------- Theme ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {font-family:'DM Sans',sans-serif;}
.stApp {background: radial-gradient(circle at 15% 0%, #172b45 0, #0b1423 36%, #08111e 100%); color:#edf4ff;}
[data-testid="stSidebar"] {background:linear-gradient(180deg,#101f33,#0b1524); border-right:1px solid #263b54;}
h1,h2,h3 {font-family:'Manrope',sans-serif; color:#f3f7ff;}
.hero {padding:28px 30px; border:1px solid #2b4561; border-radius:22px; background:linear-gradient(110deg,rgba(27,55,82,.94),rgba(15,31,50,.88)); margin-bottom:20px;}
.eyebrow {color:#69dfc0; letter-spacing:2px; font-size:11px; font-weight:800;}
.hero-title {font-size:38px; font-weight:800; line-height:1.15; margin:8px 0;}
.hero-sub {color:#b4c5d9; font-size:14px; max-width:850px;}
.kpi {background:linear-gradient(145deg,#142940,#102033); border:1px solid #29445f; border-radius:16px; padding:17px 19px; min-height:112px;}
.kpi-label {color:#9eb3ca; font-size:12px; text-transform:uppercase; letter-spacing:1px;}
.kpi-value {font-family:'Manrope'; color:#f5f9ff; font-size:27px; font-weight:800; margin-top:7px;}
.kpi-note {color:#68d9bd; font-size:11px; margin-top:2px;}
.panel {background:rgba(18,35,55,.78); border:1px solid #263f59; border-radius:18px; padding:18px 20px; margin:8px 0 16px;}
.small-muted {color:#9fb2c8; font-size:12px;}
div[data-testid="stMetric"] {background:#142940; border:1px solid #29445f; padding:14px; border-radius:14px;}
.stButton>button {background:linear-gradient(90deg,#19b99b,#3c9bd8); color:#071522; font-weight:800; border:0; border-radius:10px;}
div[data-testid="stAlert"] {border-radius:12px;}
</style>
""", unsafe_allow_html=True)

# ---------- Demo data ----------
@st.cache_data
def make_demo():
    rng = np.random.default_rng(21)
    n = 240
    dates = pd.date_range("2025-01-01", periods=n, freq="D")
    wave = 900 + 450*np.sin(np.arange(n)/17) + 0.9*np.arange(n)
    noise = rng.normal(0, 95, n)
    cases = np.maximum(30, wave + noise)
    mobility = np.clip(58 + 9*np.sin(np.arange(n)/23) + rng.normal(0,4,n), 20, 95)
    vaccination = np.clip(35 + np.arange(n)*0.18 + rng.normal(0,1.5,n), 0, 95)
    return pd.DataFrame({"date":dates,"new_cases":cases.round().astype(int),
                         "mobility_index":mobility.round(1),
                         "vaccination_rate":vaccination.round(1)})

def prepare_data(raw):
    d = raw.copy()
    d.columns = [str(c).strip().lower() for c in d.columns]
    if "date" not in d or "new_cases" not in d:
        raise ValueError("Your CSV must contain at least these columns: date and new_cases.")
    d["date"] = pd.to_datetime(d["date"], errors="coerce")
    d["new_cases"] = pd.to_numeric(d["new_cases"], errors="coerce")
    d = d.dropna(subset=["date","new_cases"]).sort_values("date")
    d["new_cases"] = d["new_cases"].clip(lower=0)
    d = d.drop_duplicates("date").reset_index(drop=True)
    if len(d) < 15:
        raise ValueError("Please provide at least 15 valid daily observations.")
    for col in ["mobility_index","vaccination_rate"]:
        if col in d:
            d[col] = pd.to_numeric(d[col], errors="coerce").interpolate().ffill().bfill()
    return d

def classify_states(series):
    # Quantile categories are calculated from the supplied series.
    q1, q2 = series.quantile([1/3, 2/3]).values
    if np.isclose(q1, q2):
        q1, q2 = series.min() + 1e-9, series.max() - 1e-9
    labels = np.where(series <= q1, "Low", np.where(series <= q2, "Medium", "High"))
    return pd.Series(labels, index=series.index), float(q1), float(q2)

def markov_matrix(states):
    names = ["Low","Medium","High"]
    counts = pd.crosstab(states.iloc[:-1], states.iloc[1:]).reindex(index=names, columns=names, fill_value=0)
    probs = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    return counts, probs

def build_svm(d):
    x = pd.DataFrame(index=d.index)
    x["cases_today"] = d["new_cases"]
    x["cases_yesterday"] = d["new_cases"].shift(1)
    x["change_1d"] = d["new_cases"].pct_change().replace([np.inf,-np.inf],np.nan)
    x["rolling_mean_7d"] = d["new_cases"].rolling(7).mean()
    x["rolling_std_7d"] = d["new_cases"].rolling(7).std()
    if "mobility_index" in d: x["mobility_index"] = d["mobility_index"]
    if "vaccination_rate" in d: x["vaccination_rate"] = d["vaccination_rate"]
    states, _, _ = classify_states(d["new_cases"])
    y = states.shift(-1)
    data = x.replace([np.inf,-np.inf],np.nan).join(y.rename("target")).dropna()
    if len(data) < 20 or data["target"].nunique() < 2:
        return None, None, None, None, None, "Not enough variation or rows to train the SVM. Add more observations."
    X, Y = data.drop(columns="target"), data["target"]
    split = max(1, int(len(data)*0.8))
    if split >= len(data): split = len(data)-1
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = Y.iloc[:split], Y.iloc[split:]
    if y_train.nunique() < 2 or len(X_test) == 0:
        return None, None, None, None, None, "The chronological training split needs at least two target classes."
    model = Pipeline([("scale",StandardScaler()),("svm",SVC(kernel="rbf", C=1.0, gamma="scale", class_weight="balanced"))])
    model.fit(X_train,y_train)
    pred = model.predict(X_test)
    metrics = {
        "Accuracy":accuracy_score(y_test,pred),
        "Precision (macro)":precision_score(y_test,pred,average="macro",zero_division=0),
        "Recall (macro)":recall_score(y_test,pred,average="macro",zero_division=0),
        "F1 (macro)":f1_score(y_test,pred,average="macro",zero_division=0)
    }
    return model, X.columns.tolist(), (X_test,y_test,pred), metrics, list(model.named_steps["svm"].classes_), None

# ---------- Sidebar ----------
st.sidebar.markdown("## 🧬 EpiCast")
st.sidebar.caption("Disease spread analytics studio")
source = st.sidebar.radio("Data source", ["Built-in demonstration data", "Upload my CSV"])
uploaded = None
if source == "Upload my CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    st.sidebar.caption("Required columns: date, new_cases. Optional: mobility_index, vaccination_rate.")
if st.sidebar.button("↻ Reset to demo data"):
    st.session_state.pop("raw_data", None)
    st.rerun()

if source == "Upload my CSV" and uploaded is not None:
    try:
        raw = pd.read_csv(uploaded)
        st.session_state["raw_data"] = raw
    except Exception as e:
        st.sidebar.error(f"Could not read CSV: {e}")
if source == "Built-in demonstration data":
    raw = make_demo()
else:
    raw = st.session_state.get("raw_data")
    if raw is None:
        st.markdown('<div class="hero"><div class="eyebrow">WELCOME TO EPICAST</div><div class="hero-title">Turn disease data into<br>meaningful signals.</div><div class="hero-sub">Upload a CSV from the sidebar to explore disease trends, transition probabilities and SVM-based spread classification.</div></div>', unsafe_allow_html=True)
        st.info("Waiting for your dataset. The CSV needs date and new_cases columns.")
        st.stop()

try:
    df = prepare_data(raw)
except Exception as e:
    st.error(str(e))
    st.stop()

states, low_cut, high_cut = classify_states(df["new_cases"])
df["spread_state"] = states
counts, trans = markov_matrix(states)
model, features, test_pack, metrics, classes, svm_error = build_svm(df)

# ---------- Header ----------
st.markdown(f"""
<div class="hero">
 <div class="eyebrow">PUBLIC HEALTH • STOCHASTIC MODELLING • MACHINE LEARNING</div>
 <div class="hero-title">EpiCast <span style="color:#62dfc1">/</span> Disease Spread Intelligence</div>
 <div class="hero-sub">A data-driven learning dashboard combining probability, Markov chains and Support Vector Machines to explore disease-spread patterns. Dataset window: {df.date.min().strftime('%d %b %Y')} — {df.date.max().strftime('%d %b %Y')}.</div>
</div>""", unsafe_allow_html=True)

# KPIs
c1,c2,c3,c4 = st.columns(4)
latest = int(df.new_cases.iloc[-1])
avg7 = df.new_cases.tail(7).mean()
change = ((df.new_cases.iloc[-1] / max(df.new_cases.iloc[-8],1))-1)*100 if len(df)>7 else 0
for col,label,value,note in [
    (c1,"Latest daily cases",f"{latest:,}","Most recent observation"),
    (c2,"7-day average",f"{avg7:,.0f}","Smoothed recent level"),
    (c3,"7-day change",f"{change:+.1f}%","Compared with 7 days earlier"),
    (c4,"Current spread state",states.iloc[-1],"Quantile-based category")]:
    with col:
        st.markdown(f'<div class="kpi"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div><div class="kpi-note">{note}</div></div>',unsafe_allow_html=True)

tab1,tab2,tab3,tab4,tab5 = st.tabs(["📊 Overview","🔁 Markov Chain","🤖 SVM Classifier","🎲 Probability Lab","📁 Dataset & Method"])

with tab1:
    left,right = st.columns([1.65,1])
    with left:
        st.markdown("### Disease activity timeline")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.date,y=df.new_cases,mode="lines",name="Daily cases",line=dict(width=2,color="#54d6bd")))
        fig.add_trace(go.Scatter(x=df.date,y=df.new_cases.rolling(7).mean(),mode="lines",name="7-day moving average",line=dict(width=3,color="#f5b85b")))
        fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(12,27,43,.7)",height=370,margin=dict(l=10,r=10,t=20,b=10),legend=dict(orientation="h",y=1.12),xaxis_title="",yaxis_title="New cases")
        st.plotly_chart(fig,use_container_width=True)
    with right:
        st.markdown("### Spread-state distribution")
        dist = states.value_counts().reindex(["Low","Medium","High"],fill_value=0).rename_axis("State").reset_index(name="Days")
        fig2=px.pie(dist,names="State",values="Days",hole=.62,color="State",color_discrete_map={"Low":"#43c6a7","Medium":"#e9b75e","High":"#e36f79"})
        fig2.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",height=370,margin=dict(l=5,r=5,t=15,b=5),legend=dict(orientation="h",y=-.1))
        st.plotly_chart(fig2,use_container_width=True)
    st.markdown("### Recent observations")
    show = df[["date","new_cases","spread_state"]].tail(12).copy()
    show["date"] = show.date.dt.strftime("%d %b %Y")
    st.dataframe(show.rename(columns={"date":"Date","new_cases":"New cases","spread_state":"Spread state"}),use_container_width=True,hide_index=True)

with tab2:
    st.markdown("### Markov transition model")
    st.write("A Markov chain estimates the probability of moving from one spread state today to another state in the next observation.")
    st.caption(f"State boundaries derived from this dataset: Low ≤ {low_cut:,.1f}; Medium ≤ {high_cut:,.1f}; High > {high_cut:,.1f} cases.")
    a,b = st.columns([1.2,1])
    with a:
        fig=px.imshow(trans, text_auto=".2f", color_continuous_scale="Teal", aspect="auto", zmin=0,zmax=1)
        fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",height=340,coloraxis_colorbar_title="Probability")
        fig.update_xaxes(title="Next state"); fig.update_yaxes(title="Current state")
        st.plotly_chart(fig,use_container_width=True)
    with b:
        st.markdown("#### Transition probability matrix")
        st.dataframe(trans.style.format("{:.3f}"),use_container_width=True)
        st.markdown("#### Transition counts")
        st.dataframe(counts,use_container_width=True)
    st.markdown("#### One-step forecast")
    current = states.iloc[-1]
    row = trans.loc[current]
    st.info(f"Current state: **{current}**. Based on historical transitions, the most likely next state is **{row.idxmax()}** ({row.max():.1%}).")
    dist0 = np.array([1 if current==s else 0 for s in ["Low","Medium","High"]],dtype=float)
    future = dist0 @ trans.values
    st.write("Starting from the current state, the one-step state probabilities are:")
    fcols=st.columns(3)
    for col,s,v in zip(fcols,["Low","Medium","High"],future):
        col.metric(f"{s} probability",f"{v:.1%}")

with tab3:
    st.markdown("### Support Vector Machine")
    st.write("The SVM uses recent case history and optional mobility/vaccination indicators to classify the next observation's spread state.")
    if svm_error:
        st.warning(svm_error)
    else:
        mcols=st.columns(4)
        for col,(name,val) in zip(mcols,metrics.items()):
            col.metric(name,f"{val:.1%}")
        X_test,y_test,pred = test_pack
        cm=confusion_matrix(y_test,pred,labels=["Low","Medium","High"])
        c1,c2=st.columns([1,1])
        with c1:
            fig=px.imshow(cm,text_auto=True,x=["Pred Low","Pred Medium","Pred High"],y=["Actual Low","Actual Medium","Actual High"],color_continuous_scale="Teal",aspect="auto")
            fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",height=340)
            st.plotly_chart(fig,use_container_width=True)
        with c2:
            st.markdown("#### Model configuration")
            st.write("- Algorithm: SVM with RBF kernel")
            st.write("- Scaling: StandardScaler")
            st.write("- Split: chronological 80% train / 20% test")
            st.write("- Class handling: balanced weights")
            st.write("**Features used:**")
            st.code("\n".join(features))
        result=pd.DataFrame({"Actual":y_test.values,"Predicted":pred},index=y_test.index)
        st.markdown("#### Held-out test predictions")
        st.dataframe(result.tail(15),use_container_width=True)
        st.caption("Metrics are calculated on the held-out tail of the supplied time series. They are not a guarantee of future performance.")

with tab4:
    st.markdown("### Probability laboratory")
    st.write("Explore empirical probabilities calculated from the spread-state observations.")
    chosen=st.selectbox("Select a spread state",["Low","Medium","High"],index=["Low","Medium","High"].index(current if 'current' in globals() else states.iloc[-1]))
    nstate=int((states==chosen).sum())
    total=len(states)
    st.latex(r"P(\\text{State})=\\frac{\\text{Number of observations in state}}{\\text{Total observations}}")
    p=nstate/total
    q1,q2=st.columns(2)
    q1.metric(f"P({chosen})",f"{p:.3f}",f"{p:.1%} of observations")
    q2.metric("Observations",f"{nstate} / {total}","Empirical frequency")
    st.progress(float(p))
    st.markdown("#### Conditional probability from the Markov matrix")
    origin=st.selectbox("Given that today's state is",["Low","Medium","High"],key="origin")
    target=st.selectbox("What is the probability that the next state is",["Low","Medium","High"],key="target")
    st.success(f"P(Next = {target} | Current = {origin}) = **{trans.loc[origin,target]:.3f} ({trans.loc[origin,target]:.1%})**")

with tab5:
    st.markdown("### Data quality and modelling notes")
    d1,d2,d3=st.columns(3)
    d1.metric("Valid observations",len(df))
    d2.metric("Date coverage",(df.date.max()-df.date.min()).days+1,"calendar days")
    d3.metric("Available features",len([x for x in ["mobility_index","vaccination_rate"] if x in df]))
    st.markdown("#### Data preview")
    st.dataframe(df.tail(20),use_container_width=True,hide_index=True)
    st.markdown("#### Methodology")
    st.markdown("""
    1. **Pre-processing:** parse dates, remove invalid rows, sort observations and handle missing optional indicators.
    2. **State construction:** divide daily case counts into Low, Medium and High groups using one-third and two-third quantiles.
    3. **Probability:** calculate empirical state frequencies and conditional transition probabilities.
    4. **Markov chain:** estimate a 3 × 3 transition matrix from consecutive states.
    5. **SVM:** create lagged case features, rolling statistics and optional mobility/vaccination features; scale inputs and train an RBF-kernel SVM.
    6. **Evaluation:** use a chronological 80/20 split and report accuracy, macro precision, macro recall and macro F1.
    """)
    st.markdown("#### Important interpretation")
    st.warning("This is an educational decision-support prototype. Built-in data is synthetic demonstration data, not real surveillance data. For research conclusions, upload a verified dataset, document its source, and validate the model carefully. Associations do not prove causation.")

st.markdown("---")
st.markdown('<div style="text-align:center;color:#8198b1;font-size:12px;padding:8px">EPICAST • Academic prototype for Mathematics for Computing-3 • Markov chains + probability + SVM</div>',unsafe_allow_html=True)
