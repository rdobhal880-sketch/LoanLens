import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st

st.set_page_config(page_title="LoanLens", page_icon="🔍", layout="wide")

@st.cache_resource
def load():
    return lgb.Booster(model_file="model.txt"), json.load(open("meta.json"))

booster, META = load()
FEATURES, FEATURES_CAT, CATS = META["features"], META["features_cat"], META["categories"]
SCENARIOS, MARGIN, LOSS_RATE = META["scenarios"], META["margin"], META["loss_rate"]

SECTORS = {"11": "Agriculture", "21": "Mining & oil", "22": "Utilities", "23": "Construction",
           "31": "Manufacturing", "32": "Manufacturing", "33": "Manufacturing", "42": "Wholesale trade",
           "44": "Retail", "45": "Retail", "48": "Transport & warehousing", "49": "Transport & warehousing",
           "51": "Information & media", "52": "Finance & insurance", "53": "Real estate",
           "54": "Professional services", "55": "Company management", "56": "Admin & support services",
           "61": "Education", "62": "Health care", "71": "Arts & recreation", "72": "Restaurants & hotels",
           "81": "Other services (repair, salons...)", "92": "Public administration", "Unknown": "Unknown"}
PLAIN = {"NoEmp": "Number of employees", "CreateJob": "Jobs the loan will create", "RetainedJob": "Jobs the loan will keep",
         "log_loan": "Loan size", "sba_share": "Share guaranteed by the SBA", "franchise": "Franchise or not",
         "new_business": "New vs established business", "bank_out_of_state": "Bank is in another state",
         "State": "Business location (state)", "BankState": "Bank's state", "naics_sector": "Industry",
         "RevLineCr": "Revolving line of credit", "LowDoc": "Fast-track 'LowDoc' loan"}

def idx(options, preferred):
    return options.index(preferred) if preferred in options else 0

st.title("🔍 LoanLens: small-business loan risk, stress-tested for a recession")
st.write("Enter a loan application. LoanLens estimates the chance of default, explains why, and shows whether a "
         "lender should approve it in a normal economy or a recession. Trained on 1990-2002 US SBA loans and tested "
         "on 2005-07 loans. *Prototype, not financial advice.*")

left, right = st.columns(2)
with left:
    state = st.selectbox("Business state", CATS["State"], index=idx(CATS["State"], "CA"))
    bank_state = st.selectbox("Bank's state", CATS["BankState"], index=idx(CATS["BankState"], "CA"))
    sector = st.selectbox("Industry", CATS["naics_sector"], index=idx(CATS["naics_sector"], "72"),
                          format_func=lambda c: f"{SECTORS.get(c, c)} ({c})")
    amount = st.number_input("Loan amount ($)", min_value=1000, value=150000, step=5000)
    sba_pct = st.slider("Share guaranteed by the SBA (%)", 0, 100, 75, step=5)
    employees = st.number_input("Number of employees", min_value=0, value=8)
    jobs_created = st.number_input("Jobs the loan will create", min_value=0, value=2)
    jobs_retained = st.number_input("Jobs the loan will keep", min_value=0, value=8)
    new_biz = st.radio("Business age", ["Existing business", "New business (under 2 years)"], horizontal=True)
    franchise = st.checkbox("Franchise")
    revline = st.checkbox("Revolving line of credit")
    lowdoc = st.checkbox("Fast-track 'LowDoc' loan")
    scenario = st.radio("Economic scenario (stress test)", list(SCENARIOS))

row = pd.DataFrame([{
    "NoEmp": employees, "CreateJob": jobs_created, "RetainedJob": jobs_retained,
    "log_loan": np.log1p(amount), "sba_share": sba_pct / 100, "franchise": int(franchise),
    "new_business": int(new_biz == "New business (under 2 years)"),
    "bank_out_of_state": int(state != bank_state),
    "State": state, "BankState": bank_state, "naics_sector": sector,
    "RevLineCr": "Y" if revline else "N", "LowDoc": "Y" if lowdoc else "N"}])
for c in FEATURES_CAT:
    row[c] = pd.Categorical(row[c], categories=CATS[c])
row = row[FEATURES]

contrib = booster.predict(row, pred_contrib=True)[0]          # SHAP values: each factor's push on log-odds
base_logit = contrib.sum()
p_normal = 1 / (1 + np.exp(-base_logit))
p = 1 / (1 + np.exp(-(base_logit + SCENARIOS[scenario])))
gain = MARGIN * amount
loss = LOSS_RATE * amount * (1 - sba_pct / 100)
ev = (1 - p) * gain - p * loss

with right:
    if ev > 0:
        st.success("## ✅ APPROVE")
    else:
        st.error("## ❌ DECLINE")
    st.markdown(f"**Default risk:** {p:.0%} under *{scenario}* (normal-economy estimate: {p_normal:.0%})")
    st.markdown(f"**Lender's expected result:** {'+' if ev >= 0 else '−'}${abs(ev):,.0f} "
                f"(expected interest ${(1 - p) * gain:,.0f} vs expected loss ${p * loss:,.0f}; "
                f"the SBA guarantee covers {sba_pct}% of any loss)")
    s = pd.Series(contrib[:-1], index=[PLAIN[f] for f in FEATURES]).sort_values(key=np.abs, ascending=False)
    up = [k for k, v in s.items() if v > 0][:3]
    down = [k for k, v in s.items() if v < 0][:3]
    st.markdown("**What raises the risk:** " + (", ".join(up) or "nothing major"))
    st.markdown("**What lowers the risk:** " + (", ".join(down) or "nothing major"))
    top = s.head(8)[::-1]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.barh(top.index, top.values, color=["#d62728" if v > 0 else "#2ca02c" for v in top.values])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Why: each factor's push on default risk")
    ax.set_xlabel("← lowers risk      raises risk →")
    plt.tight_layout()
    st.pyplot(fig)
    st.caption("Prototype for learning purposes. Not financial advice. Built by Ram Dobhal for the Veritas AI "
               "International High School Hackathon 2026.")
