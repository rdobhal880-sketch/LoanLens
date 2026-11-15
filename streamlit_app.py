import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import streamlit as st

st.set_page_config(page_title="LoanLens", page_icon="🔍", layout="wide")

# ---------- Look and feel ----------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bricolage+Grotesque:opsz,wght@12..96,500;12..96,700;12..96,800&family=Public+Sans:wght@400;500;600;700&display=swap');
:root { --ink:#14232B; --slate:#5A6A70; --rule:#D5DBD7; --paper:#F4F5F1; --card:#FFFFFF;
        --go:#1E5B43; --stop:#8F2D21; --go-soft:#DCE9E1; --stop-soft:#F1DDD8; }
html, body, .stApp, .stMarkdown, label, button, input, select, textarea, p, li, td {
  font-family: 'Public Sans', Helvetica, Arial, sans-serif !important; font-variant-numeric: tabular-nums; }
.stApp { background: var(--paper); color: var(--ink); }
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }
.block-container { padding-top: 2.4rem; padding-bottom: 3rem; max-width: 1160px; }
h1, h2, h3, .display { font-family: 'Bricolage Grotesque', 'Public Sans', sans-serif !important; color: var(--ink); }
section[data-testid="stSidebar"] { background: var(--card); border-right: 1px solid var(--rule); }
section[data-testid="stSidebar"] h2 { font-size: 1.35rem; font-weight: 800; margin-bottom: 0; }
section[data-testid="stSidebar"] h3 { font-size: 0.92rem; font-weight: 700; color: var(--slate);
  border-top: 1px solid var(--rule); padding-top: 0.9rem; margin: 1rem 0 0.1rem; }
.brand { font-family: 'Bricolage Grotesque', 'Public Sans', sans-serif !important; font-weight: 800; font-size: 3rem; line-height: 1;
  letter-spacing: -0.03em; color: var(--ink); margin: 0 0 0.7rem; }
.lede { color: var(--slate); font-size: 1.06rem; line-height: 1.55; max-width: 64ch; margin: 0.5rem 0 1.2rem; }
.hint { color: var(--slate); font-size: 0.95rem; padding-top: 0.45rem; }
.stButton button { border: 1px solid var(--ink); background: transparent; color: var(--ink); font-weight: 600;
  border-radius: 999px; }
.stButton button:hover { background: var(--ink); color: #fff; border-color: var(--ink); }
.hero { background: var(--bg); color: #fff; border-radius: 14px; padding: 1.9rem 2.1rem; margin: 0.9rem 0 1.6rem;
  display: flex; flex-wrap: wrap; gap: 1.5rem 3rem; align-items: flex-end; justify-content: space-between; }
.hero .word { font-family: 'Bricolage Grotesque', sans-serif; font-weight: 800; font-size: 5.2rem; line-height: 0.9;
  letter-spacing: -0.04em; }
.hero .say { font-size: 1.1rem; line-height: 1.5; max-width: 46ch; margin-top: 0.9rem; opacity: 0.95; }
.hero .figs { display: flex; gap: 2.2rem; }
.hero .fig b { display: block; font-family: 'Bricolage Grotesque', sans-serif; font-size: 2.1rem; font-weight: 700;
  line-height: 1.1; }
.hero .fig span { font-size: 0.9rem; opacity: 0.85; }
.sec { font-family: 'Bricolage Grotesque', 'Public Sans', sans-serif !important; font-weight: 700; font-size: 1.45rem !important; letter-spacing: -0.01em; margin: 0 0 0.2rem; color: var(--ink); }
.sub { color: var(--slate); font-size: 0.95rem; margin: 0 0 0.9rem; max-width: 70ch; }
.track { position: relative; height: 96px; border-radius: 10px; margin-top: 2.3rem;
  background: linear-gradient(to right, var(--go-soft) 0, var(--go-soft) var(--be), var(--stop-soft) var(--be), var(--stop-soft) 100%); }
.track .zone { position: absolute; bottom: 6px; font-size: 0.82rem; font-weight: 600; }
.track .be { position: absolute; top: -8px; bottom: -8px; left: var(--be); width: 2px; background: var(--ink); }
.track .be span { position: absolute; top: -1.5rem; left: 50%; transform: translateX(-50%); white-space: nowrap;
  font-size: 0.85rem; font-weight: 700; }
.track .path { position: absolute; top: 28px; height: 3px; background: var(--ink); opacity: 0.25; }
.dot { position: absolute; top: 29px; width: 14px; height: 14px; border-radius: 50%; background: #fff;
  border: 3px solid var(--ink); transform: translate(-50%, -50%); }
.dot.on { z-index: 3; width: 26px; height: 26px; background: var(--ink); border: 4px solid #fff; box-shadow: 0 0 0 2px var(--ink); }
.dot i { display: none; }
.dot.on i { display: block; position: absolute; top: 30px; left: 50%; transform: translateX(-50%); font-style: normal; white-space: nowrap; font-size: 0.88rem; font-weight: 700; color: #fff; background: var(--ink);
  padding: 0.1rem 0.55rem; border-radius: 999px; }
.ticks { position: relative; height: 1.4rem; margin: 0.35rem 0 1.8rem; font-size: 0.8rem; color: var(--slate); }
.ticks span { position: absolute; transform: translateX(-50%); }
.drv { display: grid; grid-template-columns: 11.5rem 1fr; align-items: center; gap: 0.8rem; padding: 0.32rem 0;
  font-size: 0.95rem; }
.drv .bar { position: relative; height: 18px; }
.drv .bar::before { content: ''; position: absolute; left: 50%; top: -5px; bottom: -5px; width: 1px; background: var(--ink); }
.drv .bar b { position: absolute; top: 0; bottom: 0; border-radius: 3px; }
.drvhead { display: grid; grid-template-columns: 11.5rem 1fr; gap: 0.8rem; font-size: 0.82rem; font-weight: 600;
  margin-bottom: 0.2rem; }
.drvhead div:last-child { display: flex; justify-content: space-between; }
.maths { width: 100%; border-collapse: collapse; font-size: 0.97rem; border: none !important; }
.maths tr { border: none !important; }
.maths td { padding: 0.6rem 0; border: none !important; border-bottom: 1px solid var(--rule) !important;
  background: transparent !important; color: var(--ink); }
.maths td:last-child { text-align: right; font-weight: 700; white-space: nowrap; padding-left: 1rem; }
.maths tr.total td { border-bottom: none !important; font-weight: 800; font-size: 1.05rem; }
.proof { border-top: 2px solid var(--ink); margin-top: 2.4rem; padding-top: 1.2rem; }
.proof ul { margin: 0; padding-left: 1.1rem; columns: 2; column-gap: 3rem; color: var(--ink); font-size: 0.95rem;
  line-height: 1.55; }
.proof li { break-inside: avoid; margin-bottom: 0.6rem; }
.foot { color: var(--slate); font-size: 0.85rem; margin-top: 2rem; }

/* hide 'Press Enter to apply' hints that overlap the inputs (values apply on their own) */
[data-testid="InputInstructions"] { display: none !important; }
/* sidebar width and spacing */
@media (min-width: 900px) { section[data-testid="stSidebar"] { min-width: 320px !important; } }
section[data-testid="stSidebar"] .block-container, section[data-testid="stSidebar"] [data-testid="stSidebarContent"] { padding-top: 1.2rem; }
/* economy selector: clear selected state */
button[data-variant="segmented_control"] { border-color: var(--rule) !important; background: #fff !important; color: var(--ink) !important; }
button[data-variant="segmented_control"] p { color: var(--ink) !important; font-weight: 500; }
button[data-variant="segmented_control"][aria-checked="true"] { background: var(--ink) !important; border-color: var(--ink) !important; }
button[data-variant="segmented_control"][aria-checked="true"] p { color: #fff !important; font-weight: 600; }
/* example buttons: no wrapping */
.stButton button p { white-space: nowrap; }
/* hide the developer toolbar and its Deploy button */
[data-testid="stToolbarActions"], [data-testid="stAppDeployButton"], .stAppDeployButton, [data-testid="stMainMenu"], [data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
[data-testid="stExpandSidebarButton"] { color: var(--ink) !important; }
/* inputs: calm borders and focus ring */
section[data-testid="stSidebar"] [data-baseweb="input"], section[data-testid="stSidebar"] [data-baseweb="select"] > div {
  border-radius: 8px !important; }
*:focus-visible { outline: 2px solid var(--ink) !important; outline-offset: 2px; }
@media (max-width: 760px) { .hero { padding: 1.4rem 1.3rem; } .hero .word { font-size: 3.4rem; }
  .hero .figs { gap: 1.2rem; flex-wrap: wrap; } .hero .fig b { font-size: 1.6rem; } .brand { font-size: 2.4rem; } .proof ul { columns: 1; }
  .drv, .drvhead { grid-template-columns: 8rem 1fr; } }
</style>
""", unsafe_allow_html=True)

# ---------- Model ----------
@st.cache_resource
def load():
    return lgb.Booster(model_file="model.txt"), json.load(open("meta.json"))

booster, META = load()
FEATURES, FEATURES_CAT, CATS = META["features"], META["features_cat"], META["categories"]
SCENARIOS, MARGIN, LOSS_RATE = META["scenarios"], META["margin"], META["loss_rate"]
NAMES = list(SCENARIOS)
SHORT = dict(zip(NAMES, ["Normal economy", "Mild downturn", "Severe recession", "Extreme recession"]))
IN_A = dict(zip(NAMES, ["In a normal economy", "In a mild downturn", "In a severe recession", "In an extreme recession"]))

SECTORS = {"11": "Agriculture", "21": "Mining & oil", "22": "Utilities", "23": "Construction",
           "31": "Manufacturing (food, textiles)", "32": "Manufacturing (wood, chemicals, plastics)",
           "33": "Manufacturing (metal, machinery, electronics)", "42": "Wholesale trade",
           "44": "Retail (stores)", "45": "Retail (general & online)", "48": "Transport",
           "49": "Couriers & warehousing", "51": "Information & media", "52": "Finance & insurance",
           "53": "Real estate", "54": "Professional services", "55": "Company management",
           "56": "Admin & support services", "61": "Education", "62": "Health care",
           "71": "Arts & recreation", "72": "Restaurants & hotels",
           "81": "Other services (repair, salons, laundries)", "92": "Public administration",
           "Unknown": "Other or not listed"}
PLAIN = {"NoEmp": "Number of employees", "CreateJob": "Jobs the loan creates", "RetainedJob": "Jobs the loan keeps",
         "log_loan": "Loan size", "sba_share": "SBA guarantee", "franchise": "Franchise",
         "new_business": "Business age", "bank_out_of_state": "Bank in another state",
         "State": "Business location", "BankState": "Bank's location", "naics_sector": "Industry",
         "RevLineCr": "Revolving credit line", "LowDoc": "Fast-track paperwork"}

# ---------- Example applications ----------
DEFAULTS = dict(state="CA", bank_state="CA", sector="72", amount=150000, sba=75, emp=8, created=2, kept=8,
                age="Existing business", franchise=False, revline=False, lowdoc=False, scenario=NAMES[0])
EXAMPLES = {
    "Established clinic": dict(state="TX", bank_state="TX", sector="62", amount=300000, sba=75, emp=20,
                               created=3, kept=20),
    "New restaurant": dict(state="FL", bank_state="NC", sector="72", amount=50000, sba=50, emp=2,
                           created=2, kept=0, age="New business (under 2 years)", revline=True),
    "Small repair shop": dict(state="MI", bank_state="MI", sector="81", amount=25000, sba=50, emp=4,
                              created=0, kept=4, revline=True),
}
VALID = {"state": CATS["State"], "bank_state": CATS["BankState"], "sector": CATS["naics_sector"]}
for k, v in DEFAULTS.items():
    if k in VALID and v not in VALID[k]:
        v = VALID[k][0]
    st.session_state.setdefault(k, v)

def use_example(name):
    for k, v in {**DEFAULTS, **EXAMPLES[name]}.items():
        if k == "scenario" or (k in VALID and v not in VALID[k]):
            continue
        st.session_state[k] = v

# ---------- Sidebar: the application ----------
with st.sidebar:
    st.markdown("## Loan application")
    st.caption("After typing a number, press Enter or click outside the box to update the result.")
    st.markdown("### The business")
    st.selectbox("State", CATS["State"], key="state")
    order = sorted(CATS["naics_sector"], key=lambda c: (c == "Unknown", SECTORS.get(c, c)))
    st.selectbox("Industry", order, key="sector", format_func=lambda c: SECTORS.get(c, c),
                 help="The standard US industry sectors (NAICS). Pick 'Other or not listed' if none fits.")
    st.radio("Age", ["Existing business", "New business (under 2 years)"], key="age")
    st.number_input("Employees", min_value=0, max_value=10_000, step=1, format="%d", key="emp")
    st.checkbox("Franchise", key="franchise")
    st.markdown("### The loan")
    st.number_input("Amount ($)", min_value=1000, max_value=5_000_000, step=5000, key="amount", format="%d")
    st.slider("Share guaranteed by the SBA (%)", 0, 100, step=5, key="sba",
              help="The SBA is a US government agency that repays the lender part of a failed loan.")
    st.number_input("Jobs it will create", min_value=0, max_value=10_000, step=1, format="%d", key="created")
    st.number_input("Jobs it will keep", min_value=0, max_value=10_000, step=1, format="%d", key="kept")
    st.checkbox("Revolving line of credit", key="revline")
    st.checkbox("Fast-track (LowDoc) paperwork", key="lowdoc")
    st.markdown("### The lender")
    st.selectbox("Bank's state", CATS["BankState"], key="bank_state")

S = st.session_state

# ---------- Score the application ----------
row = pd.DataFrame([{
    "NoEmp": S.emp, "CreateJob": S.created, "RetainedJob": S.kept, "log_loan": np.log1p(S.amount),
    "sba_share": S.sba / 100, "franchise": int(S.franchise),
    "new_business": int(S.age.startswith("New")), "bank_out_of_state": int(S.state != S.bank_state),
    "State": S.state, "BankState": S.bank_state, "naics_sector": S.sector,
    "RevLineCr": "Y" if S.revline else "N", "LowDoc": "Y" if S.lowdoc else "N"}])
for c in FEATURES_CAT:
    row[c] = pd.Categorical(row[c], categories=CATS[c])
contrib = booster.predict(row[FEATURES], pred_contrib=True)[0]     # SHAP values: each factor's push on the score
base = contrib.sum()
risk = {n: 1 / (1 + np.exp(-(base + SCENARIOS[n]))) for n in NAMES}

gain = MARGIN * S.amount                                     # interest the lender earns if the loan is repaid
loss_if_default = LOSS_RATE * S.amount * (1 - S.sba / 100)   # lender's share of a typical loss
breakeven = gain / (gain + loss_if_default)                  # above this risk the loan loses money on average

# ---------- Header ----------
st.markdown("<div class='brand'>LoanLens</div><p class='lede'>Would this small-business loan still make sense if a "
            "recession hit? LoanLens scores the risk, explains it, and stress-tests the lender's decision. "
            "Fill in the loan application, or start from an example.</p>", unsafe_allow_html=True)
cols = st.columns([1, 1, 1, 2.6])
for col, name in zip(cols, EXAMPLES):
    col.button(name, on_click=use_example, args=(name,), width="stretch")

st.markdown("<div class='sec' style='margin-top:1.4rem'>Choose the economy to test against</div>", unsafe_allow_html=True)
if hasattr(st, "segmented_control"):
    st.segmented_control("Economy", NAMES, key="scenario", format_func=lambda n: SHORT[n],
                         label_visibility="collapsed")
else:
    st.radio("Economy", NAMES, key="scenario", horizontal=True, format_func=lambda n: SHORT[n],
             label_visibility="collapsed")
scenario = S.scenario or NAMES[0]

# ---------- The decision ----------
p = risk[scenario]
ev = (1 - p) * gain - p * loss_if_default
approve = ev > 0
bg = "var(--go)" if approve else "var(--stop)"
money = f"{'+' if ev >= 0 else '−'}${abs(ev):,.0f}"
say = (f"{IN_A[scenario]}, this loan has a {p:.0%} chance of default. The lender breaks even at "
       f"{breakeven:.0%}, so on average it {'gains' if approve else 'loses'} ${abs(ev):,.0f}.")
st.markdown(
    f"<div class='hero' style='--bg:{bg}'><div><div class='word'>{'Approve' if approve else 'Decline'}</div>"
    f"<div class='say'>{say}</div></div><div class='figs'>"
    f"<div class='fig'><b>{p:.0%}</b><span>chance of default</span></div>"
    f"<div class='fig'><b>{breakeven:.0%}</b><span>lender breaks even</span></div>"
    f"<div class='fig'><b>{money}</b><span>expected result</span></div></div></div>", unsafe_allow_html=True)

# ---------- The stress line ----------
top_pct = max(50, max(risk.values()) * 100 + 8, breakeven * 100 + 12)
hi = min(100, int(np.ceil(top_pct / 10) * 10))
pos = lambda v: f"{min(v * 100, hi) / hi * 100:.2f}%"
dots = "".join(
    f"<div class='dot{' on' if n == scenario else ''}' style='left:{pos(risk[n])}' title='{SHORT[n]}: {risk[n]:.0%}'>"
    f"<i>{SHORT[n]} {risk[n]:.0%}</i></div>" for n in NAMES)
lo_x, hi_x = pos(min(risk.values())), pos(max(risk.values()))
ticks = "".join(f"<span style='left:{t / hi * 100:.2f}%'>{t}%</span>" for t in range(0, hi + 1, 10))
st.markdown(
    "<div class='sec'>The same loan in four economies</div>"
    "<p class='sub'>Each dot is this loan's chance of default, from a normal economy on the left to an extreme "
    "recession on the right. Once a dot crosses the break-even line, the lender expects to lose money.</p>"
    f"<div class='track' style='--be:{pos(breakeven)}'>"
    "<span class='zone' style='left:12px;color:var(--go)'>Lender expects a profit</span>"
    "<span class='zone' style='right:12px;color:var(--stop)'>Lender expects a loss</span>"
    f"<div class='path' style='left:{lo_x};right:calc(100% - {hi_x})'></div>{dots}"
    f"<div class='be'><span>Break-even {breakeven:.0%}</span></div></div>"
    f"<div class='ticks'>{ticks}</div>", unsafe_allow_html=True)

# ---------- Why, and the lender's maths ----------
left, right = st.columns([1.25, 1], gap="large")
with left:
    s = pd.Series(contrib[:-1], index=[PLAIN[f] for f in FEATURES])
    top = s.reindex(s.abs().sort_values(ascending=False).index).head(7)
    biggest = max(top.abs().max(), 1e-9)
    rows = ""
    for name, v in top.items():
        w = abs(v) / biggest * 48
        side = f"left:50%;width:{w:.1f}%;background:var(--stop)" if v > 0 else f"right:50%;width:{w:.1f}%;background:var(--go)"
        rows += f"<div class='drv'><div>{name}</div><div class='bar'><b style='{side}'></b></div></div>"
    st.markdown(
        "<div class='sec'>What drives this risk</div>"
        "<p class='sub'>How much each detail of this application pushed its risk down or up.</p>"
        "<div class='drvhead'><div></div><div><span style='color:var(--go)'>Lowers risk</span>"
        f"<span style='color:var(--stop)'>Raises risk</span></div></div>{rows}", unsafe_allow_html=True)
with right:
    colour = "var(--go)" if approve else "var(--stop)"
    st.markdown(
        "<div class='sec'>The lender's maths</div>"
        "<p class='sub'>Approve only if the expected interest beats the expected loss.</p>"
        "<table class='maths'>"
        f"<tr><td>Interest earned if repaid ({MARGIN:.0%} of the loan)</td><td>${gain:,.0f}</td></tr>"
        f"<tr><td>Loss if it defaults, after the SBA covers {S.sba}%</td><td>${loss_if_default:,.0f}</td></tr>"
        f"<tr><td>Chance of default</td><td>{p:.0%}</td></tr>"
        f"<tr class='total'><td>Expected result</td><td style='color:{colour}'>{money}</td></tr></table>"
        "<p class='sub' style='margin-top:0.9rem'>A bigger SBA guarantee means the lender carries less of the "
        "loss. Move the guarantee slider to see the decision change.</p>", unsafe_allow_html=True)

# ---------- Evidence ----------
st.markdown(
    "<div class='proof'><div class='sec'>How LoanLens was tested</div><ul>"
    "<li>Trained on 422,033 US small-business loans approved in 1990–2002, then tested on 224,363 loans approved "
    "in 2005–07, which ran into the 2008 recession.</li>"
    "<li>It ranks risky loans better than an industry rule of thumb (PR-AUC 0.457 against 0.352), using only "
    "what a lender knows on the day of approval.</li>"
    "<li>Trained in good times, it expected 15% of those loans to default. 34% did. That gap is why the stress "
    "test exists.</li>"
    "<li>On the 2005–07 loans, the severe-recession setting would have earned lenders an estimated $327M more "
    "than trusting the model as trained.</li>"
    "<li>The cost: under that setting, 68% of good borrowers asking for under $50,000 are refused, against 21% "
    "for loans over $150,000.</li>"
    "<li>Limits: US loans from 1990–2010 only, and only loans that were approved. The 8% margin is an "
    "assumption.</li></ul></div>"
    "<p class='foot'>This project is a prototype and is not financial advice. Built by Ram Dobhal for the Veritas AI "
    "International High School Hackathon 2026. Data: US Small Business Administration loans (Li, Mickel &amp; "
    "Taylor, 2018).</p>", unsafe_allow_html=True)
