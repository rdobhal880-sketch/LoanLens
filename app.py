import json
import numpy as np
import pandas as pd
import lightgbm as lgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

booster = lgb.Booster(model_file="model.txt")
META = json.load(open("meta.json"))
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
sector_choices = [(f"{SECTORS.get(c, c)} ({c})", c) for c in CATS["naics_sector"]]

def make_row(state, bank_state, sector, amount, sba_pct, employees, jobs_created, jobs_retained,
             new_biz, franchise, revline, lowdoc):
    row = pd.DataFrame([{
        "NoEmp": employees, "CreateJob": jobs_created, "RetainedJob": jobs_retained,
        "log_loan": np.log1p(amount), "sba_share": sba_pct / 100, "franchise": int(franchise),
        "new_business": int(new_biz == "New business (under 2 years)"),
        "bank_out_of_state": int(state != bank_state),
        "State": state, "BankState": bank_state, "naics_sector": sector,
        "RevLineCr": "Y" if revline else "N", "LowDoc": "Y" if lowdoc else "N"}])
    for c in FEATURES_CAT:
        row[c] = pd.Categorical(row[c], categories=CATS[c])
    return row[FEATURES]

def assess(state, bank_state, sector, amount, sba_pct, employees, jobs_created, jobs_retained,
           new_biz, franchise, revline, lowdoc, scenario):
    row = make_row(state, bank_state, sector, float(amount), float(sba_pct), float(employees),
                   float(jobs_created), float(jobs_retained), new_biz, franchise, revline, lowdoc)
    contrib = booster.predict(row, pred_contrib=True)[0]
    base_logit = contrib.sum()
    p_normal = 1 / (1 + np.exp(-base_logit))
    p = 1 / (1 + np.exp(-(base_logit + SCENARIOS[scenario])))
    gain = MARGIN * amount
    loss = LOSS_RATE * amount * (1 - sba_pct / 100)
    ev = (1 - p) * gain - p * loss
    verdict = "✅ APPROVE" if ev > 0 else "❌ DECLINE"
    md = (f"## {verdict}\n"
          f"**Default risk:** {p:.0%} under *{scenario}* (normal-economy estimate: {p_normal:.0%})\n\n"
          f"**Lender's expected result:** {'+' if ev >= 0 else '−'}${abs(ev):,.0f} "
          f"(expected interest ${(1 - p) * gain:,.0f} vs expected loss ${p * loss:,.0f}; "
          f"the SBA guarantee covers {sba_pct:.0f}% of any loss)\n\n")
    s = pd.Series(contrib[:-1], index=[PLAIN[f] for f in FEATURES]).sort_values(key=np.abs, ascending=False)
    up = [f"- {k}" for k, v in s.items() if v > 0][:3]
    down = [f"- {k}" for k, v in s.items() if v < 0][:3]
    md += "**What raises the risk:**\n" + ("\n".join(up) or "- nothing major") + "\n\n"
    md += "**What lowers the risk:**\n" + ("\n".join(down) or "- nothing major") + "\n\n"
    md += "_Prototype for learning purposes. Not financial advice._"
    top = s.head(8)[::-1]
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.barh(top.index, top.values, color=["#d62728" if v > 0 else "#2ca02c" for v in top.values])
    ax.axvline(0, color="black", lw=0.8)
    ax.set_title("Why: each factor's push on default risk"); ax.set_xlabel("← lowers risk      raises risk →")
    plt.tight_layout()
    return md, fig

with gr.Blocks(title="LoanLens") as demo:
    gr.Markdown("# 🔍 LoanLens: small-business loan risk, stress-tested for a recession\n"
                "Enter a loan application. LoanLens estimates the chance of default, explains why, and shows "
                "whether a lender should approve it in a normal economy or a recession. Trained on 1990-2002 US "
                "SBA loans and tested on 2005-07 loans. *Prototype, not financial advice.*")
    with gr.Row():
        with gr.Column():
            state = gr.Dropdown(CATS["State"], value="CA" if "CA" in CATS["State"] else CATS["State"][0], label="Business state")
            bank_state = gr.Dropdown(CATS["BankState"], value="CA" if "CA" in CATS["BankState"] else CATS["BankState"][0], label="Bank's state")
            sector = gr.Dropdown(sector_choices, value="72" if "72" in CATS["naics_sector"] else sector_choices[0][1], label="Industry")
            amount = gr.Number(value=150000, label="Loan amount ($)")
            sba_pct = gr.Slider(0, 100, value=75, step=5, label="Share guaranteed by the SBA (%)")
            employees = gr.Number(value=8, label="Number of employees")
            jobs_created = gr.Number(value=2, label="Jobs the loan will create")
            jobs_retained = gr.Number(value=8, label="Jobs the loan will keep")
            new_biz = gr.Radio(["Existing business", "New business (under 2 years)"], value="Existing business", label="Business age")
            franchise = gr.Checkbox(label="Franchise")
            revline = gr.Checkbox(label="Revolving line of credit")
            lowdoc = gr.Checkbox(label="Fast-track 'LowDoc' loan")
            scenario = gr.Radio(list(SCENARIOS), value=list(SCENARIOS)[0], label="Economic scenario (stress test)")
            btn = gr.Button("Assess loan", variant="primary")
        with gr.Column():
            out_md = gr.Markdown()
            out_plot = gr.Plot()
    inputs = [state, bank_state, sector, amount, sba_pct, employees, jobs_created, jobs_retained,
              new_biz, franchise, revline, lowdoc, scenario]
    btn.click(assess, inputs, [out_md, out_plot])
    scenario.change(assess, inputs, [out_md, out_plot])

demo.launch()
