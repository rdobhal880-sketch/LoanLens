# LoanLens: does a lending model trained in good times survive a recession?

**Veritas AI International High School Hackathon 2026 · Track: AI + Finance · Ram Dobhal, GEMS World Academy Dubai**

> This project is a prototype and is not financial advice.

LoanLens helps a community lender and the small business applying to them. It takes a loan application and gives back the chance the loan defaults, the reasons behind that risk, and an approve or decline decision in a normal economy or under a recession stress test.

## Links
- **Live demo:** https://loanlens-dobhal.streamlit.app
- **Code and full analysis (Google Colab):** https://colab.research.google.com/drive/1rdNQhuaSAQtdd0ycQFZteLK-QwJaM40O?usp=sharing
- **Demo video:** https://docs.google.com/videos/d/1HNcrb-gCN-ZW4ccNixF5XkWdm4lJYQU_OEfR16556oU/play?usp=sharing

## Main finding
A model trained on 1990–2002 US small-business loans expected 15% of 2005–07 loans to default. **34% did.** Stress-testing lending decisions for a severe recession would have earned lenders an estimated **$327M more** on those loans, but the smallest businesses would pay the price: 68% of good borrowers seeking under $50,000 would be refused, against 21% for loans over $150,000.

| Method (tested on 224,363 loans from 2005–07) | PR-AUC | Predicted defaults |
|---|---|---|
| Approve everyone | 0.342 | – |
| Industry rule of thumb | 0.352 | 7.9% |
| Logistic regression | 0.386 | 12.7% |
| **LightGBM (final model)** | **0.457** | **15.1%** |
| What actually happened | | **34.2%** |

## How it works
- **Data:** 899,164 loans backed by the US Small Business Administration ([Kaggle](https://www.kaggle.com/datasets/mirbektoktogaraev/should-this-loan-be-approved-or-denied); Li, Mickel & Taylor, *Journal of Statistics Education*, 2018).
- **Split by time:** trained on 1990–2002, tuned on 2003–04, tested on 2005–07. Never split at random.
- **Three data leaks found and removed**, including a loan-term field that was overwritten after loans failed. It had made a first model look almost perfect (PR-AUC 0.917).
- **Model:** LightGBM with SHAP explanations for each applicant.
- **Decision rule:** approve only if expected interest beats expected loss, after the SBA guarantee.
- **Fairness audit** by urban/rural, business age and loan size.

## Files in this repository
| File | What it is |
|---|---|
| `streamlit_app.py` | The live demo app |
| `model.txt` | The trained LightGBM model |
| `meta.json` | Settings the app needs (categories, stress scenarios, margin, loss rate) |
| `requirements.txt` | Python packages the app installs |
| `.streamlit/config.toml` | The app's colour theme |
| `app.py` | An earlier Gradio version of the demo |

## Limits
US loans from 1990–2010 only; only loans that were approved; the 8% profit margin is an assumption; the stress-test levels were chosen after seeing the outcome, so they are a range, not a forecast.

## AI tools
Claude (Anthropic) helped write and debug code, check for data leakage and draft text. I made the decisions, ran every step and can explain each part.
