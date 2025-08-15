# 📧 Email Priority Assistant

A Streamlit app that **classifies email urgency** (Urgent / Normal / Low), generates a **one-line intent summary**, and drafts a **professional reply** using an LLM.

> Built for internal IT support use-cases and course evaluation requirements.

---

## 🚀 Features

- **Priority classification**: TF-IDF + Logistic Regression (scikit-learn)
- **One-line intent summary**: OpenAI (`gpt-4o-mini` by default)
- **Reply drafting**: OpenAI (concise, bullet-point next steps)
- **Fallbacks**: Heuristic summary/reply if LLM unavailable
- **Simple UI**: Paste Subject + Body → click **Classify & Draft**

---

## 🧱 Project Structure

.
├─ app.py # Streamlit UI
├─ requirements.txt # Python deps (no Python version pin here)
├─ data/
│ └─ seed_samples.jsonl # Small seed training data
└─ src/
├─ classifier.py # TF-IDF + LogisticRegression training/loading
├─ llm_optional.py # OpenAI client + summarize/reply functions
└─ heuristics.py # Fallback summary/reply (no LLM)


---

## ⚙️ Setup (Local)

1) **Python & venv**  
```bash
python -m venv .venv
# Windows (Git Bash):
source .venv/Scripts/activate
# macOS/Linux:
# source .venv/bin/activate
```
2) **Install dependencies**
```bash
pip install -r requirements.txt
```
3) **Secrets (local)**
   Create .streamlit/secrets.toml (do not commit this file):
```bash
OPENAI_API_KEY = "sk-...your key..."
# Optional (default used if omitted)
OPENAI_MODEL   = "gpt-4o-mini"
```
4) **Run the app**
```bash
streamlit run app.py
```
☁️ Deploy (Streamlit Community Cloud)

Push to GitHub (repo: email-priority-assistant).

In Streamlit Cloud: New app → GitHub repo → Branch: main → Main file: app.py

Advanced settings: Python 3.11 recommended.

Settings → Secrets and paste:

```bash
OPENAI_API_KEY = "sk-...your key..."
OPENAI_MODEL   = "gpt-4o-mini"
```
5. **Deploy. Share the public app URL.**

Note: Never commit .streamlit/secrets.toml to Git.

🧠 Models

Classifier (classical ML)

Vectorizer: TF-IDF (unigram/bigram)

Estimator: Logistic Regression (multi-class)

Labels: Urgent, Normal, Low

Trained from data/seed_samples.jsonl (expandable)

LLM (OpenAI)

Default: gpt-4o-mini (good quality + cost-effective)

Reads key/model from st.secrets (see Secrets sections)

🖱 How to Use

Enter Subject and Body for an email.

Click Classify & Draft.

Review:

Email Priority (Urgent / Normal / Low)

Intent Summary (one line)

Reply Draft (editable, with next-step bullets)

Optionally Download Reply (.txt).

🔒 Security & Privacy

API keys are stored only in Streamlit Secrets (cloud) or .streamlit/secrets.toml (local).

No secrets are committed to Git.

Do not paste sensitive emails unless you trust the deployment environment.

🧰 Troubleshooting

“Missing OPENAI_API_KEY”: Add your key to Streamlit Secrets (cloud) or .streamlit/secrets.toml (local).

Dependency errors on deploy: Ensure requirements.txt has packages only (don’t pin Python there). Use Python 3.11 in Streamlit settings.

All emails classified as Low: Expand/rebalance seed_samples.jsonl and retrain, or tune class weights/thresholds in classifier.py.

👥 Team

Satya Mohan Reddy Ginni

Rakesh Kasaragadda

Vinay Pasam

Saketh Paruchuri

Assignment 10
Course:
AIDI-2000-02 - APPLIED MACH LEARNING
