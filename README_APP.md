# Kirana Ordering Assistant — Team Guide

Welcome to the **Kirana Ordering Assistant**, an educational prototype designed for an FMCG distributor that supplies 300 kirana stores across Mumbai and Pune.

Shopkeepers send orders on WhatsApp in everyday Hinglish (*"bhaiya 2 peti parle-g bhej do"*), using local trade units (*peti*, *dabba*, *ladi*), nicknames (*"makhan"*, *"pocha liquid"*), and occasional typos.

---

## 💡 Core Architecture & Principles

- **The LLM proposes, Python validates:** Large Language Models extract messy text into candidate items, but Python strictly validates every SKU ID against the 40-product catalog.
- **Deterministic Pricing:** Prices, case sizes, and rupee totals **always** come from catalog truth (`data/catalog.json`), never from model generation.
- **The Human Checkpoint:** Nothing is dispatched until the shopkeeper reviews item names, pack sizes, and rupee totals on the confirmation screen.
- **Smart Reorder Nudges:** A Gradient Boosting classifier (`reorder_model.py`) predicts items a store usually orders but left out, triggering at most 2 polite reminders.

---

## 🚀 Running Locally

### Step 1: Install Dependencies
Open your terminal in this project folder and run:
```bash
pip install -r requirements.txt
```

### Step 2: Add your Groq API Key
1. Get a free API key at [Groq Console](https://console.groq.com/keys).
2. Open `.streamlit/secrets.toml` in your code editor.
3. Replace the placeholder with your key:
```toml
groq_api_key = "gsk_your_actual_key_here"
```
*(Note: `.streamlit/secrets.toml` is ignored by Git, ensuring your key is never committed to GitHub).*

### Step 3: Launch the Application
Run:
```bash
streamlit run app.py
```
The browser will automatically open at `http://localhost:8501`.

---

## ☁️ Deploying to Streamlit Community Cloud (Free)

Deploying takes under 3 minutes:

1. **Push this folder to a GitHub repository:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit of Kirana Ordering Assistant"
   git branch -M main
   git remote add origin https://github.com/<your-username>/kirana-ordering-assistant.git
   git push -u origin main
   ```
2. Go to [share.streamlit.io](https://share.streamlit.io/) and click **"New app"**.
3. Select your repository, branch (`main`), and set **Main file path** to `app.py`.
4. Click **"Advanced settings..."** before deploying.
5. In the **Secrets** text box, paste your Groq key:
   ```toml
   groq_api_key = "gsk_your_actual_key_here"
   ```
6. Click **Deploy!** Your app will be live on a public link.

---

## 🛠️ Changing Rules & Templates (No Code Edits Required)

All business rules, thresholds, and conversational Hinglish templates are managed via markdown files in the project root. The app reads these files fresh on every run.

### 1. Adjusting Thresholds (`ordering-rules.md`)
Open `ordering-rules.md` and edit the YAML block at the top:
```yaml
---
match_threshold_high: 85        # Retrieval score (0-100) for clear auto-match
match_threshold_low: 55         # Retrieval score below which an item is marked "Not in catalog"
match_margin: 8                 # Margin to trigger follow-up between close pack sizes
nudge_probability_threshold: 0.7  # Model confidence required before nudging (e.g. 0.7 = 70%)
max_nudges_per_order: 2         # Maximum nudges per order
quantity_flag_multiple: 3       # Multiplier above store median to flag as high quantity
retrieval_top_k: 5              # Number of catalog candidates sent to the LLM
---
```
Save the file and refresh the app in your browser — the behavior changes immediately.

### 2. Editing Hinglish Wording (`message-templates.md`)
Open `message-templates.md` to edit the exact phrasing sent to shopkeepers under headings such as:
- `## greeting`
- `## followup_choice`
- `## nudge`
- `## confirmed`

---

## 🎯 Testing the Four Required Demo Cases

The sidebar contains 4 one-click demo buttons:

1. **Clean Order (Store ST012):**
   - *Message:* *"Namaste bhaiya, 2 peti Parle-G 100g, 1 peti Maggi aur 5 packet Tata Salt 1kg bhej do"*
   - *Expected:* All 3 products matched to catalog, quantities converted to pieces, rupee totals calculated, ready for confirmation.
2. **Unclear Order (Store ST045):**
   - *Message:* *"bhaiya 3 dabba parle aur 2 peti oil bhejna"*
   - *Expected:* Triggers interactive clarification buttons: *Parle-G 50g ya 100g?* and which oil (*Fortune / Saffola / Dhara*).
3. **Reorder Nudge (Store ST238):**
   - *Message:* *"2 peti hide and seek aur 1 dabba MDH mirch bhej do"*
   - *Expected:* Store ST238 buys Fortune oil almost every week. The app notices oil is missing from the message and politely asks: *"Aapne is baar Fortune oil order nahi kiya, bhej dein?"* with Yes/No buttons.
4. **Item Not in Catalog (Store ST101):**
   - *Message:* *"1 peti good day aur 2 box Ferrero Rocher bhej do"*
   - *Expected:* Good Day matches cleanly; Ferrero Rocher is flagged in red as *"Not in catalog"* and never silently substituted with an incorrect product.

---

## 📊 App Screens & Tabs

- **💬 A. Shop Chat (WhatsApp):** Interactive conversational interface with green shopkeeper chat bubbles, interactive follow-up buttons, reorder nudges, editable review table, and dispatch confirmation.
- **📋 B. Staff: Confirmed Orders:** Live flat table of all confirmed orders, store filter, CSV export, and catalog verification details.
- **🎯 C. Accuracy (Test Set):** Runs evaluation on 150 real-world synthetic messages across 8 categories (Hinglish, nicknames, typos, mixed units, prompt injections), measuring Item, Quantity, and Unit accuracy separately. Resumes from `cache/eval_results.json`.
- **📈 D. Reorder Model:** Benchmarks the Gradient Boosting classifier against heuristic baselines (Rule 1 & Rule 2), displays feature importance, threshold trade-offs, and synthetic-data bias stress test results.
- **🛡️ E. Risks & Checkpoints:** Detailed cards covering the 4 key AI risks (misread units, hallucinations, pushy nudges, synthetic bias) with live metrics and an explanation of why warehouse-gate human checkpoints prevent costly return trips.
