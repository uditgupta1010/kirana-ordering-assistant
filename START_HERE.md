# Kirana Ordering Assistant — start here

Everything the app needs is already in this folder. The only thing you add is your Groq key.

## What's in the folder

| File | What it is |
|---|---|
| `BUILD_PROMPT.txt` | The full prompt to paste into the AI coding tool (Antigravity) |
| `data/catalog.json` | 40 products with prices, case sizes, nicknames and the unit map (peti, dabba, ladi...) |
| `data/order_history.csv` | 300 shops x 26 weeks of orders (synthetic). Trains the nudge model |
| `data/whatsapp_test_set.json` | 150 test messages with correct answers, in 8 types (clean, Hinglish, typos, nicknames, mixed units, unclear, not in catalog, prompt injection) |
| `data/demo_cases.json` | The 4 demo messages the course asks for |
| `ordering-rules.md` | The rules the AI follows + settings you can change (thresholds, max nudges, 3x flag) |
| `message-templates.md` | The exact Hinglish replies the app sends |
| `.streamlit/secrets.toml` | Where your Groq key goes (hidden file) |
| `.gitignore` | Stops the key from being uploaded to GitHub (hidden file) |

## Steps

1. **Unzip** this folder somewhere easy (e.g. Desktop).
2. **Get a free Groq key:** go to https://console.groq.com/keys, sign in, click "Create API Key", copy it.
3. **Open the folder in Antigravity** (File > Open Folder).
4. **Paste the key:** open `.streamlit/secrets.toml` and replace `PASTE_YOUR_GROQ_KEY_HERE` with your key (keep the quotes). Save.
   - If you can't see the `.streamlit` folder, it's hidden; it still shows in Antigravity's file panel.
5. **Paste the prompt:** open `BUILD_PROMPT.txt`, copy all of it, and paste it into Antigravity's agent chat. Add this line at the top:
   *"All input files are already in this folder. Read them first and do not overwrite them. Then build the app."*
6. When it finishes, ask it: *"Install the requirements and run the app."* (Or in the terminal: `pip install -r requirements.txt` then `streamlit run app.py`.)
7. Try the 4 demo buttons in the sidebar.

## Things to know

- **Model numbers:** with this data the nudge model scores roughly precision 0.78, recall 0.75, F1 0.77 (vs 0.70 F1 for the best simple rule). That's close to the midterm report (81% / 75% / 0.78) but not identical, because this is a freshly generated dataset. Update the report table with whatever the app shows.
- **Irregular shops:** stores ST271–ST300 order much less regularly on purpose, so the synthetic-data bias test has something to find (F1 drops to about 0.63 on them).
- **Test messages** were generated with known answers. Skim a few and fix any odd ones before reporting accuracy.
- **Nudge demo:** uses shop ST238, which buys Fortune oil almost every week; the demo message leaves oil out. If you change the data, re-check this case.
- The clean-order demo may also show a nudge if that shop usually buys something else; that's expected.
- **Never** paste the Groq key into the chat, the code, or GitHub.
