# SkyGuard PK — Working Prototype

This is a fully working, tested pipeline. It was verified end-to-end here
using synthetic test data (since NASA's API isn't reachable from the build
sandbox) — on your machine, it will pull REAL NASA data.

## Run order (do these in sequence, in your project terminal)

```
pip install -r requirements.txt

python fetch_data.py       # pulls real NASA data (~5-10 min, 8 cities x 5 years)
python label_risk.py       # labels flood/heatwave risk
python train_model.py      # trains the Random Forest models, prints real metrics
streamlit run app.py       # opens the dashboard in your browser
```

## What each file does

- `fetch_data.py` — pulls historical weather/soil data from NASA POWER API
- `label_risk.py` — applies flood/heatwave risk thresholds (documented, self-defined)
- `train_model.py` — trains Random Forest classifiers, prints honest precision/recall
- `explain.py` — generates bilingual plain-language explanations
- `app.py` — the Streamlit dashboard (bilingual toggle, risk badges, trend chart)

## Next steps for YOU (important for the "IBM Bob usage" requirement)

1. Run the pipeline once so you have real data and real metrics.
2. Open this code in VS Code with IBM Bob, and ask it to:
   - Review the code and suggest improvements
   - Add a feature you want (e.g. a map view, an alert threshold slider)
   - Help debug anything that errors on your machine
3. Note what Bob changed/suggested — this becomes your "How IBM Bob was used"
   section in the README.
4. Replace the placeholder threshold values in `label_risk.py` with better
   ones if you find real disaster-record data to calibrate against (optional,
   strengthens the submission but not required).

## Known limitation to mention honestly in your README

The flood/heatwave risk thresholds are self-defined approximations, not
official disaster-management classifications. This is a legitimate and
honest way to frame it — don't claim official validation you don't have.
