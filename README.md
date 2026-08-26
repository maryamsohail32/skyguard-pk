# SkyGuard PK

**NASA satellite data → plain-language flood & heatwave risk alerts for Pakistan, in Urdu and English.**

## Problem Statement

Pakistan faces recurring flood and heatwave disasters, yet the satellite-derived
weather and soil data that could provide early warning (collected daily by NASA)
remains largely inaccessible to the people who need it most — local authorities,
communities, and students — due to its raw, technical format and lack of
localized, bilingual interpretation.

## Solution Description

SkyGuard PK pulls historical and near-real-time weather data (precipitation,
temperature, humidity, soil moisture) from NASA's POWER API for 8 major
Pakistani cities, uses a trained machine learning model to classify flood and
heatwave risk, and presents it through an interactive bilingual (Urdu/English)
dashboard — including a country-wide risk map, per-city plain-language
explanations, and adjustable sensitivity controls for exploring risk thresholds.

## AI Approach & Architecture

```
NASA POWER API (8 Pakistani cities, 5 years historical data)
        │
        ▼
Feature engineering (7-day rolling rainfall, soil wetness, temperature, humidity)
        │
        ▼
Random Forest Classifier (separate models for flood risk and heatwave risk)
        │
        ▼
Plain-language bilingual explanation layer
        │
        ▼
Streamlit dashboard:
  - Pakistan-wide risk map (color-coded by city)
  - Per-city risk badges + explanations
  - Adjustable sensitivity sliders (clearly labeled as distinct from model output)
```

**Model:** Random Forest classifier, trained with `class_weight="balanced"` to
handle the natural imbalance between LOW/MEDIUM/HIGH risk days.

**Honest limitation:** Flood and heatwave risk thresholds (e.g., 100mm/7-day
rainfall, 40°C extreme heat) are self-defined, reasonable approximations based
on commonly cited regional triggers — not official government disaster
classifications. This is disclosed rather than overstated.

## Selected Challenge Theme

Advance Space Exploration with AI

## How IBM Bob Was Used

IBM Bob was used throughout development for genuine code review and design
decisions, not just code generation:

1. **Edge-case review of risk logic** — Asked Bob to review `label_risk.py`'s
   threshold logic for edge cases. Bob identified 8 issues, including a
   high-severity gap (humidity was loaded but never used in heatwave
   classification, despite wet-bulb temperature being the real physiological
   risk driver in Pakistan's pre-monsoon season) and a medium-severity gap
   (extreme rainfall alone could be under-labeled as MEDIUM instead of HIGH
   when soil wetness didn't independently cross its own threshold). These are
   documented as known limitations given project time constraints.

2. **Pakistan map view** — Asked Bob to add a map showing all 8 cities colored
   by current risk level. Bob implemented this using Streamlit's native
   Vega-Lite support rather than adding an external mapping dependency,
   keeping the project lightweight.

3. **Alert sensitivity sliders** — Asked Bob to add sliders so users could
   interactively adjust risk thresholds and explore sensitivity.

4. **Model integrity fix (design decision, Bob-implemented)** — After
   reviewing Bob's slider implementation, identified a risk: sliders were
   bypassing the trained model entirely, which would misrepresent the
   project's core claim — a real trained ML model, not just adjustable rules.
   Directed Bob to restore the trained Random Forest model as the default
   prediction source for both the map and per-city badges, with the
   slider-based view only activating when a user explicitly moves a slider
   away from its default, clearly labeled "Adjusted view — not model
   prediction" to keep the two outputs unambiguous.

This iterative process — generate, review, catch a design flaw, redirect —
reflects how the project treats AI-assisted development: Bob accelerates
implementation, but every output was reviewed against what the project
actually needed to honestly claim.

## Tech Stack

Python, pandas, scikit-learn, Streamlit, NASA POWER API, IBM Bob

## Setup

See `RUN_ME_FIRST.md` for local setup and run instructions.
