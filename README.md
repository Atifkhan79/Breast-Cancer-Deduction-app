# Breast Mass Diagnostic Reader

A Flask web app that scores 30 digitized FNA (fine-needle aspirate) cell
measurements against a logistic regression classifier trained on the
Wisconsin Diagnostic Breast Cancer dataset.

## What changed from the original notebook/app

1. **The saved model was retrained as a full pipeline.** The original
   `model.pkl` was a bare `LogisticRegression` that expected already-scaled
   input — but the `StandardScaler` used during training was never saved, and
   `app.py` was feeding it raw, unscaled numbers. That mismatch would silently
   produce wrong predictions. The new `model.pkl` bundles the scaler and the
   classifier together in one `sklearn.Pipeline`, so raw feature values go in
   and a correct prediction comes out.
2. **The stray `id` column is no longer a "feature."** The notebook's
   `X = breast.drop("diagnosis", axis=1)` left the row `id` in the training
   data (31 columns instead of 30), which the original app also expected as
   part of its comma-separated input. `id` carries no diagnostic signal, so
   it's dropped — the model now trains and predicts on the real 30
   measurements only. Accuracy on the held-out test set actually improved
   slightly (~97.9%) after this fix.
3. **The frontend is a real form**, not a single box where a user pastes 31
   comma-separated numbers. Each of the 10 cell characteristics (radius,
   texture, perimeter, area, smoothness, compactness, concavity, concave
   points, symmetry, fractal dimension) has three fields — mean, standard
   error, worst — matching how the dataset itself is structured. It's
   pre-filled with a real sample so a first-time visitor sees it working
   immediately, and it shows a confidence percentage alongside the reading.

## Project structure

```
.
├── app.py              # Flask app
├── model.pkl           # {"pipeline": sklearn Pipeline, "feature_names": [...]}
├── requirements.txt
├── Procfile            # for gunicorn on Render/Railway/Heroku-style platforms
├── templates/
│   └── index.html
└── static/
    └── style.css
```

## Run locally

```bash
pip install -r requirements.txt
python app.py            # dev server at http://127.0.0.1:5000
# or, production-style:
gunicorn app:app --bind 0.0.0.0:5000
```

## Deploy so anyone can use it (recommended: Render, free tier)

1. Push this folder to a GitHub repo (must include `app.py`, `model.pkl`,
   `requirements.txt`, `Procfile`, `templates/`, `static/`).
2. Go to [render.com](https://render.com) → sign in with GitHub → **New +** →
   **Web Service** → pick your repo.
3. Render auto-detects Python. Set:
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Instance Type:** Free
4. Click **Create Web Service**. Render builds and gives you a public URL
   like `https://your-app.onrender.com` within a couple of minutes.
5. Every future `git push` to that branch redeploys automatically.

Note: Render's free tier spins the service down after ~15 minutes of no
traffic, so the first request after idle time takes ~30–60 seconds to wake up.
That's fine for demos/personal use; upgrade to a paid instance for always-on.

### Alternatives
- **Railway** ([railway.app](https://railway.app)) — same GitHub-connect flow,
  also reads the `Procfile`, has a small free usage allowance.
- **PythonAnywhere** — good if you'd rather not touch GitHub; upload the
  files through their web UI and configure a Flask WSGI app.
- **Fly.io / a VPS + Docker** — more control, more setup; only worth it if you
  expect real traffic or need to scale.

## Important caveat

This is a machine-learning demo, not a medical device. It should not be used
for real diagnostic decisions — the UI says as much, and that disclaimer
should stay if you extend the project.
