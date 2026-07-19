from flask import Flask, request, render_template
import numpy as np
import pickle
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model.pkl")

# The pickle stores a dict: {"pipeline": <sklearn Pipeline w/ scaler+model>, "feature_names": [...]}
with open(MODEL_PATH, "rb") as f:
    bundle = pickle.load(f)

pipeline = bundle["pipeline"]
FEATURE_NAMES = bundle["feature_names"]

app = Flask(
    __name__,
    template_folder=os.path.join(BASE_DIR, "templates"),
    static_folder=os.path.join(BASE_DIR, "static"),
)

# Groups drive the 3-column layout in the form (mirrors how the dataset itself
# is structured: mean / standard-error / worst reading per cell measurement)
BASE_METRICS = [
    ("radius", "Radius"),
    ("texture", "Texture"),
    ("perimeter", "Perimeter"),
    ("area", "Area"),
    ("smoothness", "Smoothness"),
    ("compactness", "Compactness"),
    ("concavity", "Concavity"),
    ("concave points", "Concave Points"),
    ("symmetry", "Symmetry"),
    ("fractal_dimension", "Fractal Dimension"),
]
GROUPS = [
    ("mean", "Mean"),
    ("se", "Standard Error"),
    ("worst", "Worst"),
]

# A real (benign) sample from the training data, used to pre-fill the form
# so first-time users see the tool working instead of an empty sheet.
SAMPLE_VALUES = {
    "radius_mean": 12.47, "texture_mean": 18.60, "perimeter_mean": 81.09, "area_mean": 481.9,
    "smoothness_mean": 0.09965, "compactness_mean": 0.1058, "concavity_mean": 0.08005,
    "concave points_mean": 0.03821, "symmetry_mean": 0.1925, "fractal_dimension_mean": 0.06373,
    "radius_se": 0.3961, "texture_se": 1.044, "perimeter_se": 2.497, "area_se": 30.29,
    "smoothness_se": 0.006953, "compactness_se": 0.01911, "concavity_se": 0.02701,
    "concave points_se": 0.01037, "symmetry_se": 0.01782, "fractal_dimension_se": 0.003586,
    "radius_worst": 14.97, "texture_worst": 24.64, "perimeter_worst": 96.05, "area_worst": 677.9,
    "smoothness_worst": 0.1426, "compactness_worst": 0.2378, "concavity_worst": 0.2671,
    "concave points_worst": 0.1015, "symmetry_worst": 0.3014, "fractal_dimension_worst": 0.0875,
}


def field_id(metric_key, group_key):
    return f"{metric_key}_{group_key}"


@app.route("/")
def index():
    return render_template(
        "index.html",
        base_metrics=BASE_METRICS,
        groups=GROUPS,
        field_id=field_id,
        values=SAMPLE_VALUES,
        result=None,
    )


@app.route("/predict", methods=["POST"])
def predict():
    errors = []
    row = []
    submitted = {}

    # IMPORTANT: build the row in the same order as FEATURE_NAMES
    # (all "mean" features, then all "se" features, then all "worst" features).
    # The form is laid out by metric for readability, but the model expects
    # the dataset's original column order.
    for group_key, _ in GROUPS:
        for metric_key, _ in BASE_METRICS:
            fid = field_id(metric_key, group_key)
            raw = request.form.get(fid, "").strip()
            submitted[fid] = raw
            try:
                row.append(float(raw))
            except ValueError:
                errors.append(fid)

    result = None
    if errors:
        result = {
            "ok": False,
            "message": f"{len(errors)} field(s) need a valid number before this can be scored.",
        }
    else:
        input_data = np.array(row, dtype=float).reshape(1, -1)
        prediction = pipeline.predict(input_data)[0]
        proba = pipeline.predict_proba(input_data)[0]
        confidence = proba[int(prediction)] * 100

        is_malignant = bool(prediction == 1)
        result = {
            "ok": True,
            "malignant": is_malignant,
            "label": "Yes — this reading indicates cancer" if is_malignant else "No — this reading does not indicate cancer",
            "confidence": round(confidence, 1),
            "low_confidence": confidence < 70,
        }

    return render_template(
        "index.html",
        base_metrics=BASE_METRICS,
        groups=GROUPS,
        field_id=field_id,
        values=submitted,
        result=result,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
