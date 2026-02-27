from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

# ===============================
# 1) Inicializar Flask
# ===============================
app = Flask(__name__)
CORS(app)

# ===============================
# 2) Cargar modelo
# ===============================
try:
    model = joblib.load("best_model_overall.pkl")
    print("✅ Modelo cargado correctamente.")
except Exception as e:
    model = None
    print("❌ Error cargando el modelo:", e)

# ===============================
# 3) ORDEN EXACTO DE FEATURES
# (debe coincidir con entrenamiento)
# ===============================
FEATURE_ORDER = [
    "rp_count",
    "rp_mean",
    "rp_std",
    "rp_diff5",
    "rp_diff7",
    "rp_diff10",
    "lp_count",
    "lp_mean",
    "lp_std",
    "lp_diff5",
    "lp_diff7",
    "lp_diff10",
    "tap_diff"
]

# ===============================
# 4) Endpoint de predicción
# ===============================
@app.route("/predict", methods=["POST"])
def predict():

    if model is None:
        return jsonify({"error": "Modelo no cargado"}), 500

    data = request.get_json()

    try:
        features = np.array([[data["features"][f] for f in FEATURE_ORDER]])
    except KeyError as e:
        return jsonify({
            "error": f"Falta la variable: {str(e)}"
        }), 400

    prediction = int(model.predict(features)[0])
    probability = float(model.predict_proba(features)[0][1])

    return jsonify({
        "prediction": prediction,
        "probability": probability,
        "label": "Bradykinesia" if prediction == 1 else "Normal"
    })

# ===============================
# 5) Ejecutar servidor
# ===============================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
