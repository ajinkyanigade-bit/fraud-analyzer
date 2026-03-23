"""
FraudSense AI — Cloud Banking Fraud Analyzer
Single-file version: Flask + ML model all in one file.
Run: python app.py
"""
import sys, os, random, uuid, logging
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Optional

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.utils import resample
from flask import Flask, request, jsonify, render_template

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# ══════════════════════════════════════════════════════════════════════════════
# FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════
LOCATION_RISK = {
    "Mumbai":0,"Delhi":0,"New York":0,"London":0,"Toronto":0,"Singapore":0,
    "Dubai":1,"Different State":1,"Foreign Country":2,
    "Lagos":3,"Bucharest":3,"Unknown IP":3,"High-Risk Region":3,
}
DEVICE_RISK = {"known":0,"new":1,"vpn":2,"tor":3}
TXN_TYPE_MAP = {
    "bill_payment":0,"pos_payment":1,"atm_withdrawal":2,
    "online_transfer":3,"wire_transfer":4,"crypto_exchange":5,
}
FEATURE_NAMES = [
    "amount_log","amount_bin","location_risk","device_risk","txn_type_encoded",
    "is_crypto","is_wire","hour_of_day","is_late_night","is_business_hours",
    "txns_last_hour","high_velocity","combined_risk","amount_x_velocity",
    "late_night_foreign","crypto_foreign",
]

def transform(raw: dict) -> np.ndarray:
    amount   = float(raw.get("amount", 0))
    location = str(raw.get("location", ""))
    device   = str(raw.get("device_type", "known")).lower()
    txn_type = str(raw.get("transaction_type", "")).lower()
    hour     = int(raw.get("hour_of_day", 12))
    velocity = int(raw.get("txns_last_hour", 1))

    amount_log  = np.log1p(amount)
    amount_bin  = (0 if amount<100 else 1 if amount<500 else 2 if amount<2000 else 3 if amount<10000 else 4)
    loc_risk    = LOCATION_RISK.get(location, 1)
    dev_risk    = DEVICE_RISK.get(device, 1)
    txn_enc     = TXN_TYPE_MAP.get(txn_type, 3)
    is_crypto   = int(txn_type == "crypto_exchange")
    is_wire     = int(txn_type == "wire_transfer")
    is_late     = int(1 <= hour <= 5)
    is_biz      = int(9 <= hour <= 17)
    hi_velocity = int(velocity > 5)
    combined    = loc_risk * dev_risk
    amt_x_vel   = amount_log * velocity
    late_foreign   = int(is_late and loc_risk >= 2)
    crypto_foreign = int(is_crypto and loc_risk >= 2)

    return np.array([[
        amount_log, amount_bin, loc_risk, dev_risk, txn_enc,
        is_crypto, is_wire, hour, is_late, is_biz,
        velocity, hi_velocity, combined, amt_x_vel,
        late_foreign, crypto_foreign,
    ]], dtype=np.float32)


# ══════════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATASET
# ══════════════════════════════════════════════════════════════════════════════
def _loc(r): return {0:"Mumbai",1:"Dubai",2:"Foreign Country",3:"Lagos"}[r]
def _dev(r): return {0:"known",1:"new",2:"vpn",3:"tor"}[r]
def _txn(e): return {0:"bill_payment",1:"pos_payment",2:"atm_withdrawal",3:"online_transfer",4:"wire_transfer",5:"crypto_exchange"}[e]

def generate_dataset(n=20000, fraud_rate=0.04, seed=42):
    rng = np.random.default_rng(seed)
    rows, labels = [], []
    n_fraud = int(n * fraud_rate)
    n_legit = n - n_fraud

    # Legitimate transactions — realistic noise added
    for _ in range(n_legit):
        # Occasionally legit txns have high velocity or odd hours (noise)
        vel = int(rng.integers(1, 4)) if rng.random() > 0.1 else int(rng.integers(4, 9))
        hr  = int(rng.integers(7, 22)) if rng.random() > 0.08 else int(rng.integers(0, 7))
        rows.append(transform({
            "amount":           float(rng.lognormal(5.5, 1.4)),
            "location":         _loc(int(rng.choice([0,1,2,3], p=[0.60,0.22,0.12,0.06]))),
            "device_type":      _dev(int(rng.choice([0,1,2,3], p=[0.75,0.15,0.07,0.03]))),
            "transaction_type": _txn(int(rng.choice([0,1,2,3,4,5], p=[0.25,0.33,0.16,0.14,0.07,0.05]))),
            "hour_of_day":      hr,
            "txns_last_hour":   vel,
        })[0])
        labels.append(0)

    # Fraudulent transactions — realistic noise added
    for _ in range(n_fraud):
        # Some fraud txns have low velocity to evade detection
        vel = int(rng.integers(5, 14)) if rng.random() > 0.2 else int(rng.integers(1, 5))
        hr  = int(rng.choice([1,2,3,4,22,23,0])) if rng.random() > 0.25 else int(rng.integers(8, 20))
        rows.append(transform({
            "amount":           float(rng.lognormal(7.2, 2.0)),
            "location":         _loc(int(rng.choice([0,1,2,3], p=[0.08,0.12,0.22,0.58]))),
            "device_type":      _dev(int(rng.choice([0,1,2,3], p=[0.06,0.16,0.32,0.46]))),
            "transaction_type": _txn(int(rng.choice([0,1,2,3,4,5], p=[0.03,0.06,0.10,0.18,0.21,0.42]))),
            "hour_of_day":      hr,
            "txns_last_hour":   vel,
        })[0])
        labels.append(1)

    X = np.array(rows, dtype=np.float32)
    y = np.array(labels, dtype=np.int32)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════
@dataclass
class Metrics:
    accuracy:float=0; precision:float=0; recall:float=0; f1:float=0; auc_roc:float=0

class FraudModel:
    WEIGHTS = {"gbm":0.55,"rf":0.30,"iso":0.15}
    HIGH_T, MED_T = 70, 40

    def __init__(self):
        self._gbm=None; self._rf=None; self._iso=None
        self._scaler=StandardScaler()
        self.m_gbm=Metrics(); self.m_rf=Metrics()
        self.accuracy=0.0; self.feature_importances={}; self._trained=False

    def train(self, X=None, y=None):
        if X is None:
            logger.info("Generating synthetic dataset (20,000 samples)...")
            X, y = generate_dataset()

        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

        # Oversample minority
        X_maj, X_min = X_tr[y_tr==0], X_tr[y_tr==1]
        if len(X_min) > 0:
            X_min_up = resample(X_min, replace=True, n_samples=len(X_maj)//2, random_state=42)
            X_tr = np.vstack([X_maj, X_min_up])
            y_tr = np.hstack([np.zeros(len(X_maj)), np.ones(len(X_min_up))]).astype(np.int32)

        X_tr_sc = self._scaler.fit_transform(X_tr)
        X_te_sc = self._scaler.transform(X_te)

        # 1. Gradient Boosting (primary — XGBoost equivalent)
        self._gbm = GradientBoostingClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.08,
            subsample=0.75, min_samples_leaf=10,
            max_features=0.7, random_state=42)
        self._gbm.fit(X_tr, y_tr)
        p = self._gbm.predict(X_te); pb = self._gbm.predict_proba(X_te)[:,1]
        self.m_gbm = Metrics(round(accuracy_score(y_te,p),4), round(precision_score(y_te,p,zero_division=0),4),
                             round(recall_score(y_te,p,zero_division=0),4), round(f1_score(y_te,p,zero_division=0),4),
                             round(roc_auc_score(y_te,pb),4))
        logger.info(f"GBM → acc={self.m_gbm.accuracy:.3f}  auc={self.m_gbm.auc_roc:.3f}")

        # 2. Random Forest (secondary)
        self._rf = RandomForestClassifier(
            n_estimators=150, max_depth=8, class_weight="balanced",
            min_samples_leaf=5, max_features="sqrt",
            random_state=42, n_jobs=-1)
        self._rf.fit(X_tr_sc, y_tr)
        p2 = self._rf.predict(X_te_sc); pb2 = self._rf.predict_proba(X_te_sc)[:,1]
        self.m_rf = Metrics(round(accuracy_score(y_te,p2),4), round(precision_score(y_te,p2,zero_division=0),4),
                            round(recall_score(y_te,p2,zero_division=0),4), round(f1_score(y_te,p2,zero_division=0),4),
                            round(roc_auc_score(y_te,pb2),4))
        logger.info(f"RF  → acc={self.m_rf.accuracy:.3f}")

        # 3. Isolation Forest (unsupervised anomaly)
        self._iso = IsolationForest(n_estimators=150, contamination=0.04, random_state=42)
        self._iso.fit(X_tr_sc)

        self.accuracy = self.m_gbm.accuracy
        if hasattr(self._gbm, "feature_importances_"):
            raw = self._gbm.feature_importances_
            total = raw.sum() or 1.0
            self.feature_importances = {n: round(float(v/total)*100,1) for n,v in zip(FEATURE_NAMES, raw)}

        self._trained = True
        logger.info("Model training complete.")

    def predict(self, features: np.ndarray) -> dict:
        if not self._trained:
            raise RuntimeError("Call train() first.")
        sc = self._scaler.transform(features)
        p_gbm = float(self._gbm.predict_proba(features)[:,1][0])
        p_rf  = float(self._rf.predict_proba(sc)[:,1][0])
        iso_s = self._iso.decision_function(sc)[0]
        p_iso = float(np.clip((-iso_s + 0.5), 0.0, 1.0))
        W = self.WEIGHTS
        ensemble = W["gbm"]*p_gbm + W["rf"]*p_rf + W["iso"]*p_iso
        risk_score = int(np.clip(round(ensemble*100), 0, 99))

        if risk_score >= self.HIGH_T:
            level = "high"; decision = "BLOCK" if risk_score >= 85 else "REVIEW"
        elif risk_score >= self.MED_T:
            level = "medium"; decision = "REVIEW"
        else:
            level = "low"; decision = "APPROVE"

        confidence = int(np.clip(80 + risk_score*0.18, 80, 99))
        return {
            "risk_score":   risk_score, "risk_level": level,
            "decision":     decision,   "confidence": confidence,
            "risk_factors": self._explain(features[0], risk_score),
            "model_scores": {
                "gradient_boosting": round(p_gbm*100,1),
                "random_forest":     round(p_rf*100,1),
                "isolation_forest":  round(p_iso*100,1),
                "ensemble":          round(ensemble*100,1),
            }
        }

    def _explain(self, v, score):
        factors = []
        if v[1] >= 3: factors.append("Very high transaction amount")
        elif v[1] == 2: factors.append("Above-average amount")
        if v[2] == 3: factors.append("High-risk geographic location")
        elif v[2] == 2: factors.append("Foreign country transaction")
        elif v[2] == 1: factors.append("Out-of-region transaction")
        if v[3] == 3: factors.append("Tor/anonymising network detected")
        elif v[3] == 2: factors.append("VPN usage detected")
        elif v[3] == 1: factors.append("Unrecognised device")
        if v[8]: factors.append(f"Late-night transaction (hour {int(v[7])}:00)")
        if v[11] or v[10] >= 5: factors.append(f"High velocity: {int(v[10])} txns/hr")
        if v[5]: factors.append("Crypto exchange (irreversible)")
        elif v[6]: factors.append("Wire transfer (irreversible)")
        if v[12] >= 6: factors.append("Dangerous device+location combo")
        return factors or ["No significant risk factors detected"]

    def info(self):
        return {
            "name": "Gradient Boosting Ensemble",
            "version": "2.4.0", "trained": self._trained,
            "accuracy": round(self.accuracy*100, 2),
            "gradient_boosting": {k: round(getattr(self.m_gbm,k)*100,2) for k in ["accuracy","precision","recall","f1","auc_roc"]},
            "random_forest":     {k: round(getattr(self.m_rf,k)*100,2)  for k in ["accuracy","precision","recall","f1","auc_roc"]},
            "feature_importances": self.feature_importances,
            "features": FEATURE_NAMES,
        }


# ══════════════════════════════════════════════════════════════════════════════
# FLASK APP
# ══════════════════════════════════════════════════════════════════════════════
app = Flask(__name__, template_folder=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates'))

@app.after_request
def cors(r):
    r.headers["Access-Control-Allow-Origin"]  = "*"
    r.headers["Access-Control-Allow-Headers"] = "Content-Type"
    r.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    return r

def _now(): return datetime.now(timezone.utc).isoformat().replace("+00:00","Z")

logger.info("Booting FraudSense AI...")
fraud_model = FraudModel()
fraud_model.train()

TRANSACTIONS, ALERTS = [], []

def _run(raw):
    pred  = fraud_model.predict(transform(raw))
    level = pred["risk_level"]
    txn = {
        "id": "TXN-"+uuid.uuid4().hex[:8].upper(),
        "account": raw.get("account","Unknown"), "amount": raw["amount"],
        "merchant": raw.get("merchant","Unknown"), "location": raw["location"],
        "device_type": raw["device_type"], "transaction_type": raw["transaction_type"],
        "hour_of_day": raw["hour_of_day"], "txns_last_hour": raw["txns_last_hour"],
        **pred, "status": "flagged" if level=="high" else "review" if level=="medium" else "clean",
        "timestamp": _now(),
    }
    if level in ("high","medium"):
        ALERTS.append({"id":"ALT-"+uuid.uuid4().hex[:6].upper(),"severity":level,
            "title":f"{'High' if level=='high' else 'Medium'} risk: {txn['account']}",
            "desc":f"${txn['amount']:,.2f} | {raw['transaction_type'].replace('_',' ').title()} from {raw['location']}",
            "txn_id":txn["id"],"time":_now(),"dismissed":False})
    return txn

# Seed with sample data
_accs  = ["Alice Chen","Bob Kumar","Sara Patel","James Wu","Priya Nair","Mike Torres","Aisha Malik","Raj Sharma"]
_merch = ["Amazon","Netflix","Shell Oil","Zara","Unknown Vendor","Crypto Exchange","Emirates Bank","Walmart"]
_locs  = ["Mumbai","Delhi","New York","London","Dubai","Singapore","Lagos","Bucharest"]
_devs  = ["known","new","vpn","tor"]
_types = ["online_transfer","atm_withdrawal","pos_payment","wire_transfer","crypto_exchange","bill_payment"]
for _ in range(40):
    TRANSACTIONS.append(_run({"amount":round(random.uniform(10,15000),2),
        "transaction_type":random.choice(_types),"location":random.choice(_locs),
        "device_type":random.choice(_devs),"hour_of_day":random.randint(0,23),
        "txns_last_hour":random.randint(1,12),"merchant":random.choice(_merch),
        "account":random.choice(_accs)}))
logger.info(f"Seeded {len(TRANSACTIONS)} sample transactions.")


@app.route("/dashboard")
def dashboard():
    return render_template("index.html")

@app.route("/",methods=["GET"])
def health():
    return jsonify({"service":"FraudSense AI","status":"running","model":fraud_model.info(),"timestamp":_now()})

@app.route("/api/analyze",methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    missing = [k for k in ["amount","transaction_type","location","device_type","hour_of_day","txns_last_hour"] if k not in data]
    if missing: return jsonify({"error":f"Missing: {missing}"}), 400
    return jsonify(fraud_model.predict(transform(data)))

@app.route("/api/transactions",methods=["GET"])
def get_txns():
    risk=request.args.get("risk"); search=request.args.get("search","").lower()
    page=int(request.args.get("page",1)); limit=int(request.args.get("limit",20))
    txns=list(reversed(TRANSACTIONS))
    if risk:   txns=[t for t in txns if t["risk_level"]==risk]
    if search: txns=[t for t in txns if search in t["id"].lower() or search in t["account"].lower()]
    total=len(txns); txns=txns[(page-1)*limit:page*limit]
    return jsonify({"data":txns,"total":total,"page":page,"limit":limit})

@app.route("/api/transactions",methods=["POST"])
def add_txn():
    txn=_run(request.get_json(force=True)); TRANSACTIONS.append(txn)
    return jsonify(txn), 201

@app.route("/api/transactions/<tid>",methods=["GET"])
def get_txn(tid):
    t=next((t for t in TRANSACTIONS if t["id"]==tid),None)
    return jsonify(t) if t else (jsonify({"error":"Not found"}),404)

@app.route("/api/transactions/<tid>/block",methods=["POST"])
def block_txn(tid):
    global TRANSACTIONS; TRANSACTIONS=[t for t in TRANSACTIONS if t["id"]!=tid]
    return jsonify({"message":f"{tid} blocked."})

@app.route("/api/alerts",methods=["GET"])
def get_alerts():
    active=[a for a in reversed(ALERTS) if not a["dismissed"]]
    return jsonify({"data":active,"total":len(active)})

@app.route("/api/alerts/<aid>/dismiss",methods=["POST"])
def dismiss(aid):
    a=next((a for a in ALERTS if a["id"]==aid),None)
    if not a: return jsonify({"error":"Not found"}),404
    a["dismissed"]=True; return jsonify({"message":"Dismissed."})

@app.route("/api/metrics",methods=["GET"])
def metrics():
    total=len(TRANSACTIONS)
    return jsonify({"total_transactions":total,
        "flagged":sum(1 for t in TRANSACTIONS if t["risk_level"]=="high"),
        "under_review":sum(1 for t in TRANSACTIONS if t["risk_level"]=="medium"),
        "clean":sum(1 for t in TRANSACTIONS if t["risk_level"]=="low"),
        "model_accuracy":round(fraud_model.accuracy*100,1),
        "savings_blocked":round(sum(t["amount"] for t in TRANSACTIONS if t["risk_level"]=="high"),2),
        "average_risk_score":round(sum(t["risk_score"] for t in TRANSACTIONS)/total if total else 0,1),
        "active_alerts":sum(1 for a in ALERTS if not a["dismissed"])})

@app.route("/api/model",methods=["GET"])
def model_info(): return jsonify(fraud_model.info())

@app.route("/api/model/retrain",methods=["POST"])
def retrain(): fraud_model.train(); return jsonify({"message":"Retrained.","info":fraud_model.info()})

@app.route("/api/stats/hourly",methods=["GET"])
def hourly():
    counts=[0]*24; fraud=[0]*24
    for t in TRANSACTIONS:
        h=t["hour_of_day"]; counts[h]+=1
        if t["risk_level"]=="high": fraud[h]+=1
    return jsonify({"hours":list(range(24)),"volumes":counts,
        "fraud_rates":[round(fraud[h]/counts[h]*100,1) if counts[h] else 0 for h in range(24)]})

if __name__ == "__main__":
    is_local = os.environ.get("RENDER") is None
    if is_local:
        import webbrowser
        if os.environ.get("WERKZEUG_RUN_MAIN") != "true":
            webbrowser.open("http://127.0.0.1:5000/dashboard")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=is_local)
