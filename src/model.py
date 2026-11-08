import os
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(BASE_DIR, "data", "cleaned_laptops.csv")
DEFAULT_REPORT = os.path.join(BASE_DIR, "outputs", "model_evaluation.txt")
DEFAULT_MODEL_PATH = os.path.join(BASE_DIR, "outputs", "laptop_price_model.joblib")

def train_and_evaluate_model(data_file=DEFAULT_INPUT, output_report=DEFAULT_REPORT, model_path=DEFAULT_MODEL_PATH):
    print("=== HUAN LUYEN VA DANH GIA MO HINH ===")
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Khong tim thay tep: {data_file}")

    df = pd.read_csv(data_file)

    # Chon cac dac trung so hoc va thuong hieu
    features = ["brand", "ram_gb", "storage_gb", "screen_size_inch"]
    target = "price"

    X = df[features]
    y = df[target]

    # One-Hot Encoding cho bien brand
    X_encoded = pd.get_dummies(X, columns=["brand"], drop_first=True, dtype=int)

    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42
    )

    # 1. Mo hinh Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)

    r2_lr = r2_score(y_test, y_pred_lr)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    rmse_lr = mean_squared_error(y_test, y_pred_lr) ** 0.5

    # 2. Mo hinh Random Forest Regressor
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    r2_rf = r2_score(y_test, y_pred_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = mean_squared_error(y_test, y_pred_rf) ** 0.5

    # In ket qua ra console
    print(f"Linear Regression - R2: {r2_lr:.4f} | MAE: {mae_lr:,.0f} | RMSE: {rmse_lr:,.0f}")
    print(f"Random Forest     - R2: {r2_rf:.4f} | MAE: {mae_rf:,.0f} | RMSE: {rmse_rf:,.0f}")

    # Luu artifact mo hinh da train san vao outputs/
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    joblib.dump({"model": rf, "feature_columns": list(X_encoded.columns)}, model_path)
    print(f"[+] Da luu mo hinh train san tai: {model_path}")

    # Ghi file bao cao danh gia
    os.makedirs(os.path.dirname(output_report) or ".", exist_ok=True)
    with open(output_report, "w", encoding="utf-8") as f:
        f.write("KET QUA DANH GIA MO HINH DU DOAN GIA LAPTOP\n")
        f.write("=" * 45 + "\n\n")
        f.write("1. Linear Regression:\n")
        f.write(f"   - R2   : {r2_lr:.4f}\n")
        f.write(f"   - MAE  : {mae_lr:,.0f}\n")
        f.write(f"   - RMSE : {rmse_lr:,.0f}\n\n")
        f.write("2. Random Forest Regressor:\n")
        f.write(f"   - R2   : {r2_rf:.4f}\n")
        f.write(f"   - MAE  : {mae_rf:,.0f}\n")
        f.write(f"   - RMSE : {rmse_rf:,.0f}\n\n")
        better = "Random Forest" if r2_rf > r2_lr else "Linear Regression"
        f.write(f"Ket luan: Mo hinh {better} cho ket qua tot hon.\n")

    print(f"[+] Da luu ket qua danh gia vao: {output_report}")
    return {
        "linear_regression": (r2_lr, mae_lr, rmse_lr),
        "random_forest": (r2_rf, mae_rf, rmse_rf)
    }

def load_or_train_model(model_path=DEFAULT_MODEL_PATH):
    if os.path.exists(model_path):
        artifact = joblib.load(model_path)
        return artifact["model"], artifact["feature_columns"]
    train_and_evaluate_model()
    artifact = joblib.load(model_path)
    return artifact["model"], artifact["feature_columns"]

def predict_price(brand, ram, storage, screen, model_path=DEFAULT_MODEL_PATH):
    model, feature_cols = load_or_train_model(model_path)
    row = pd.DataFrame([{
        "brand": str(brand).strip(),
        "ram_gb": int(ram),
        "storage_gb": int(storage),
        "screen_size_inch": float(screen)
    }])
    row_enc = pd.get_dummies(row, columns=["brand"], dtype=int)
    for col in feature_cols:
        if col not in row_enc.columns:
            row_enc[col] = 0
    row_enc = row_enc[feature_cols]
    return float(model.predict(row_enc)[0])

if __name__ == "__main__":
    train_and_evaluate_model()
