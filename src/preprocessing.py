import os
import re
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_INPUT = os.path.join(BASE_DIR, "data", "raw_laptops.csv")
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "data", "cleaned_laptops.csv")

def clean_text(val):
    if pd.isna(val):
        return ""
    # Xoa the HTML va khoang trang thua
    text = re.sub(r"<[^>]+>", " ", str(val))
    return re.sub(r"\s+", " ", text).strip()

def clean_price(val):
    if pd.isna(val):
        return None
    if isinstance(val, (int, float)):
        return float(val) if val > 0 else None
    cleaned = re.sub(r"[^\d]", "", str(val))
    return float(cleaned) if cleaned else None

def extract_ram(val):
    match = re.search(r"(\d+)\s*GB", str(val), re.IGNORECASE)
    if not match:
        match = re.search(r"(\d+)", str(val))
    return int(match.group(1)) if match else None

def extract_storage(val):
    s = str(val).upper()
    match_tb = re.search(r"(\d+)\s*TB", s)
    if match_tb:
        return int(match_tb.group(1)) * 1024
    match_gb = re.search(r"(\d+)\s*GB", s)
    if match_gb:
        return int(match_gb.group(1))
    return None

def extract_screen(val):
    match = re.search(r"(\d{1,2}(?:\.\d{1,2})?)", str(val))
    return float(match.group(1)) if match else None

def clean_data(input_file=DEFAULT_INPUT, output_file=DEFAULT_OUTPUT):
    print("=== TIEN XU LY: SUA CHINH TA VA TRICH XUAT DAC TRUNG ===")
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Khong tim thay tep: {input_file}")

    df = pd.read_csv(input_file)
    print(f"Tong so ban ghi ban dau: {len(df)}")

    # Lay gia ban thuc te
    price_sp = df["special_price"].apply(clean_price)
    price_reg = df["price"].apply(clean_price)
    price_clean = price_sp.combine_first(price_reg)

    valid_mask = price_clean.notna() & (price_clean > 0)
    df = df[valid_mask].copy()
    df["price"] = price_clean[valid_mask].astype(float)
    print(f"So luong co gia hop le: {len(df)}")

    # Sua loi chinh ta, xoa khoang trang thua va ky tu HTML
    df["name"] = df["name"].apply(clean_text)
    df["brand"] = df["brand"].apply(clean_text)
    df["cpu"] = df["cpu"].apply(clean_text)

    # Trich xuat cac dac trung so hoc
    df["ram_gb"] = df["ram"].apply(extract_ram)
    df["storage_gb"] = df["storage"].apply(extract_storage)
    df["screen_size_inch"] = df["screen_size"].apply(extract_screen)

    # Dien gia tri thieu bang median
    df["ram_gb"] = df["ram_gb"].fillna(df["ram_gb"].median()).astype(int)
    df["storage_gb"] = df["storage_gb"].fillna(df["storage_gb"].median()).astype(int)
    df["screen_size_inch"] = df["screen_size_inch"].fillna(df["screen_size_inch"].median()).round(1)

    # Giu nguyen cac cot goc va them cot dac trung da trich xuat
    result_df = df[["product_id", "name","brand", "cpu", "ram_gb", "storage_gb","screen_size_inch", "price"]].copy()

    result_df.drop_duplicates(subset=["product_id"], inplace=True)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    result_df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print(f"So luong ban ghi da lam sach: {len(result_df)}")
    print(f"Da luu tep cleaned tai: {output_file}")
    return result_df

if __name__ == "__main__":
    clean_data()
