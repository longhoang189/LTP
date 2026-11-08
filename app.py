import os
import pandas as pd
import streamlit as st
from src.model import load_or_train_model, predict_price, train_and_evaluate_model

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "cleaned_laptops.csv")
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "laptop_price_model.joblib")

st.set_page_config(
    page_title="Hệ Thống Dự Đoán Giá Laptop",
    layout="wide"
)

@st.cache_resource
def get_model():
    return load_or_train_model(MODEL_PATH)

@st.cache_data
def get_model_scores():
    result = train_and_evaluate_model()
    r2_lr, mae_lr, rmse_lr = result["linear_regression"]
    r2_rf, mae_rf, rmse_rf = result["random_forest"]
    return r2_lr, mae_lr, rmse_lr, r2_rf, mae_rf, rmse_rf

@st.cache_data
def get_data():
    if os.path.exists(DATA_FILE):
        return pd.read_csv(DATA_FILE)
    return pd.DataFrame()

# Tai du lieu va mo hinh da train san
df = get_data()
model, feature_cols = get_model()

st.title("HỆ THỐNG DỰ ĐOÁN GIÁ LAPTOP")

if df.empty:
    st.error("Chưa có tệp dữ liệu cleaned_laptops.csv. Vui lòng chạy lệnh 'python main.py' trước.")
    st.stop()

brands = sorted(df["brand"].dropna().unique().tolist())

tab_build, tab_products, tab_stats = st.tabs([
    "Tự Build Laptop & Dự Đoán Giá",
    "Danh Sách Sản Phẩm",
    "Thống Kê Tổng Quan"
])

# TAB 1: TU BUILD LAPTOP VA DU DOAN GIA
with tab_build:
    st.subheader("Tự build cấu hình Laptop và tính giá dự đoán")
    st.write("Vui lòng chọn các thông số cấu hình từ danh sách bên dưới:")

    col1, col2 = st.columns(2)

    with col1:
        selected_brand = st.selectbox(
            "1. Chọn thương hiệu (Brand):",
            options=brands,
            index=0
        )
        selected_ram = st.selectbox(
            "2. Chọn dung lượng RAM:",
            options=[8, 16, 24, 32, 64, 128],
            index=1,
            format_func=lambda x: f"{x} GB RAM"
        )

    with col2:
        selected_storage = st.selectbox(
            "3. Chọn dung lượng ổ cứng SSD:",
            options=[256, 512, 1024, 2048],
            index=1,
            format_func=lambda x: f"{x} GB ({x//1024} TB) SSD" if x >= 1024 else f"{x} GB SSD"
        )
        selected_screen = st.selectbox(
            "4. Chọn kích thước màn hình:",
            options=[13.3, 14.0, 15.3, 15.6, 16.0, 17.3],
            index=3,
            format_func=lambda x: f"{x} inch"
        )

    st.write("")
    if st.button("Dự Đoán Giá Laptop", type="primary"):
        predicted_price = predict_price(
            selected_brand, selected_ram, selected_storage, selected_screen, MODEL_PATH
        )

        st.divider()
        st.subheader("Kết quả dự đoán")
        col_res1, col_res2 = st.columns([1, 2])

        with col_res1:
            st.metric(
                label="Giá dự đoán (Mô hình Random Forest)",
                value=f"{predicted_price:,.0f} VNĐ",
                delta=f"{predicted_price / 1e6:.2f} Triệu VNĐ"
            )

        with col_res2:
            st.write("Thông số cấu hình đã chọn:")
            summary_df = pd.DataFrame([{
                "Thương hiệu": selected_brand,
                "Dung lượng RAM": f"{selected_ram} GB",
                "Ổ cứng SSD": f"{selected_storage} GB",
                "Kích thước màn hình": f"{selected_screen} inch"
            }])
            st.table(summary_df)

# TAB 2: DANH SACH SAN PHAM DA CAO
with tab_products:
    st.subheader(f"Danh sách sản phẩm hiện có: {len(df)} máy")

    col_filter1, col_filter2 = st.columns([1, 2])
    with col_filter1:
        filter_brand = st.selectbox(
            "Lọc theo thương hiệu:",
            options=["Tất cả"] + brands
        )
    with col_filter2:
        search_kw = st.text_input("Tìm kiếm theo tên máy:")

    # Ap dung bo loc
    view_df = df.copy()
    if filter_brand != "Tất cả":
        view_df = view_df[view_df["brand"] == filter_brand]
    if search_kw.strip():
        view_df = view_df[view_df["name"].str.contains(search_kw.strip(), case=False, na=False)]

    st.write(f"Hiển thị {len(view_df)} sản phẩm phù hợp:")

    display_cols = ["name", "brand", "cpu", "ram_gb", "storage_gb", "screen_size_inch", "price"]
    renamed_cols = {
        "name": "Tên sản phẩm",
        "brand": "Thương hiệu",
        "cpu": "Thông số CPU",
        "ram_gb": "RAM (GB)",
        "storage_gb": "Ổ cứng (GB)",
        "screen_size_inch": "Màn hình (inch)",
        "price": "Giá bán (VNĐ)"
    }

    table_data = view_df[display_cols].rename(columns=renamed_cols)
    table_data["Giá bán (VNĐ)"] = table_data["Giá bán (VNĐ)"].apply(lambda x: f"{x:,.0f}")

    st.dataframe(table_data, height=450)

# TAB 3: THONG KE TONG QUAN
with tab_stats:
    st.subheader("Thống kê tổng quan dữ liệu và mô hình")

    col_stat1, col_stat2, col_stat3 = st.columns(3)
    r2_lr, mae_lr, rmse_lr, r2_rf, mae_rf, rmse_rf = get_model_scores()

    with col_stat1:
        st.metric(label="Tổng số sản phẩm", value=len(df))
        st.metric(label="Số lượng thương hiệu", value=len(brands))
    with col_stat2:
        st.write("**Linear Regression**")
        st.metric(label="R2 Score", value=f"{r2_lr:.4f}")
        st.metric(label="MAE", value=f"{mae_lr:,.0f} VNĐ")
        st.metric(label="RMSE", value=f"{rmse_lr:,.0f} VNĐ")
    with col_stat3:
        st.write("**Random Forest**")
        st.metric(label="R2 Score", value=f"{r2_rf:.4f}")
        st.metric(label="MAE", value=f"{mae_rf:,.0f} VNĐ")
        st.metric(label="RMSE", value=f"{rmse_rf:,.0f} VNĐ")

    st.write("")
    st.write("Bảng thống kê giá bán theo thương hiệu:")
    stats_df = df.groupby("brand")["price"].agg(["count", "mean", "min", "max"]).reset_index()
    stats_df["mean"] = (stats_df["mean"] / 1e6).round(2)
    stats_df["min"] = (stats_df["min"] / 1e6).round(2)
    stats_df["max"] = (stats_df["max"] / 1e6).round(2)
    stats_df.columns = ["Thương hiệu", "Số lượng", "Giá trung bình (Triệu VNĐ)", "Giá thấp nhất (Triệu VNĐ)", "Giá cao nhất (Triệu VNĐ)"]
    st.dataframe(stats_df)
