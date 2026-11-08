# Bai Tap Lon Python: Thu Thap Du Lieu Va Du Doan Gia Laptop

De tai: Thu thap du lieu laptop tu API CellphoneS, trich xuat dac trung va du doan gia ban bang mo hinh Linear Regression va Random Forest.

---

## 1. Cau Truc Thu Muc Du An

```
LTP/
├── data/
│   ├── raw_laptops.csv          # Du lieu tho cao tu API (688 san pham)
│   └── cleaned_laptops.csv      # Du lieu sau khi trich xuat dac trung (556 san pham)
│
├── src/
│   ├── __init__.py              # Khoi tao package
│   ├── crawler.py               # Mo-dun 1: Cào du lieu tu web
│   ├── preprocessing.py         # Mo-dun 2: Trich xuat dac trung (RAM, Storage, Screen, Price)
│   └── model.py                 # Mo-dun 3: Huan luyen va luu mo hinh train san
│
├── outputs/
│   ├── laptop_price_model.joblib # Artifact mo hinh Random Forest da train san
│   └── model_evaluation.txt     # Bao cao chi so R2, MAE, RMSE cua 2 mo hinh
│
├── app.py                       # Ung dung Web cuc bo giao dien Streamlit
├── main.py                      # Tep chay toan bo quy trinh du an
├── requirements.txt             # Danh sach thu vien can thiet
└── README.md                    # Huong dan su dung chi tiet
```

---

## 2. Cai Dat & Chay Chuong Trinh

### Cai dat thu vien:
```bash
pip install -r requirements.txt
```

### Chay toan bo quy trinh (Cào -> Trich xuat -> Huan luyen model):
```bash
python main.py
```

### Chay ung dung Web cuc bo (Streamlit):
```bash
streamlit run app.py
```
Giao dien web mo tren trinh duyet (dia chi: `http://localhost:8501`) voi cac chuc nang:
1. **Tu Build Laptop & Du Doan Gia:** Chon thong so Hang, RAM, O cung, Man hinh tu dropdown va nhan nut tinh gia du doan bang mo hinh Random Forest da train san.
2. **Danh Sach San Pham Da Cao:** Bang du lieu day du cac san pham da cao, ho tro loc theo hang va tim kiem ten san pham.
3. **Thong Ke Tong Quan:** Bang thong ke so luong, gia trung binh, min/max theo tung hang va chi so R2, MAE, RMSE cua ca 2 mo hinh.

---

## 3. Ket Qua Danh Gia Mo Hinh

| Mo Hinh | R2 Score | MAE (VND) | RMSE (VND) |
|---|---|---|---|
Linear Regression |0.8171 | 8,351,617 | 11,207,015
**Random Forest**     |**0.8722** | **6,730,736** | **9,369,102**

>
> - **R2**: ti le phuong sai giai thich duoc 
> - **MAE**: sai so tuyet doi trung binh — model thuong sai bao nhieu tien
> - **RMSE**: phat nang hon sai so lon — lon hon MAE nhieu la co du lieu bi du doan sai xa
