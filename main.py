import os
from src.crawler import scrape_laptop_data
from src.preprocessing import clean_data
from src.model import train_and_evaluate_model

def main():
    print("==================================================")
    print("  DU AN PYTHON: PHAN TICH VA DU DOAN GIA LAPTOP   ")
    print("==================================================")

    os.makedirs("data", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Buoc 1: Thu thap du lieu
    raw_data_path = "data/raw_laptops.csv"
    if not os.path.exists(raw_data_path):
        print("\n[+] Buoc 1: Thu thap du lieu tu web...")
        scrape_laptop_data(pages=15, page_size=50, output_file=raw_data_path)
    else:
        print(f"\n[+] Buoc 1: Su dung tep du lieu tho co san ({raw_data_path})")

    # Buoc 2: Trich xuat dac trung
    print("\n[+] Buoc 2: Trich xuat dac trung du lieu...")
    cleaned_data_path = "data/cleaned_laptops.csv"
    clean_data(input_file=raw_data_path, output_file=cleaned_data_path)

    # Buoc 3: Huan luyen va danh gia mo hinh
    print("\n[+] Buoc 3: Huan luyen mo hinh ML (Linear Regression & Random Forest)...")
    train_and_evaluate_model(data_file=cleaned_data_path)

    print("\n[+] Hoan thanh toan bo quy trinh! Kiem tra thu muc outputs/ de xem ket qua.")

if __name__ == "__main__":
    main()
