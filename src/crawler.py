import os
import time
import requests
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "data", "raw_laptops.csv")

API_URL = "https://api.cellphones.com.vn/v2/graphql/query"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Content-Type": "application/json",
    "Origin": "https://cellphones.com.vn",
    "Referer": "https://cellphones.com.vn/"
}

def fetch_page(page, page_size=50, timeout=10):
    query_str = f"""
    query GetProductsByCateId {{
        products(
            filter: {{
                static: {{
                    categories: ["380"],
                    excluded: {{ categories: [] }},
                    province_id: 30,
                    stock: {{ from: 0 }},
                    company_stock_id: [46, 56, 152, 4920],
                    filter_price: {{ from: 0, to: 223000000 }}
                }},
                dynamic: {{}}
            }},
            page: {page},
            size: {page_size},
            sort: [{{ view: desc }}]
        ) {{
            general {{
                product_id
                name
                manufacturer
                attributes
            }},
            filterable {{
                price
                special_price
            }}
        }}
    }}
    """
    payload = {"query": query_str, "variables": {}}
    response = requests.post(API_URL, headers=HEADERS, json=payload, timeout=timeout)
    response.raise_for_status()
    data = response.json()
    return (data.get("data") or {}).get("products") or []

def parse_laptop(product):
    gen = product.get("general") or {}
    filt = product.get("filterable") or {}
    attrs = gen.get("attributes") or {}

    return {
        "product_id": gen.get("product_id"),
        "name": gen.get("name"),
        "brand": gen.get("manufacturer"),
        "price": filt.get("price"),
        "special_price": filt.get("special_price"),
        "cpu": attrs.get("cpu") or attrs.get("laptop_cpu"),
        "ram": attrs.get("laptop_ram"),
        "storage": attrs.get("hdd_sdd") or attrs.get("o_cung_laptop"),
        "screen_size": attrs.get("display_size") or attrs.get("laptop_screen_size_filter"),
        "resolution": attrs.get("display_resolution") or attrs.get("laptop_resolution_filter"),
        "gpu": attrs.get("vga") or attrs.get("laptop_vga_filter"),
        "weight": attrs.get("product_weight"),
        "battery": attrs.get("battery"),
        "os": attrs.get("os_version")
    }

def scrape_laptop_data(pages=15, page_size=50, output_file=DEFAULT_OUTPUT, delay=1.0):
    print("=== BAT DAU THU THAP DU LIEU LAPTOP ===")
    print(f"Cau hinh: Toi da {pages} trang | Kich thuoc trang: {page_size} san pham")

    all_laptops = []
    for page in range(1, pages + 1):
        print(f"[+] Dang cao trang {page}/{pages}...")
        try:
            products = fetch_page(page=page, page_size=page_size)
            if not products:
                print(f"[-] Trang {page} khong co san pham. Dung cao som.")
                break

            for prod in products:
                all_laptops.append(parse_laptop(prod))

            print(f"    Lay duoc {len(products)} san pham (Tong cong: {len(all_laptops)})")
        except Exception as e:
            print(f"[!] Loi tai trang {page}: {e}")
            break

        time.sleep(delay)

    if not all_laptops:
        print("[!] Khong thu thap duoc du lieu nao.")
        return pd.DataFrame()

    df = pd.DataFrame(all_laptops)
    df.drop_duplicates(subset=["product_id"], inplace=True)

    os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
    df.to_csv(output_file, index=False, encoding="utf-8-sig")

    print("=== HOAN THANH THU THAP DU LIEU ===")
    print(f"Tong so san pham duy nhat: {len(df)}")
    print(f"Tep ket qua da luu tai: {output_file}")
    return df

scrape_cellphones_laptops = scrape_laptop_data

if __name__ == "__main__":
    scrape_laptop_data()
