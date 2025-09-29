import pandas as pd
import sqlite3

# ---------------- Load dataset ----------------
df = pd.read_excel("Online Retail.xlsx")

# ---------------- Preprocessing ----------------
print(df.isnull().sum())

# drop rows without CustomerID
df = df.dropna(subset=['CustomerID'])
df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
df['CustomerID'] = df['CustomerID'].astype(int)

# ---------------- Connect with sqlite ----------------
conn = sqlite3.connect("online_retail.db")
cur = conn.cursor()

# ---------------- Create normalized tables (2NF) ----------------
cur.execute("""
CREATE TABLE IF NOT EXISTS customers (
    customer_id INTEGER PRIMARY KEY,
    country TEXT
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS products (
    stock_code TEXT PRIMARY KEY,
    description TEXT
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS invoices (
    invoice_no TEXT PRIMARY KEY,
    invoice_date TIMESTAMP,
    customer_id INTEGER,
    country TEXT,
    FOREIGN KEY (customer_id) REFERENCES customers(customer_id)
);
""")

cur.execute("""
CREATE TABLE IF NOT EXISTS invoice_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    invoice_no TEXT,
    stock_code TEXT,
    quantity INTEGER,
    unit_price REAL,
    FOREIGN KEY (invoice_no) REFERENCES invoices(invoice_no),
    FOREIGN KEY (stock_code) REFERENCES products(stock_code)
);
""")

# ---------------- Insert at least 1000 records ----------------

# customers (remove duplicates to avoid UNIQUE constraint error)
customers = (
    df[['CustomerID','Country']]
    .drop_duplicates(subset=['CustomerID'])
    .rename(columns={'CustomerID': 'customer_id', 'Country': 'country'})
)
customers.head(1000).to_sql("customers", conn, if_exists="append", index=False)

# products
products = (
    df[['StockCode','Description']]
    .drop_duplicates(subset=['StockCode'])
    .rename(columns={'StockCode': 'stock_code', 'Description': 'description'})
)
products.head(1000).to_sql("products", conn, if_exists="append", index=False)

# invoices
invoices = (
    df[['InvoiceNo','InvoiceDate','CustomerID','Country']]
    .drop_duplicates(subset=['InvoiceNo'])
    .rename(columns={
        'InvoiceNo': 'invoice_no',
        'InvoiceDate': 'invoice_date',
        'CustomerID': 'customer_id',
        'Country': 'country'
    })
)
invoices.head(1000).to_sql("invoices", conn, if_exists="append", index=False)

# invoice items (no PK conflict since `id` is autoincrement)
items = df[['InvoiceNo','StockCode','Quantity','UnitPrice']].rename(
    columns={
        'InvoiceNo': 'invoice_no',
        'StockCode': 'stock_code',
        'Quantity': 'quantity',
        'UnitPrice': 'unit_price'
    }
)
items.head(1000).to_sql("invoice_items", conn, if_exists="append", index=False)

# ---------------- Verify counts ----------------
cur.execute("SELECT COUNT(*) FROM customers")
print("Customers inserted:", cur.fetchone()[0])

cur.execute("SELECT COUNT(*) FROM products")
print("Products inserted:", cur.fetchone()[0])

cur.execute("SELECT COUNT(*) FROM invoices")
print("Invoices inserted:", cur.fetchone()[0])

cur.execute("SELECT COUNT(*) FROM invoice_items")
print("Invoice items inserted:", cur.fetchone()[0])

# commit and close
conn.commit()
conn.close()
