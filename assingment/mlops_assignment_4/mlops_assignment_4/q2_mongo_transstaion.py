# mongo_transaction.py
"""
Transaction-centric model:
 - Collection: transactions
 - Each document = one invoice (invoice_no) with embedded items list.
 - Example doc:
   {
     "invoice_no": "536365",
     "invoice_date": ISODate(...),
     "customer_id": 17850,
     "country": "United Kingdom",
     "items": [
         {"stock_code":"85123A","description":"WHITE HANGING ...","quantity":6,"unit_price":2.55},
         ...
     ]
   }
"""

from pymongo import MongoClient, errors, ASCENDING
import pandas as pd
import time
from pymongo.errors import BulkWriteError, DuplicateKeyError

# ---------- CONFIG ----------
MONGO_URI = "mongodb://localhost:27017"  # replace with your Atlas URI if needed
DB_NAME = "onlineretail"
COLL_NAME = "transactions"

# Connection pooling settings
# - maxPoolSize controls maximum pooled connections
# - minPoolSize sets minimum
# - serverSelectionTimeoutMS fails fast if server unreachable
client = MongoClient(
    MONGO_URI,
    maxPoolSize=50,
    minPoolSize=5,
    serverSelectionTimeoutMS=5000,
    waitQueueTimeoutMS=2000
)

db = client[DB_NAME]
col = db[COLL_NAME]

def ensure_indexes():
    # unique index on invoice_no for fast lookup and uniqueness
    col.create_index([("invoice_no", ASCENDING)], unique=True)
    # index on customer_id for queries by customer
    col.create_index([("customer_id", ASCENDING)])
    print("Indexes ensured:", col.index_information())

def excel_to_transaction_docs(path, nrows=None, drop_na_customer=True):
    # read Excel, convert to grouped invoice documents
    df = pd.read_excel(path, nrows=nrows, engine='openpyxl', dtype={'StockCode': str})
    if drop_na_customer:
        df = df.dropna(subset=['CustomerID'])
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['CustomerID'] = df['CustomerID'].apply(lambda x: int(x) if pd.notna(x) else None)

    docs = []
    grouped = df.groupby('InvoiceNo', sort=False)
    for invoice_no, g in grouped:
        # if customerid missing skip (optional)
        try:
            cust = int(g['CustomerID'].dropna().iloc[0])
        except Exception:
            cust = None
        items = []
        for _, r in g.iterrows():
            items.append({
                "stock_code": str(r['StockCode']),
                "description": r.get('Description', None),
                "quantity": int(r['Quantity']) if pd.notna(r['Quantity']) else 0,
                "unit_price": float(r['UnitPrice']) if pd.notna(r['UnitPrice']) else None
            })
        doc = {
            "invoice_no": str(invoice_no),
            "invoice_date": g['InvoiceDate'].iloc[0].to_pydatetime(),
            "customer_id": cust,
            "country": g['Country'].iloc[0] if 'Country' in g else None,
            "items": items
        }
        docs.append(doc)
    return docs

def bulk_insert_transactions(docs, batch_size=1000):
    inserted = 0
    start = time.perf_counter()
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        try:
            col.insert_many(batch, ordered=False)  # unordered for speed
            inserted += len(batch)
            print(f"Inserted batch {i}..{i+len(batch)}")
        except BulkWriteError as bwe:
            # partial success possible; count inserted from result if needed
            print("BulkWriteError:", bwe.details.get('writeErrors', [])[:3])
        except Exception as e:
            print("Insert error:", e)
    elapsed = time.perf_counter() - start
    print(f"Inserted ~{inserted} docs in {elapsed:.2f}s ({inserted/elapsed:.1f} docs/s)")

# -------- CRUD examples ----------
def find_invoice(invoice_no):
    try:
        return col.find_one({"invoice_no": str(invoice_no)})
    except Exception as e:
        print("Find error:", e)

def update_item_quantity(invoice_no, stock_code, new_qty):
    try:
        res = col.update_one(
            {"invoice_no": str(invoice_no), "items.stock_code": str(stock_code)},
            {"$set": {"items.$.quantity": int(new_qty)}}
        )
        return res.modified_count
    except Exception as e:
        print("Update error:", e)

def delete_invoice(invoice_no):
    try:
        res = col.delete_one({"invoice_no": str(invoice_no)})
        return res.deleted_count
    except Exception as e:
        print("Delete error:", e)

# --------- example run ----------
if __name__ == "__main__":
    ensure_indexes()
    docs = excel_to_transaction_docs("Online Retail.xlsx", nrows=5000)  # change nrows if you want
    print("Prepared docs:", len(docs))
    # Insert first 3000 invoices for quick testing
    bulk_insert_transactions(docs[:3000], batch_size=500)

    # sample CRUD
    s = docs[0]['invoice_no']
    print("Sample invoice doc:", find_invoice(s))
    print("Update result:", update_item_quantity(s, docs[0]['items'][0]['stock_code'], 999))
    print("After update:", find_invoice(s))
    print("Delete result:", delete_invoice(s))
    print("Query after delete:", find_invoice(s))
