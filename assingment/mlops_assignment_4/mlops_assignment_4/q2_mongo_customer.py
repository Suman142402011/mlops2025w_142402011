# mongo_customer.py
"""
Customer-centric model:
 - Collection: customers
 - Each document = one customer, with an array of transactions (embedded),
   OR a documents with reference to separate transactions collection (alternative).
 - This script embeds a limited number of transactions per customer.
"""

from pymongo import MongoClient, errors, ASCENDING
import pandas as pd
import time
from collections import defaultdict
from pymongo.errors import BulkWriteError

# ---------- CONFIG ----------
MONGO_URI = "mongodb://localhost:27017"  # replace with your Atlas URI
DB_NAME = "onlineretail"
COLL_NAME = "customers"

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
    col.create_index([("customer_id", ASCENDING)], unique=True)
    # You can also create index on "transactions.invoice_no" if you need fast look-up inside embedded array
    col.create_index([("transactions.invoice_no", ASCENDING)])
    print("Indexes ensured:", col.index_information())

def excel_to_customers_docs(path, nrows=None, drop_na_customer=True):
    df = pd.read_excel(path, nrows=nrows, engine='openpyxl', dtype={'StockCode': str})
    if drop_na_customer:
        df = df.dropna(subset=['CustomerID'])
    df['InvoiceDate'] = pd.to_datetime(df['InvoiceDate'])
    df['CustomerID'] = df['CustomerID'].astype(int)

    # group by customer -> invoice -> items
    cust_map = defaultdict(lambda: {"customer_id": None, "country": None, "transactions": []})
    grouped = df.groupby(['CustomerID','InvoiceNo'], sort=False)
    for (cust_id, inv_no), g in grouped:
        items = []
        for _, r in g.iterrows():
            items.append({
                "stock_code": str(r['StockCode']),
                "description": r.get('Description', None),
                "quantity": int(r['Quantity']) if pd.notna(r['Quantity']) else 0,
                "unit_price": float(r['UnitPrice']) if pd.notna(r['UnitPrice']) else None
            })
        trans = {
            "invoice_no": str(inv_no),
            "invoice_date": g['InvoiceDate'].iloc[0].to_pydatetime(),
            "items": items
        }
        cid = int(cust_id)
        cust_map[cid]['customer_id'] = cid
        cust_map[cid]['country'] = g['Country'].iloc[0] if 'Country' in g else None
        cust_map[cid]['transactions'].append(trans)

    docs = list(cust_map.values())
    return docs

def bulk_insert_customers(docs, batch_size=200):
    start = time.perf_counter()
    inserted = 0
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        try:
            col.insert_many(batch, ordered=False)
            inserted += len(batch)
            print(f"Inserted customers {i}..{i+len(batch)}")
        except BulkWriteError as bwe:
            print("BulkWriteError:", bwe.details.get('writeErrors', [])[:3])
        except Exception as e:
            print("Insert error:", e)
    elapsed = time.perf_counter() - start
    print(f"Inserted ~{inserted} customer docs in {elapsed:.2f}s")

# CRUD examples
def find_customer(customer_id):
    try:
        return col.find_one({"customer_id": int(customer_id)})
    except Exception as e:
        print("Find error:", e)

def add_transaction_to_customer(customer_id, transaction):
    """
    Append a transaction to an existing customer document.
    If customer doesn't exist, you might want to upsert.
    """
    try:
        res = col.update_one(
            {"customer_id": int(customer_id)},
            {"$push": {"transactions": transaction}},
            upsert=False
        )
        return res.modified_count
    except Exception as e:
        print("Update error:", e)

def create_or_upsert_customer(customer_doc):
    """
    Upsert a full customer doc (create or replace).
    """
    try:
        res = col.update_one(
            {"customer_id": int(customer_doc['customer_id'])},
            {"$set": customer_doc},
            upsert=True
        )
        return res.upserted_id or res.matched_count
    except Exception as e:
        print("Upsert error:", e)

def delete_customer(customer_id):
    try:
        res = col.delete_one({"customer_id": int(customer_id)})
        return res.deleted_count
    except Exception as e:
        print("Delete error:", e)

# ---------- example run ----------
if __name__ == "__main__":
    ensure_indexes()
    docs = excel_to_customers_docs("Online Retail.xlsx", nrows=5000)  # nrows for speed in testing
    print("Prepared customer docs:", len(docs))
    bulk_insert_customers(docs[:2000], batch_size=200)  # insert first 2000 customers

    # sample CRUD
    sample_cid = docs[0]['customer_id']
    print("Customer doc sample:", find_customer(sample_cid))

    # append a fake transaction to sample customer
    fake_trans = {
        "invoice_no": "TEST-0001",
        "invoice_date": pd.Timestamp.now().to_pydatetime(),
        "items": [{"stock_code": "X001", "description": "TEST", "quantity": 1, "unit_price": 9.99}]
    }
    print("Append result:", add_transaction_to_customer(sample_cid, fake_trans))
    print("After append:", find_customer(sample_cid))
