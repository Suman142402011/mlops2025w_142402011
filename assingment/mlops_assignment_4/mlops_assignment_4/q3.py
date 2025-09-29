# q3_crud_comparison.py

import sqlite3
import time
from pymongo import MongoClient
import pandas as pd

# ---------------- SQL (SQLite 2NF) ----------------
def sql_crud():
    conn = sqlite3.connect("online_retail.db")
    cur = conn.cursor()
    results = {}

    # INSERT
    start = time.perf_counter()
    cur.execute("INSERT INTO customers (customer_id, country) VALUES (?,?)", (999999, "TestLand"))
    conn.commit()
    results['insert'] = time.perf_counter() - start

    # READ
    start = time.perf_counter()
    cur.execute("SELECT * FROM customers WHERE customer_id=?", (999999,))
    _ = cur.fetchone()
    results['read'] = time.perf_counter() - start

    # UPDATE
    start = time.perf_counter()
    cur.execute("UPDATE customers SET country=? WHERE customer_id=?", ("UpdatedLand", 999999))
    conn.commit()
    results['update'] = time.perf_counter() - start

    # DELETE
    start = time.perf_counter()
    cur.execute("DELETE FROM customers WHERE customer_id=?", (999999,))
    conn.commit()
    results['delete'] = time.perf_counter() - start

    conn.close()
    return results

# ---------------- MongoDB Customer-centric ----------------
def mongo_customer_crud():
    client = MongoClient("mongodb://localhost:27017")
    db = client["onlineretail"]
    col = db["customers"]
    results = {}

    doc = {"customer_id": 999999, "country": "TestLand", "transactions": []}

    # INSERT
    start = time.perf_counter()
    col.insert_one(doc)
    results['insert'] = time.perf_counter() - start

    # READ
    start = time.perf_counter()
    _ = col.find_one({"customer_id": 999999})
    results['read'] = time.perf_counter() - start

    # UPDATE (append a transaction)
    start = time.perf_counter()
    col.update_one({"customer_id": 999999}, {"$push": {"transactions": {"invoice_no":"TEST1"}}})
    results['update'] = time.perf_counter() - start

    # DELETE
    start = time.perf_counter()
    col.delete_one({"customer_id": 999999})
    results['delete'] = time.perf_counter() - start

    return results

# ---------------- MongoDB Transaction-centric ----------------
def mongo_transaction_crud():
    client = MongoClient("mongodb://localhost:27017")
    db = client["onlineretail"]
    col = db["transactions"]
    results = {}

    doc = {"invoice_no": "TEST-INV", "customer_id": 999999, "items": []}

    # INSERT
    start = time.perf_counter()
    col.insert_one(doc)
    results['insert'] = time.perf_counter() - start

    # READ
    start = time.perf_counter()
    _ = col.find_one({"invoice_no": "TEST-INV"})
    results['read'] = time.perf_counter() - start

    # UPDATE
    start = time.perf_counter()
    col.update_one({"invoice_no": "TEST-INV"}, {"$set": {"items":[{"stock_code":"X001"}]}})
    results['update'] = time.perf_counter() - start

    # DELETE
    start = time.perf_counter()
    col.delete_one({"invoice_no": "TEST-INV"})
    results['delete'] = time.perf_counter() - start

    return results

# ---------------- Compare and Print ----------------
def print_comparison(sql_res, cust_res, txn_res):
    print("\nCRUD Performance Comparison (time in seconds):")
    print(f"{'Operation':<10} {'SQL 2NF':<12} {'Mongo Customer':<18} {'Mongo Transaction':<18}")
    for op in ['insert', 'read', 'update', 'delete']:
        print(f"{op:<10} {sql_res[op]:<12.6f} {cust_res[op]:<18.6f} {txn_res[op]:<18.6f}")

# ---------------- Main ----------------
if __name__ == "__main__":
    print("Running SQL CRUD...")
    sql_res = sql_crud()

    print("Running MongoDB Customer-centric CRUD...")
    cust_res = mongo_customer_crud()

    print("Running MongoDB Transaction-centric CRUD...")
    txn_res = mongo_transaction_crud()

    print_comparison(sql_res, cust_res, txn_res)
