import sqlite3

db_path = r"c:\Users\a7med\Desktop\PROJECT1\backend 1\smart_pt.db"

def upgrade():
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    try:
        c.execute("ALTER TABLE users ADD COLUMN is_verified BOOLEAN DEFAULT 0")
        print("Added is_verified to users")
    except sqlite3.OperationalError as e:
        print(f"Skipped is_verified (may already exist): {e}")

    conn.commit()
    conn.close()
    print("Migration complete.")

if __name__ == "__main__":
    upgrade()
