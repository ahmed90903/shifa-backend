import sqlite3

db_path = r"c:\Users\a7med\Desktop\PROJECT1\backend 1\smart_pt.db"

def upgrade():
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    try:
        c.execute("""
            CREATE TABLE IF NOT EXISTS password_resets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                email VARCHAR(128) NOT NULL,
                reset_code VARCHAR(256) NOT NULL,
                reset_token VARCHAR(256) UNIQUE,
                expires_at DATETIME NOT NULL,
                is_used BOOLEAN DEFAULT 0,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY(user_id) REFERENCES users(id)
            )
        """)
        c.execute("CREATE INDEX IF NOT EXISTS ix_password_resets_email ON password_resets (email)")
        c.execute("CREATE INDEX IF NOT EXISTS ix_password_resets_id ON password_resets (id)")
        print("Created password_resets table")
    except sqlite3.OperationalError as e:
        print(f"Error: {e}")

    conn.commit()
    conn.close()
    print("Migration complete.")

if __name__ == "__main__":
    upgrade()
