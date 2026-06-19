import sqlite3

db_path = r"c:\Users\a7med\Desktop\PROJECT1\backend 1\smart_pt.db"

def upgrade():
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    columns = [
        ("tutorial_video", "VARCHAR(256)"),
        ("instructions", "TEXT"),
        ("safety_notes", "TEXT"),
        ("difficulty_level", "VARCHAR(32)"),
        ("target_angle", "VARCHAR(64)")
    ]

    for col_name, col_type in columns:
        try:
            c.execute(f"ALTER TABLE exercises ADD COLUMN {col_name} {col_type}")
            print(f"Added {col_name}")
        except sqlite3.OperationalError as e:
            print(f"Skipped {col_name} (may already exist): {e}")

    conn.commit()
    conn.close()
    print("Migration complete.")

if __name__ == "__main__":
    upgrade()
