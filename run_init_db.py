import asyncio
from backend.models.db_engine import init_db

async def main():
    print("Creating new tables...")
    await init_db()
    print("Done.")

if __name__ == "__main__":
    asyncio.run(main())
