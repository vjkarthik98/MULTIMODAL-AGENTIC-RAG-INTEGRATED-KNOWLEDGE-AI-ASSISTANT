"""
Seed a pre-activated test user directly into MongoDB.
Run once:  python scripts/seed_test_user.py
"""
import os, sys, uuid
from datetime import datetime, timezone
from pathlib import Path

# ensure project root is on sys.path so config loads
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from passlib.context import CryptContext
from pymongo import MongoClient

TEST_EMAIL    = "testuser@ragdev.local"
TEST_PASSWORD = "RagTest@2026!"

_pwd_ctx = CryptContext(schemes=["argon2", "bcrypt"], deprecated="auto")

def main():
    mongo_uri = os.environ.get("MONGO_URI") or os.environ.get("MONGODB_URI")
    if not mongo_uri:
        sys.exit("ERROR: MONGO_URI env var not set.")

    db_name    = os.environ.get("MONGO_DB_NAME", "rag_memory")
    collection = os.environ.get("AUTH_COLLECTION", "users")

    client = MongoClient(mongo_uri)
    col    = client[db_name][collection]

    existing = col.find_one({"email": TEST_EMAIL})
    if existing:
        print(f"Test user already exists (user_id={existing['user_id']}). Nothing to do.")
        return

    user_id = str(uuid.uuid4())
    doc = {
        "user_id":         user_id,
        "email":           TEST_EMAIL,
        "hashed_password": _pwd_ctx.hash(TEST_PASSWORD),
        "auth_providers":  ["email"],
        "is_active":       True,          # skip OTP — dev-only seed
        "created_at":      datetime.now(timezone.utc),
        "last_login":      None,
    }
    col.insert_one(doc)
    print(f"Test user created.")
    print(f"  email   : {TEST_EMAIL}")
    print(f"  password: {TEST_PASSWORD}")
    print(f"  user_id : {user_id}")

if __name__ == "__main__":
    main()
