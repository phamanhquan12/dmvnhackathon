#!/usr/bin/env python3
"""Script to update admin password with proper bcrypt hash."""
import asyncio
import bcrypt
from sqlalchemy import text
from app.core.database import get_async_session

async def main():
    # Generate proper bcrypt hash
    password = "admin123"
    hash_bytes = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt())
    hash_str = hash_bytes.decode('utf-8')
    
    print(f"Generated hash: {hash_str}")
    
    # Update database
    session_maker = get_async_session()
    async with session_maker() as session:
        # Use raw SQL to update
        result = await session.execute(
            text("UPDATE users SET password_hash = :hash WHERE employee_id = :emp_id"),
            {"hash": hash_str, "emp_id": "S00058"}
        )
        await session.commit()
        print(f"Updated {result.rowcount} row(s)")
        
        # Verify
        verify = await session.execute(
            text("SELECT employee_id, password_hash FROM users WHERE employee_id = :emp_id"),
            {"emp_id": "S00058"}
        )
        row = verify.fetchone()
        if row:
            print(f"Verified: {row[0]} has hash: {row[1][:30]}...")
            
            # Test the password
            stored_hash = row[1]
            if bcrypt.checkpw(password.encode('utf-8'), stored_hash.encode('utf-8')):
                print("✅ Password verification successful!")
            else:
                print("❌ Password verification failed!")

if __name__ == "__main__":
    asyncio.run(main())
