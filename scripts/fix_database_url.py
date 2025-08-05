#!/usr/bin/env python3
"""
Helper script to properly format DATABASE_URL for Railway deployment
Handles special characters in passwords by URL encoding them
"""
import sys
import urllib.parse
from sqlalchemy.engine.url import make_url


def encode_database_url(url: str) -> str:
    """
    Properly encode a database URL, handling special characters in password
    
    Args:
        url: Raw database URL
        
    Returns:
        Properly encoded database URL
    """
    try:
        # Parse the URL
        parsed = make_url(url)
        
        # Check if password contains special characters
        if parsed.password:
            special_chars = ['@', ':', '/', '?', '#', '[', ']', '!', '$', '&', "'", '(', ')', '*', '+', ',', ';', '=', ' ']
            
            if any(char in parsed.password for char in special_chars):
                # URL encode the password
                encoded_password = urllib.parse.quote(parsed.password, safe='')
                parsed = parsed.set(password=encoded_password)
                print(f"✅ Password encoded successfully")
                print(f"   Original password length: {len(parsed.password)}")
                print(f"   Special characters found: {[c for c in special_chars if c in parsed.password]}")
            else:
                print("ℹ️  Password doesn't contain special characters, no encoding needed")
        
        return str(parsed)
        
    except Exception as e:
        print(f"❌ Error parsing DATABASE_URL: {e}")
        return url


def main():
    print("🔧 Railway Database URL Encoder\n")
    
    if len(sys.argv) > 1:
        # URL provided as argument
        url = sys.argv[1]
    else:
        # Interactive mode
        print("Enter your DATABASE_URL (it will be hidden):")
        print("Format: postgresql://user:password@host:port/database")
        url = input("> ").strip()
    
    if not url:
        print("❌ No URL provided")
        sys.exit(1)
    
    # Encode the URL
    encoded_url = encode_database_url(url)
    
    print("\n📋 Encoded DATABASE_URL for Railway:")
    print("-" * 50)
    print(encoded_url)
    print("-" * 50)
    
    print("\n📝 Instructions:")
    print("1. Copy the encoded URL above")
    print("2. Go to Railway dashboard → Variables")
    print("3. Set DATABASE_URL to the encoded value")
    print("4. Redeploy your service")
    
    # Test the connection
    print("\n🧪 Testing connection...")
    try:
        from sqlalchemy import create_engine
        engine = create_engine(encoded_url)
        with engine.connect() as conn:
            result = conn.execute("SELECT 1")
            print("✅ Connection successful!")
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        print("   Make sure your database is accessible from your current location")


if __name__ == "__main__":
    main()