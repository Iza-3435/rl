import requests

def test_finnhub_connection(api_key):
    """Test your Finnhub API key"""
    print("🧪 Testing Finnhub connection...")
    
    try:
        url = "https://finnhub.io/api/v1/quote"
        params = {'symbol': 'AAPL', 'token': api_key}
        
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        if 'c' in data:
            print(f"✅ SUCCESS: AAPL price = ${data['c']:.2f}")
            print(f"   High: ${data.get('h', 0):.2f}")
            print(f"   Low: ${data.get('l', 0):.2f}")
            print(f"   Previous Close: ${data.get('pc', 0):.2f}")
            return True
        else:
            print(f"❌ ERROR: Invalid response: {data}")
            return False
    
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False

if __name__ == "__main__":
    API_KEY = "d253t4pr01qns40d1v00d253t4pr01qns40d1v0g"  # Replace with your key
    test_finnhub_connection(API_KEY)