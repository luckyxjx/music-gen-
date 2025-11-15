#!/usr/bin/env python3
"""
Simple script to test the API endpoints
"""

import requests
import json
import time

API_BASE = "http://localhost:5000"

def test_health():
    """Test health endpoint"""
    print("\n" + "="*60)
    print("Testing Health Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{API_BASE}/api/health")
        print(f"Status Code: {response.status_code}")
        print(f"Response: {json.dumps(response.json(), indent=2)}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_emotions():
    """Test emotions endpoint"""
    print("\n" + "="*60)
    print("Testing Emotions Endpoint")
    print("="*60)
    
    try:
        response = requests.get(f"{API_BASE}/api/emotions")
        print(f"Status Code: {response.status_code}")
        data = response.json()
        print(f"Available Emotions: {len(data['emotions'])}")
        for emotion in data['emotions']:
            print(f"  - {emotion['name']}: {emotion['description']}")
        return response.status_code == 200
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_generate_text():
    """Test text generation endpoint"""
    print("\n" + "="*60)
    print("Testing Text Generation Endpoint")
    print("="*60)
    
    payload = {
        "text": "I'm happy, give me an upbeat 2-minute track",
        "temperature": 1.0,
        "top_k": 20
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    print("\nGenerating music... (this may take 15-30 seconds)")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_BASE}/api/generate",
            json=payload,
            timeout=60
        )
        elapsed = time.time() - start_time
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Generation Time: {elapsed:.2f} seconds")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            print(f"\n✓ MIDI file available at: {API_BASE}{data['midi_file']}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_generate_emotion():
    """Test emotion generation endpoint"""
    print("\n" + "="*60)
    print("Testing Emotion Generation Endpoint")
    print("="*60)
    
    payload = {
        "emotion": "calm",
        "duration": 1.0,
        "temperature": 1.0,
        "top_k": 20
    }
    
    print(f"Request: {json.dumps(payload, indent=2)}")
    print("\nGenerating music... (this may take 15-30 seconds)")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_BASE}/api/generate-emotion",
            json=payload,
            timeout=60
        )
        elapsed = time.time() - start_time
        
        print(f"\nStatus Code: {response.status_code}")
        print(f"Generation Time: {elapsed:.2f} seconds")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response: {json.dumps(data, indent=2)}")
            print(f"\n✓ MIDI file available at: {API_BASE}{data['midi_file']}")
            return True
        else:
            print(f"Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("EMOPIA API Test Suite")
    print("="*60)
    print("\nMake sure the API server is running on http://localhost:5000")
    print("Start it with: python3 api.py")
    
    input("\nPress Enter to start tests...")
    
    results = {
        "Health Check": test_health(),
        "Emotions List": test_emotions(),
        "Text Generation": test_generate_text(),
        "Emotion Generation": test_generate_emotion()
    }
    
    print("\n" + "="*60)
    print("Test Results Summary")
    print("="*60)
    
    for test_name, passed in results.items():
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"{test_name}: {status}")
    
    total = len(results)
    passed = sum(results.values())
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! Your API is working correctly.")
    else:
        print("\n⚠️  Some tests failed. Check the output above for details.")

if __name__ == "__main__":
    main()
