"""
Test script for FastAPI inference server
"""
import requests
import sys

# Server URL
BASE_URL = "http://localhost:6336"


def test_health():
    """Test health endpoint"""
    print("Testing health endpoint...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}\n")


def test_predict_upload(video_path: str):
    """Test prediction with file upload"""
    print(f"Uploading video: {video_path}")
    
    with open(video_path, "rb") as f:
        files = {"file": (video_path, f, "video/mp4")}
        response = requests.post(f"{BASE_URL}/predict", files=files)
    
    print(f"Status: {response.status_code}")
    result = response.json()
    
    if result.get("success"):
        print(f"Success!")
        print(f"Prediction: {result['prediction']}")
        print(f"Frames processed: {result['frames_processed']}\n")
    else:
        print(f"Error: {result.get('error')}\n")
    
    return result


def test_predict_url(video_url: str):
    """Test prediction with video URL"""
    print(f"Predicting from URL: {video_url}")
    
    response = requests.post(
        f"{BASE_URL}/predict_url",
        params={"video_url": video_url}
    )
    
    print(f"Status: {response.status_code}")
    result = response.json()
    
    if result.get("success"):
        print(f"Success!")
        print(f"Prediction: {result['prediction']}")
        print(f"Frames processed: {result['frames_processed']}\n")
    else:
        print(f"Error: {result.get('error')}\n")
    
    return result


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Uni-Sign API Server")
    print("=" * 60 + "\n")
    
    # Test health
    test_health()
    
    # Test with local video file
    if len(sys.argv) > 1:
        video_path = sys.argv[1]
        test_predict_upload(video_path)
    else:
        print("No video path provided")
        print("Usage: python test_api.py <video_path>")
        print("\nExample:")
        print("  python test_api.py E:/yes/IEC/cosign/cosign-ai/Uni-Sign/test/BG1_S001.mp4")
