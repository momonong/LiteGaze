import urllib.request
import json
import time

def test_endpoint(url, method="GET", data=None):
    print(f"Testing {method} {url}...")
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(url, method=method, headers=headers)
    if data is not None:
        req.data = json.dumps(data).encode("utf-8")
    
    try:
        with urllib.request.urlopen(req, timeout=15) as res:
            resp_data = json.loads(res.read().decode("utf-8"))
            print(f"Success! Keys in response: {list(resp_data.keys())}")
            return resp_data
    except Exception as e:
        print(f"Error: {e}")
        return None

def main():
    time.sleep(2) # Wait for server to be fully ready
    # 1. Test cognitive health
    cog_health = test_endpoint("http://localhost:8080/api/cognitive/health")
    
    # 2. Test fusion health
    fuse_health = test_endpoint("http://localhost:8080/api/fuse/health")
    
    # 3. Test cognitive analyze text (English with GPT-2)
    cog_text_en = test_endpoint(
        "http://localhost:8080/api/cognitive/analyze/text",
        method="POST",
        data={
            "text": "The quick brown fox jumps over the lazy dog.",
            "lang": "en",
            "domain": "auto"
        }
    )
    
    # 4. Test cognitive analyze text (Chinese with BERT)
    cog_text_zh = test_endpoint(
        "http://localhost:8080/api/cognitive/analyze/text",
        method="POST",
        data={
            "text": "胡衛東是中國江蘇隊的籃球國手。",
            "lang": "zh",
            "domain": "auto"
        }
    )
    
    if cog_text_en:
        # 5. Test fusion endpoint
        gaze_events = [
            {
                "word": "quick",
                "confidence": "high",
                "dwell_count": 3,
                "fixation_count": 1,
                "timestamp_ms": 1749912000000
            },
            {
                "word": "brown",
                "confidence": "high",
                "dwell_count": 4,
                "fixation_count": 1,
                "timestamp_ms": 1749912000100
            }
        ]
        
        fuse_res = test_endpoint(
            "http://localhost:8080/api/fuse/",
            method="POST",
            data={
                "session_id": "test_session_123",
                "persist": True,
                "cognitive_result": cog_text_en,
                "gaze_events": gaze_events
            }
        )

if __name__ == "__main__":
    main()
