import requests
import os

def test_api():
    print("Testing Resume Screening API...")
    base_url = "http://127.0.0.1:5000"
    
    # 1. Check if server is up
    try:
        # We don't have a GET / but we can try to POST /upload
        print("Uploading test file...")
        
        # Create a dummy PDF if not exists
        test_pdf = "test_resume.pdf"
        with open(test_pdf, "wb") as f:
            f.write(b"%PDF-1.1\n1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj\n2 0 obj<</Type/Pages/Count 1/Kids[3 0 R]>>endobj\n3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R>>endobj\n4 0 obj<</Length 51>>stream\nBT /F1 12 Tf 100 700 Td (Python Machine Learning SQL) Tj ET\nendstream\nendobj\nxref\n0 5\n0000000000 65535 f\n0000000009 00000 n\n0000000052 00000 n\n0000000101 00000 n\n0000000178 00000 n\ntrailer<</Size 5/Root 1 0 R>>\nstartxref\n278\n%%EOF")
            
        with open(test_pdf, 'rb') as f:
            files = {'file': f}
            r = requests.post(f"{base_url}/upload", files=files)
            
        if r.status_code == 200:
            upload_data = r.json()
            print(f"Upload Success: {upload_data}")
            
            # 2. Test Prediction
            print("Requesting prediction...")
            r_predict = requests.post(f"{base_url}/predict", json={"path": upload_data['path']})
            
            if r_predict.status_code == 200:
                result = r_predict.json()
                print(f"Prediction Success!")
                print(f"Role: {result['role']}")
                print(f"Score: {result['match_score']}")
                print(f"Skills: {result['skills']}")
                return True
            else:
                print(f"Prediction Failed: {r_predict.text}")
        else:
            print(f"Upload Failed: {r.text}")
            
    except Exception as e:
        print(f"Error: {e}")
        
    return False

if __name__ == "__main__":
    if test_api():
        print("\nSUMMARY: Backend is fully functional!")
    else:
        print("\nSUMMARY: Verification failed.")
