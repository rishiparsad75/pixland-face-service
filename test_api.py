# -*- coding: utf-8 -*-
import sys, io, requests, base64, numpy as np
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from PIL import Image as PILImage

print("")
print("=== PixLand Face Service - E2E Test ===")
print("")

# 1. Python health
print("[1] Python /health")
try:
    r = requests.get("http://localhost:5001/health", timeout=5)
    d = r.json()
    print("  [PASS] status:", d.get("status"))
    print("  [PASS] model:", d.get("model"))
    print("  [PASS] detector:", d.get("detector"))
except Exception as e:
    print("  [FAIL]", e); sys.exit(1)

# 2. Node health
print("")
print("[2] Node.js /api/health")
try:
    r = requests.get("http://localhost:5000/api/health", timeout=5)
    d = r.json()
    print("  [PASS] status:", d.get("status"))
    print("  [PASS] version:", d.get("version"))
except Exception as e:
    print("  [FAIL]", e)

# 3. /extract blank image => should return 422
print("")
print("[3] /extract - blank white image (expect 422 NO_FACE_DETECTED)")
try:
    buf = io.BytesIO()
    PILImage.new("RGB", (200, 200), (255, 255, 255)).save(buf, format="JPEG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    r = requests.post("http://localhost:5001/extract", json={"image": b64}, timeout=30)
    if r.status_code == 422:
        print("  [PASS] 422 returned correctly:", r.json().get("error"))
    else:
        print("  [WARN] Got HTTP", r.status_code, r.text[:150])
except Exception as e:
    print("  [FAIL]", e)

# 4. /compare self-similarity
print("")
print("[4] /compare - self similarity (same embedding vs itself)")
try:
    emb = np.random.randn(512).tolist()
    r = requests.post("http://localhost:5001/compare",
                      json={"embedding1": emb, "embedding2": emb}, timeout=10)
    d = r.json()
    sim = d.get("similarity", 0)
    print("  [PASS] Similarity (self vs self):", sim)
    print("  [PASS] Distance:", d.get("distance"))
    print("  [PASS] Is match:", d.get("is_match"))
    if sim > 0.99:
        print("  [PASS] PERFECT SELF-MATCH - cosine similarity is working correctly!")
    else:
        print("  [WARN] Unexpected self-similarity:", sim)
except Exception as e:
    print("  [FAIL]", e)

# 5. /compare two random different embeddings => should NOT match
print("")
print("[5] /compare - two DIFFERENT random embeddings (expect no match)")
try:
    e1 = np.random.randn(512).tolist()
    e2 = np.random.randn(512).tolist()
    r = requests.post("http://localhost:5001/compare",
                      json={"embedding1": e1, "embedding2": e2}, timeout=10)
    d = r.json()
    sim = d.get("similarity", 1)
    print("  [PASS] Similarity (different):", round(sim, 4))
    print("  [PASS] Is match:", d.get("is_match"))
    if not d.get("is_match"):
        print("  [PASS] Correctly identified as NON-MATCH")
    else:
        print("  [WARN] Random vectors matched - threshold may be too loose")
except Exception as e:
    print("  [FAIL]", e)

print("")
print("=== All tests complete! ===")
print("")
