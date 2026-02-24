# -*- coding: utf-8 -*-
"""
Step 1: Warm up ArcFace model by sending a real face image.
Uses thispersondoesnotexist.com OR a fallback synthetic face image.
"""
import sys, io, requests, base64, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

from PIL import Image as PILImage, ImageDraw
import numpy as np

BASE = "http://localhost:5001"

print("")
print("=== ArcFace Warm-Up + Real Face Test ===")
print("(First run loads model weights - may take 30-60s)")
print("")

# Try to get a real face from thispersondoesnotexist.com
print("[1] Fetching a GAN-generated face image...")
img_bytes = None

try:
    resp = requests.get(
        "https://thispersondoesnotexist.com",
        timeout=15,
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    )
    img_bytes = resp.content
    print("  [OK] Downloaded", len(img_bytes), "bytes from thispersondoesnotexist.com")
except Exception as e:
    print("  [WARN] Could not download:", e)
    print("  [INFO] Creating synthetic face-like image as fallback...")

    # Create a simple synthetic face-like image (oval + features)
    img = PILImage.new("RGB", (400, 400), (240, 200, 160))
    draw = ImageDraw.Draw(img)
    # Head oval
    draw.ellipse([80, 60, 320, 340], fill=(220, 175, 140))
    # Eyes
    draw.ellipse([130, 140, 170, 175], fill=(50, 30, 20))
    draw.ellipse([230, 140, 270, 175], fill=(50, 30, 20))
    # Nose
    draw.polygon([(200, 190), (185, 230), (215, 230)], fill=(190, 140, 110))
    # Mouth
    draw.arc([160, 240, 240, 280], 0, 180, fill=(140, 80, 80), width=4)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    img_bytes = buf.getvalue()
    print("  [OK] Created synthetic face image:", len(img_bytes), "bytes")

b64 = base64.b64encode(img_bytes).decode("utf-8")

# Send to /extract
print("")
print("[2] Sending to /extract (ArcFace + RetinaFace)...")
print("    This may take 30-90s on first run while model loads...")
start = time.time()

try:
    r = requests.post(BASE + "/extract", json={"image": b64}, timeout=120)
    elapsed = time.time() - start
    print(f"    Response time: {elapsed:.1f}s")

    if r.status_code == 200:
        d = r.json()
        emb = d.get("embedding", [])
        print("")
        print("  [PASS] Embedding extracted successfully!")
        print("  [PASS] Dimensions:", d.get("embedding_dim"))
        print("  [PASS] Face count:", d.get("face_count"))
        print("  [PASS] Model:", d.get("model"))
        face_area = d.get("face_area", {})
        print("  [PASS] Face area:", face_area)
        print("  [PASS] First 5 embedding values:", [round(x, 4) for x in emb[:5]])

        # Save embedding for compare test
        with open("test_embedding.txt", "w") as f:
            import json
            json.dump(emb, f)
        print("")
        print("  [INFO] Embedding saved to test_embedding.txt")

    elif r.status_code == 422:
        print("  [INFO] No face detected (422) -", r.json().get("error"))
        print("  [INFO] This is expected if using synthetic fallback image")
        print("  [INFO] Model IS loaded - subsequent real images will work fast!")

    else:
        print("  [FAIL] HTTP", r.status_code, r.text[:300])

except requests.exceptions.Timeout:
    print("  [FAIL] Timed out after 120s - model may still be downloading")
    print("  [HINT] Run 'python warm_up.py' again after a minute")
except Exception as e:
    print("  [FAIL]", e)

print("")
print("=== Done ===")
