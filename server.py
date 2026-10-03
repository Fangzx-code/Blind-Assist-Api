from flask import Flask, request, jsonify
import cv2
import numpy as np
from google import genai
from PIL import Image
from ultralytics import YOLO

app = Flask(__name__)

# 1. 初始化雲端大腦 (上雲端後我們會用環境變數來隱藏 API Key)
GOOGLE_API_KEY = "請把你的_API_KEY_貼在這裡"
client = genai.Client(api_key=GOOGLE_API_KEY)

# 2. 初始化邊緣視覺
yolo_model = YOLO('yolo26m.pt')

@app.route('/', methods=['GET'])
def health_check():
    return "✅ 盲人輔助系統雲端 API 運作中！", 200

@app.route('/analyze', methods=['POST'])
def analyze_image():
    if 'image' not in request.files:
        return jsonify({"error": "沒有收到影像檔案"}), 400
    
    file = request.files['image']
    img_bytes = file.read()
    np_img = np.frombuffer(img_bytes, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    frame = cv2.resize(frame, (640, 480))
    
    # [層級 1] YOLO 偵測
    results = yolo_model(frame, verbose=False)
    annotated_frame = results[0].plot()
    detected_classes = [yolo_model.names[int(c)] for c in results[0].boxes.cls]
    detected_str = ", ".join(set(detected_classes))
    
    # [層級 2] Gemini 分析
    rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(rgb_frame)
    prompt = f"你是盲人輔助系統。YOLO 模型初步框出了：{detected_str}。請用繁體中文以簡短口語的一句話，向盲人描述這些重要物品的大致方位。"
    
    try:
        response = client.models.generate_content(
            model='gemini-3.6-flash',
            contents=[prompt, pil_img]
        )
        reply = response.text
        
        # 只回傳 JSON 文字，不發聲
        return jsonify({"status": "success", "message": reply, "yolo_detected": detected_str})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # 上傳到雲端時，通常會需要綁定 PORT 環境變數，本地端預設為 5000
    import os
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)