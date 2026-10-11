import cv2
import requests

# 你的 Render 雲端伺服器網址
url = 'https://mang-ren-fu-zhu-api.onrender.com/analyze'

cap = cv2.VideoCapture(0)

print("👓 智慧眼鏡模擬器已啟動！")
print("▶ 按 'c' 鍵：拍攝並傳送給雲端伺服器")
print("▶ 按 'q' 鍵：關閉眼鏡")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("無法讀取攝影機畫面")
        break

    cv2.imshow("Smart Glasses Simulator", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        print("\n📸 已拍照！正在上傳至雲端分析...")
        
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = img_encoded.tobytes()
        
        try:
            files = {'image': ('frame.jpg', img_bytes, 'image/jpeg')}
            response = requests.post(url, files=files)
            
            # --- 新增的除錯檢查 ---
            if response.status_code != 200:
                print(f"⚠️ 伺服器崩潰了！狀態碼：{response.status_code}")
                print(f"伺服器真實回傳內容：\n{response.text}")
                continue # 跳過這次，重新等待拍照
                
            # 接收正常回傳的 JSON
            result = response.json()
            if 'status' in result and result['status'] == 'success':
                print(f"✅ 伺服器回傳：{result['message']}")
            else:
                print(f"⚠️️ 伺服器錯誤：{result}")
                
        except Exception as e: # 這個 except 絕對不能漏掉
            print(f"發送失敗。錯誤：{e}")

cap.release()
cv2.destroyAllWindows()