import os
import cv2
import numpy as np
import pyttsx3
from ultralytics import YOLO

# ==========================================
# 功能函式 1：OpenCV 紅綠燈顏色判斷 (HSV 色彩空間)
# ==========================================
def detect_light_color(crop_img):
    if crop_img is None or crop_img.size == 0:
        return "UNKNOWN", "號誌狀態不明"
    
    hsv = cv2.cvtColor(crop_img, cv2.COLOR_BGR2HSV)
    
    # 紅光範圍 (HSV 中紅色跨在 0 度與 180 度兩端)
    lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    
    # 綠光範圍 (涵蓋台灣小綠人偏青綠色的色調 35~95)
    lower_green, upper_green = np.array([35, 80, 80]), np.array([95, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    red_pixels = cv2.countNonZero(mask_red)
    green_pixels = cv2.countNonZero(mask_green)
    
    # 比較紅綠亮點數量
    if red_pixels > green_pixels and red_pixels > 15:
        return "RED (STOP)", "紅燈請停止"
    elif green_pixels > red_pixels and green_pixels > 15:
        return "GREEN (WALK)", "綠燈可通行"
    else:
        return "LIGHT", "發現號誌燈"

# ==========================================
# 功能函式 2：計算物件在畫面的「左/中/右」與「遠/近」
# ==========================================
def get_spatial_info(box_xyxy, img_w, img_h):
    x1, y1, x2, y2 = box_xyxy
    center_x = (x1 + x2) / 2
    
    # 1. 判斷左 / 正前方 / 右 (將畫面寬度切三等份)
    if center_x < img_w / 3:
        pos_tw, pos_en = "左前方", "L"
    elif center_x < (img_w * 2) / 3:
        pos_tw, pos_en = "正前方", "C"
    else:
        pos_tw, pos_en = "右前方", "R"
        
    # 2. 判斷遠近 (物體底部 y2 越靠近畫面下方，代表離視障者腳邊越近)
    if y2 > img_h * 0.75:
        dist_tw, dist_en = "近距離", "NEAR!"
    else:
        dist_tw, dist_en = "", ""
        
    return pos_tw, pos_en, dist_tw, dist_en

# ==========================================
# 主程式：讀取照片與多模型推論
# ==========================================
image_path = "d:/畢業專題/test/test.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"❌ 找不到照片：{image_path}，請確認資料夾內有放 test.jpg！")
    exit()

img_h, img_w = img.shape[:2]
combined_img = img.copy()

# 收集語音播報內容的清單
nav_messages = set()
light_status_msg = None

# 1. 載入官方進階模型 yolov8s.pt (抓車、人、機車與車道紅綠燈)
print("正在進行環境與障礙物分析...")
model_obs = YOLO("yolov8s.pt")
res_obs = model_obs(img, conf=0.25, classes=[0, 1, 2, 3, 5, 7, 9], verbose=False)

obs_name_map = {0: "行人", 1: "腳踏車", 2: "汽車", 3: "機車", 5: "公車", 7: "大貨車", 9: "號誌燈"}

for box in res_obs[0].boxes:
    cls_id = int(box.cls[0])
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    pos_tw, pos_en, dist_tw, dist_en = get_spatial_info((x1, y1, x2, y2), img_w, img_h)
    
    # 若官方模型抓到紅綠燈 (class 9)，也用 OpenCV 判斷紅綠
    if cls_id == 9:
        color_en, color_tw = detect_light_color(img[y1:y2, x1:x2])
        light_status_msg = f"{pos_tw}{color_tw}"
        cv2.rectangle(combined_img, (x1, y1), (x2, y2), (0, 255, 255), 3)
        cv2.putText(combined_img, f"[{pos_en}] {color_en}", (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    else:
        obj_tw = obs_name_map.get(cls_id, "障礙物")
        # 針對「近距離」或「正前方」的車與人發出語音警示
        if dist_tw == "近距離" or pos_tw == "正前方":
            nav_messages.add(f"注意{pos_tw}{dist_tw}有{obj_tw}")
        
        color = (0, 0, 255) if dist_tw == "近距離" else (255, 140, 0)
        cv2.rectangle(combined_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(combined_img, f"[{pos_en}] {model_obs.names[cls_id]} {dist_en}", (x1, max(y1 - 8, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# 2. 載入你訓練的三顆專屬模型（三合一核心）
custom_models = [
    ("號誌燈", "d:/畢業專題/test/light_model.pt", 0.20, (0, 255, 255)),
    ("人行道", "d:/畢業專題/test/sidewalk_model.pt", 0.20, (0, 255, 0)),
    ("斑馬線", "d:/畢業專題/test/crosswalk_model.pt", 0.20, (255, 0, 255)),
]

road_guidance = []

for label_tw, path, conf_thres, box_color in custom_models:
    if not os.path.exists(path):
        print(f"⚠️ 找不到 {path}，自動略過")
        continue
    model = YOLO(path)
    res = model(img, conf=conf_thres, verbose=False)
    
    for box in res[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        pos_tw, pos_en, _, _ = get_spatial_info((x1, y1, x2, y2), img_w, img_h)
        
        if label_tw == "號誌燈":
            color_en, color_tw = detect_light_color(img[y1:y2, x1:x2])
            light_status_msg = f"{pos_tw}{color_tw}"
            tag = f"[{pos_en}] {color_en}"
        else:
            road_guidance.append(f"{pos_tw}有{label_tw}")
            tag = f"[{pos_en}] {label_tw}"
            
        cv2.rectangle(combined_img, (x1, y1), (x2, y2), box_color, 3)
        cv2.putText(combined_img, tag, (x1, max(y1 - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, box_color, 2)

# ==========================================
# 功能 3：繪製「左/中/右導航分割線」與「近距離警戒線」
# ==========================================
cv2.line(combined_img, (img_w // 3, 0), (img_w // 3, img_h), (200, 200, 200), 2)
cv2.line(combined_img, (img_w * 2 // 3, 0), (img_w * 2 // 3, img_h), (200, 200, 200), 2)
cv2.line(combined_img, (0, int(img_h * 0.75)), (img_w, int(img_h * 0.75)), (0, 0, 255), 2)

cv2.putText(combined_img, "LEFT ZONE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
cv2.putText(combined_img, "CENTER ZONE", (img_w // 3 + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
cv2.putText(combined_img, "RIGHT ZONE", (img_w * 2 // 3 + 20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

# ==========================================
# 功能 4：組裝中文導航播報並用喇叭唸出來
# ==========================================
final_speech_parts = []
if light_status_msg:
    final_speech_parts.append(light_status_msg)
if road_guidance:
    final_speech_parts.extend(list(set(road_guidance)))
if nav_messages:
    final_speech_parts.extend(list(nav_messages))

if not final_speech_parts:
    final_speech = "前方無明顯號誌或障礙物，請小心慢行。"
else:
    final_speech = "導航提示：" + "，".join(final_speech_parts) + "。"

print("\n" + "=" * 50)
print(f"🔊 【AI 語音導航播報】：{final_speech}")
print("=" * 50 + "\n")

# 儲存結果圖
output_path = "d:/畢業專題/test/smart_nav_result.jpg"
cv2.imwrite(output_path, combined_img)
print(f"✅ 智慧導航分析圖已存檔至：{output_path}")

# 先彈出視窗顯示畫面，再啟動語音播報
cv2.namedWindow("Smart Blind Navigation System", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Smart Blind Navigation System", 950, 680)
cv2.imshow("Smart Blind Navigation System", combined_img)
cv2.waitKey(500)

# 啟動 TTS 語音播報
engine = pyttsx3.init()
engine.setProperty("rate", 160)
engine.say(final_speech)
engine.runAndWait()

print("請點擊圖片視窗並按下『任意鍵』關閉...")
cv2.waitKey(0)
cv2.destroyAllWindows()