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
    lower_red1, upper_red1 = np.array([0, 100, 100]), np.array([10, 255, 255])
    lower_red2, upper_red2 = np.array([160, 100, 100]), np.array([180, 255, 255])
    mask_red = cv2.inRange(hsv, lower_red1, upper_red1) | cv2.inRange(hsv, lower_red2, upper_red2)
    
    lower_green, upper_green = np.array([35, 80, 80]), np.array([95, 255, 255])
    mask_green = cv2.inRange(hsv, lower_green, upper_green)
    
    red_pixels = cv2.countNonZero(mask_red)
    green_pixels = cv2.countNonZero(mask_green)
    
    if red_pixels > green_pixels and red_pixels > 15:
        return "RED (STOP)", "紅燈請停止"
    elif green_pixels > red_pixels and green_pixels > 15:
        return "GREEN (WALK)", "綠燈可通行"
    else:
        return "LIGHT", "發現號誌燈"

# ==========================================
# 功能函式 2：相對位置精算 (角度、點鐘方向、距離、LRA 震動)
# ==========================================
def calculate_relative_position(box_xyxy, img_w, img_h, fov_deg=70.0):
    x1, y1, x2, y2 = box_xyxy
    cx = (x1 + x2) / 2.0
    
    norm_x = (cx - (img_w / 2.0)) / (img_w / 2.0)
    angle_deg = norm_x * (fov_deg / 2.0)
    
    if angle_deg < -20:
        clock_str = "10點鐘方向"
    elif angle_deg < -7:
        clock_str = "11點鐘方向"
    elif angle_deg <= 7:
        clock_str = "12點鐘正前方"
    elif angle_deg <= 20:
        clock_str = "1點鐘方向"
    else:
        clock_str = "2點鐘方向"
        
    bottom_ratio = max(0.1, min(1.0, y2 / float(img_h)))
    est_distance_m = round(0.8 / (bottom_ratio ** 1.8), 1)
    
    urgency = max(0.0, min(1.0, (4.0 - est_distance_m) / 3.2))
    if angle_deg < -7:
        lra_left, lra_right = int(urgency * 100), 0
    elif angle_deg > 7:
        lra_left, lra_right = 0, int(urgency * 100)
    else:
        lra_left, lra_right = int(urgency * 100), int(urgency * 100)
        
    return round(angle_deg, 1), clock_str, est_distance_m, lra_left, lra_right

# ==========================================
# 主程式：全自動道路偵測與避障
# ==========================================
image_path = "d:/畢業專題/test/test.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"❌ 找不到照片：{image_path}，請確認資料夾內有放 test.jpg！")
    exit()

img_h, img_w = img.shape[:2]
combined_img = img.copy()

nav_messages = []
light_status_msg = None
road_guidance = []
max_lra_left = 0
max_lra_right = 0

tw_dict = {
    "stairs": "階梯", "tactile paving": "導盲磚", "bollard": "阻車柱",
    "traffic cone": "三角錐", "door": "門口", "pole": "柱子",
    "person": "行人", "car": "汽車", "motorcycle": "機車", "bus": "公車",
    "bicycle": "腳踏車", "truck": "大貨車"
}

# 1. 自動載入預設障礙物定位模型
locate_path = "d:/畢業專題/test/locate_model.pt"
if os.path.exists(locate_path):
    print("✅ [全自動模式] 正在使用 locate_model.pt 進行道路障礙物偵測...")
    model_locate = YOLO(locate_path)
else:
    print("ℹ️ [全自動模式] 正在使用官方 yolov8s.pt 進行基礎障礙物偵測...")
    model_locate = YOLO("yolov8s.pt")

res_loc = model_locate(img, conf=0.22, verbose=False)

print("\n" + "=" * 65)
print("🎯 【全自動道路障礙物與相對位置報告】")
print("=" * 65)

for box in res_loc[0].boxes:
    cls_id = int(box.cls[0])
    en_name = model_locate.names[cls_id]
    if en_name not in tw_dict:
        continue
    tw_name = tw_dict[en_name]
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    angle, clock_dir, dist_m, lra_l, lra_r = calculate_relative_position((x1, y1, x2, y2), img_w, img_h)
    max_lra_left = max(max_lra_left, lra_l)
    max_lra_right = max(max_lra_right, lra_r)
    
    print(f"📍 障礙物：【{tw_name}】 -> {clock_dir} ({angle:+}°) | 距離約 {dist_m}m | 震動 L:{lra_l}% R:{lra_r}%")
    
    if dist_m <= 2.5:
        nav_messages.append(f"{clock_dir}約{dist_m}公尺有{tw_name}")
        
    color = (0, 0, 255) if dist_m <= 2.0 else (255, 160, 0)
    cv2.rectangle(combined_img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(combined_img, f"{en_name} {dist_m}m ({angle:+}deg)", (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# 2. 自動載入三合一專屬模型（號誌燈、人行道、斑馬線）
custom_models = [
    ("號誌燈", "LIGHT", "d:/畢業專題/test/light_model.pt", 0.20, (0, 255, 255)),
    ("人行道", "SIDEWALK", "d:/畢業專題/test/sidewalk_model.pt", 0.20, (0, 255, 0)),
    ("斑馬線", "CROSSWALK", "d:/畢業專題/test/crosswalk_model.pt", 0.20, (255, 0, 255)),
]

for label_tw, label_en, path, conf_thres, box_color in custom_models:
    if not os.path.exists(path):
        continue
    model = YOLO(path)
    res = model(img, conf=conf_thres, verbose=False)
    
    for box in res[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        angle, clock_dir, dist_m, _, _ = calculate_relative_position((x1, y1, x2, y2), img_w, img_h)
        
        if label_tw == "號誌燈":
            color_en, color_tw = detect_light_color(img[y1:y2, x1:x2])
            light_status_msg = f"{clock_dir}{color_tw}"
            tag = f"{color_en} | {angle:+}deg"
        else:
            road_guidance.append(f"{clock_dir}有{label_tw}")
            tag = f"{label_en} | {dist_m}m | {angle:+}deg"
            
        cv2.rectangle(combined_img, (x1, y1), (x2, y2), box_color, 3)
        cv2.putText(combined_img, tag, (x1, max(y1 - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

# 3. 繪製中心線與雙 LRA 震動儀表板
cv2.line(combined_img, (img_w // 2, 0), (img_w // 2, img_h), (0, 255, 0), 2)
cv2.rectangle(combined_img, (0, 0), (img_w, 55), (25, 25, 25), -1)
hud_text = f"LRA Motor Vibration -> [LEFT: {max_lra_left}%]  |  [CENTER 0 deg]  |  [RIGHT: {max_lra_right}%]"
cv2.putText(combined_img, hud_text, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

# 4. 組裝中文語音並播報
final_speech_parts = []
if light_status_msg:
    final_speech_parts.append(light_status_msg)
if road_guidance:
    final_speech_parts.extend(list(dict.fromkeys(road_guidance)))
if nav_messages:
    final_speech_parts.extend(list(dict.fromkeys(nav_messages))[:3])

if not final_speech_parts:
    final_speech = "前方無明顯號誌或近距離障礙物，請小心慢行。"
else:
    final_speech = "導航提示：" + "，".join(final_speech_parts) + "。"

print("\n" + "=" * 65)
print(f"📳 【眼鏡雙 LRA 震動輸出】：左馬達 {max_lra_left}%  |  右馬達 {max_lra_right}%")
print(f"🔊 【AI 語音導航播報】    ：{final_speech}")
print("=" * 65 + "\n")

output_path = "d:/畢業專題/test/auto_nav_result.jpg"
cv2.imwrite(output_path, combined_img)

cv2.namedWindow("Mode 1: Auto Blind Navigation", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Mode 1: Auto Blind Navigation", 950, 680)
cv2.imshow("Mode 1: Auto Blind Navigation", combined_img)
cv2.waitKey(500)

engine = pyttsx3.init()
engine.setProperty("rate", 160)
engine.say(final_speech)
engine.runAndWait()

print("請點擊圖片視窗並按下『任意鍵』關閉...")
cv2.waitKey(0)
cv2.destroyAllWindows()