import os
import cv2
import numpy as np
import pyttsx3
from ultralytics import YOLO, YOLOWorld

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
# 功能函式 2：Locate Anything 相對位置精算插件
# (計算：水平角度、點鐘方向、估算距離公尺、雙 LRA 震動馬達強度)
# ==========================================
def calculate_relative_position(box_xyxy, img_w, img_h, fov_deg=70.0):
    x1, y1, x2, y2 = box_xyxy
    cx = (x1 + x2) / 2.0
    
    # 1. 計算水平相對角度 (-35度 ~ +35度)
    norm_x = (cx - (img_w / 2.0)) / (img_w / 2.0)
    angle_deg = norm_x * (fov_deg / 2.0)
    
    # 2. 換算點鐘方向 (10點鐘 ~ 2點鐘)
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
        
    # 3. 估算相對距離 (公尺)
    bottom_ratio = max(0.1, min(1.0, y2 / float(img_h)))
    est_distance_m = round(0.8 / (bottom_ratio ** 1.8), 1)
    
    # 4. 換算眼鏡雙 LRA 震動馬達強度 (0% ~ 100%)
    urgency = max(0.0, min(1.0, (4.0 - est_distance_m) / 3.2))
    if angle_deg < -7:
        lra_left, lra_right = int(urgency * 100), 0
    elif angle_deg > 7:
        lra_left, lra_right = 0, int(urgency * 100)
    else:
        lra_left, lra_right = int(urgency * 100), int(urgency * 100)
        
    return round(angle_deg, 1), clock_str, est_distance_m, lra_left, lra_right

# ==========================================
# 主程式開始：讀取測試照片
# ==========================================
image_path = "d:/畢業專題/test/test.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"❌ 找不到照片：{image_path}，請確認資料夾內有放 test.jpg！")
    exit()

img_h, img_w = img.shape[:2]
combined_img = img.copy()

# 內建中英對照字典（讓你輸入中文或英文都能通！）
en_to_tw = {
    "stairs": "階梯", "tactile paving": "導盲磚", "bollard": "阻車柱",
    "traffic cone": "三角錐", "door": "門口", "pole": "柱子",
    "person": "行人", "car": "汽車", "motorcycle": "機車", "bus": "公車",
    "bicycle": "腳踏車", "truck": "大貨車", "trash can": "垃圾桶",
    "bench": "長椅", "chair": "椅子", "dog": "狗", "cat": "貓",
    "tree": "樹", "fire hydrant": "消防栓", "umbrella": "雨傘",
    "backpack": "背包", "handbag": "手提包", "sign": "招牌", "bottle": "水瓶"
}
# 自動建立「中文轉英文」反查字典
tw_to_en = {v: k for k, v in en_to_tw.items()}

# ==========================================
# 1. 🔍 互動式提示詞輸入區（想找什麼自己打！）
# ==========================================
print("\n" + "=" * 65)
print("🔍 【Locate Anything 開放詞彙尋物設定】")
print("   您可以輸入想在照片中尋找的物品（英文或常見中文皆可，多個請用逗號 , 隔開）")
print("   💡 範例輸入：tree, trash can, bus, traffic cone  或  樹, 垃圾桶, 公車")
print("   💡 若直接按【Enter 鍵】，將自動載入預設的 10 種視障避障目標！")
print("=" * 65)

user_input = input("👉 請輸入想尋找的物件名稱 (直接按 Enter 使用預設清單): ").strip()

default_prompts = [
    "stairs", "tactile paving", "bollard", "traffic cone",
    "door", "pole", "person", "car", "motorcycle", "bus"
]

if user_input:
    # 支援半形逗號 (,) 與全形逗號 (，) 分隔
    raw_items = [x.strip() for x in user_input.replace("，", ",").split(",") if x.strip()]
    target_prompts = []
    for item in raw_items:
        if item in tw_to_en:
            # 如果使用者打中文（例如「垃圾桶」），自動轉成英文 "trash can" 給模型
            target_prompts.append(tw_to_en[item])
        else:
            # 如果使用者打英文（例如 "tree"），直接加入
            target_prompts.append(item.lower())
    is_custom_search = True
    print(f"✅ 已設定自訂尋找目標：{target_prompts}")
else:
    target_prompts = default_prompts
    is_custom_search = False
    print(f"ℹ️ 使用預設視障避障目標清單：{target_prompts}")

# ==========================================
# 2. 載入 Locate Anything 模型並執行定位
# ==========================================
print("⏳ 正在載入視覺定位模型並搜尋目標...")
try:
    # 使用 YOLOWorld 基底模型，可自由動態切換任意數量的提示詞
    model_locate = YOLOWorld("yolov8s-worldv2.pt")
    model_locate.set_classes(target_prompts)
except Exception:
    # 備用方案：讀取本地 locate_model.pt
    model_locate = YOLO("d:/畢業專題/test/locate_model.pt")
    if is_custom_search:
        model_locate.set_classes(target_prompts)

res_loc = model_locate.predict(img, conf=0.18, verbose=False)

nav_messages = []
light_status_msg = None
road_guidance = []
max_lra_left = 0
max_lra_right = 0

print("\n" + "=" * 65)
print("🎯 【視覺定位與目標相對位置計算報告】")
print("=" * 65)

if len(res_loc[0].boxes) == 0:
    print("⚠️ 在照片中沒有找到您指定的目標，可以試著換個英文單字喔！")

for box in res_loc[0].boxes:
    cls_id = int(box.cls[0])
    en_name = model_locate.names[cls_id]
    tw_name = en_to_tw.get(en_name, en_name)  # 若字典沒有中文，直接唸出英文名稱
    x1, y1, x2, y2 = map(int, box.xyxy[0])
    
    angle, clock_dir, dist_m, lra_l, lra_r = calculate_relative_position((x1, y1, x2, y2), img_w, img_h)
    max_lra_left = max(max_lra_left, lra_l)
    max_lra_right = max(max_lra_right, lra_r)
    
    print(f"📍 找到目標：【{tw_name} ({en_name})】 -> {clock_dir} ({angle:+}°) | 距離約 {dist_m}m | 震動 L:{lra_l}% R:{lra_r}%")
    
    # 如果是使用者自己輸入想找的東西，不管多遠都加入語音播報；若是預設避障，則播報 3 公尺內的目標
    if is_custom_search or dist_m <= 3.0:
        nav_messages.append(f"{clock_dir}約{dist_m}公尺有{tw_name}")
        
    color = (0, 0, 255) if dist_m <= 2.0 else (255, 160, 0)
    cv2.rectangle(combined_img, (x1, y1), (x2, y2), color, 2)
    cv2.putText(combined_img, f"{en_name} {dist_m}m ({angle:+}deg)", (x1, max(y1 - 8, 20)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

# ==========================================
# 3. 載入三合一專屬模型（號誌燈、人行道、斑馬線）
# ==========================================
# 新增英文標籤 (LIGHT, SIDEWALK, CROSSWALK) 避免 OpenCV 圖片出現 ??? 亂碼
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
            road_guidance.append(f"{clock_dir}有{label_tw}")  # 語音維持唸中文
            tag = f"{label_en} | {dist_m}m | {angle:+}deg"   # 圖片顯示英文，不出現 ???
            
        cv2.rectangle(combined_img, (x1, y1), (x2, y2), box_color, 3)
        cv2.putText(combined_img, tag, (x1, max(y1 - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

# ==========================================
# 4. 繪製中心 0° 基準線與「雙 LRA 震動馬達狀態儀表板」
# ==========================================
cv2.line(combined_img, (img_w // 2, 0), (img_w // 2, img_h), (0, 255, 0), 2)
cv2.rectangle(combined_img, (0, 0), (img_w, 55), (25, 25, 25), -1)
hud_text = f"LRA Motor Vibration -> [LEFT: {max_lra_left}%]  |  [CENTER 0 deg]  |  [RIGHT: {max_lra_right}%]"
cv2.putText(combined_img, hud_text, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

# ==========================================
# 5. 組裝中文語音並播報
# ==========================================
final_speech_parts = []
if light_status_msg:
    final_speech_parts.append(light_status_msg)
if road_guidance:
    final_speech_parts.extend(list(dict.fromkeys(road_guidance)))
if nav_messages:
    # 最多播報前 4 個目標，避免語音太長
    final_speech_parts.extend(list(dict.fromkeys(nav_messages))[:4])

if not final_speech_parts:
    final_speech = "前方無明顯號誌或指定目標，請小心慢行。"
else:
    final_speech = "導航提示：" + "，".join(final_speech_parts) + "。"

print("\n" + "=" * 65)
print(f"📳 【眼鏡雙 LRA 震動輸出】：左馬達 {max_lra_left}%  |  右馬達 {max_lra_right}%")
print(f"🔊 【AI 語音導航播報】    ：{final_speech}")
print("=" * 65 + "\n")

output_path = "d:/畢業專題/test/locate_smart_result.jpg"
cv2.imwrite(output_path, combined_img)

cv2.namedWindow("Locate Anything + 3-in-1 Blind Navigation", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Locate Anything + 3-in-1 Blind Navigation", 950, 680)
cv2.imshow("Locate Anything + 3-in-1 Blind Navigation", combined_img)
cv2.waitKey(500)

engine = pyttsx3.init()
engine.setProperty("rate", 160)
engine.say(final_speech)
engine.runAndWait()

print("請點擊圖片視窗並按下『任意鍵』關閉...")
cv2.waitKey(0)
cv2.destroyAllWindows()