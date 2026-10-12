import os
import cv2
import numpy as np
import pyttsx3
from ultralytics import YOLO, YOLOWorld

# ==========================================
# 功能函式 1：OpenCV 紅綠燈顏色判斷
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
        return "LIGHT", "號誌燈"

# ==========================================
# 功能函式 2：計算目標相對位置 (角度、點鐘方向、距離、LRA 震動)
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
# 主程式：英文提示詞精準搜尋 (中文語音播報)
# ==========================================
image_path = "d:/畢業專題/test/test.jpg"
img = cv2.imread(image_path)

if img is None:
    print(f"❌ 找不到照片：{image_path}，請確認資料夾內有放 test.jpg！")
    exit()

img_h, img_w = img.shape[:2]
search_img = img.copy()

# 英文轉中文語音播報字典 (讓電腦唸出中文)
en_to_tw = {
    "light": "號誌燈", "sidewalk": "人行道", "crosswalk": "斑馬線",
    "stairs": "階梯", "tactile paving": "導盲磚", "bollard": "阻車柱",
    "traffic cone": "三角錐", "door": "門口", "pole": "電線桿與柱子",
    "person": "行人", "car": "汽車", "motorcycle": "機車", "bus": "公車",
    "bicycle": "腳踏車", "truck": "貨車", "trash can": "垃圾桶",
    "bench": "長椅", "chair": "椅子", "dog": "狗", "cat": "貓",
    "tree": "路樹", "fire hydrant": "消防栓", "umbrella": "雨傘",
    "backpack": "背包", "handbag": "手提包", "sign": "招牌", "bottle": "水瓶",
    "building": "建築物", "streetlight": "路燈", "fence": "圍欄"
}

# 1. 顯示英文提示詞小抄，請使用者輸入英文
print("\n" + "=" * 70)
print("🔍 【模式二：Locate Anything 英文提示詞精準尋物系統】")
print("-" * 70)
print("📌 專屬模型目標：light (紅綠燈), sidewalk (人行道), crosswalk (斑馬線)")
print("📌 道路與車輛  ：bus (公車), car (汽車), motorcycle (機車), person (行人)")
print("📌 常見路障    ：traffic cone (三角錐), bollard (阻車柱), pole (電線桿), tree (樹)")
print("📌 設施與尋物  ：stairs (階梯), door (門), trash can (垃圾桶), bench (長椅), sign (招牌)")
print("-" * 70)
print("💡 請直接輸入【英文單字】，多個目標請用逗號 , 隔開 (例如: bus, traffic cone, light)")
print("=" * 70)

user_input = input("👉 請輸入英文目標名稱 (直接按 Enter 預設找 bus, traffic cone, crosswalk): ").strip()

if not user_input:
    wanted_en_list = ["bus", "traffic cone", "crosswalk"]
else:
    wanted_en_list = [x.strip().lower() for x in user_input.replace("，", ",").split(",") if x.strip()]

print(f"\n⏳ 正在搜尋英文目標：{wanted_en_list} ...")

found_messages = []
max_lra_left = 0
max_lra_right = 0

print("\n" + "=" * 65)
print("🎯 【指定目標搜尋與相對位置報告】")
print("=" * 65)

# ==========================================
# 2. 若包含 light, sidewalk, crosswalk，呼叫你訓練的專屬模型
# ==========================================
custom_map = {
    "light": ("號誌燈", "LIGHT", "d:/畢業專題/test/light_model.pt", 0.20, (0, 255, 255)),
    "sidewalk": ("人行道", "SIDEWALK", "d:/畢業專題/test/sidewalk_model.pt", 0.20, (0, 255, 0)),
    "crosswalk": ("斑馬線", "CROSSWALK", "d:/畢業專題/test/crosswalk_model.pt", 0.20, (255, 0, 255)),
}

for target_key in wanted_en_list:
    if target_key in custom_map:
        label_tw, label_en, path, conf_thres, box_color = custom_map[target_key]
        if os.path.exists(path):
            c_model = YOLO(path)
            res_c = c_model(img, conf=conf_thres, verbose=False)
            for box in res_c[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                angle, clock_dir, dist_m, lra_l, lra_r = calculate_relative_position((x1, y1, x2, y2), img_w, img_h)
                max_lra_left = max(max_lra_left, lra_l)
                max_lra_right = max(max_lra_right, lra_r)
                
                if target_key == "light":
                    color_en, color_tw = detect_light_color(img[y1:y2, x1:x2])
                    print(f"✅ 找到目標：【{label_tw} ({color_tw})】 -> {clock_dir} ({angle:+}°)")
                    found_messages.append(f"在{clock_dir}找到{color_tw}")
                    tag = f"FOUND: {color_en} ({angle:+}deg)"
                else:
                    print(f"✅ 找到目標：【{label_tw}】 -> {clock_dir} ({angle:+}°) | 距離約 {dist_m}m")
                    found_messages.append(f"在{clock_dir}約{dist_m}公尺處找到{label_tw}")
                    tag = f"FOUND: {label_en} | {dist_m}m ({angle:+}deg)"
                    
                cv2.rectangle(search_img, (x1, y1), (x2, y2), box_color, 3)
                cv2.putText(search_img, tag, (x1, max(y1 - 10, 25)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, box_color, 2)

# ==========================================
# 3. 針對其他英文目標：保留 10 大基準詞背景對比，確保高精準度
# ==========================================
general_targets = [t for t in wanted_en_list if t not in custom_map]

if general_targets:
    base_anchor_classes = [
        "stairs", "tactile paving", "bollard", "traffic cone",
        "door", "pole", "person", "car", "motorcycle", "bus"
    ]
    combined_classes = list(dict.fromkeys(base_anchor_classes + general_targets))
    
    locate_path = "d:/畢業專題/test/locate_model.pt"
    if set(general_targets).issubset(set(base_anchor_classes)) and os.path.exists(locate_path):
        model_search = YOLO(locate_path)
    else:
        model_search = YOLOWorld("yolov8s-worldv2.pt")
        model_search.set_classes(combined_classes)
        
    res = model_search.predict(img, conf=0.20, iou=0.45, verbose=False)
    
    for box in res[0].boxes:
        cls_id = int(box.cls[0])
        en_name = model_search.names[cls_id]
        
        if en_name not in general_targets:
            continue
            
        tw_name = en_to_tw.get(en_name, en_name)
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        
        angle, clock_dir, dist_m, lra_l, lra_r = calculate_relative_position((x1, y1, x2, y2), img_w, img_h)
        max_lra_left = max(max_lra_left, lra_l)
        max_lra_right = max(max_lra_right, lra_r)
        
        print(f"✅ 找到目標：【{tw_name} ({en_name})】")
        print(f"   ├─ 方向：{clock_dir} (水平角 {angle:+}°)")
        print(f"   ├─ 距離：約 {dist_m} 公尺")
        print(f"   └─ 引導震動：左馬達 {lra_l}% | 右馬達 {lra_r}%")
        
        found_messages.append(f"在{clock_dir}約{dist_m}公尺處找到{tw_name}")
        
        cv2.rectangle(search_img, (x1, y1), (x2, y2), (0, 255, 0), 3)
        cv2.putText(search_img, f"FOUND: {en_name} | {dist_m}m ({angle:+}deg)", (x1, max(y1 - 10, 25)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

# 4. 彙整語音播報
if not found_messages:
    print(f"⚠️ 報告：畫面中沒有找到 {wanted_en_list}，請確認英文拼字或換個單字試試！")
    final_speech = "搜尋結果：目前畫面中沒有找到您指定的物品。"
else:
    unique_msgs = list(dict.fromkeys(found_messages))[:4]
    final_speech = "尋物結果：" + "，".join(unique_msgs) + "。"

# 5. 繪製中心準心與頂部狀態列
cv2.line(search_img, (img_w // 2, 0), (img_w // 2, img_h), (0, 255, 255), 2)
cv2.rectangle(search_img, (0, 0), (img_w, 55), (25, 25, 25), -1)
hud_text = f"Search: {wanted_en_list} | LRA -> [L: {max_lra_left}% | R: {max_lra_right}%]"
cv2.putText(search_img, hud_text, (20, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)

print("\n" + "=" * 65)
print(f"🔊 【AI 尋物語音播報】：{final_speech}")
print("=" * 65 + "\n")

output_path = "d:/畢業專題/test/search_item_result.jpg"
cv2.imwrite(output_path, search_img)

cv2.namedWindow("Mode 2: English Prompt Search", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Mode 2: English Prompt Search", 950, 680)
cv2.imshow("Mode 2: English Prompt Search", search_img)
cv2.waitKey(500)

engine = pyttsx3.init()
engine.setProperty("rate", 160)
engine.say(final_speech)
engine.runAndWait()

print("請點擊圖片視窗並按下『任意鍵』關閉...")
cv2.waitKey(0)
cv2.destroyAllWindows()