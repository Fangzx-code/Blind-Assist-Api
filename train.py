from ultralytics import YOLO
import cv2

# 同時載入多顆模型（完全不互相覆蓋！）
model_light = YOLO("d:/畢業專題/test/light_model.pt")  # 專門認行人號誌
model_road = YOLO("d:/畢業專題/test/yolov8n.pt")       # 之後可換成訓練好的斑馬線模型 road_model.pt

image_path = "d:/畢業專題/test/test.jpg"
img = cv2.imread(image_path)

# 1. 先讓第一顆模型辨識，並把框畫在圖上
results_1 = model_light(img, conf=0.3)
annotated_img = results_1[0].plot()

# 2. 再讓第二顆模型辨識原圖，把框疊加畫在剛剛的圖上
results_2 = model_road(img, conf=0.3)
final_img = results_2[0].plot(img=annotated_img)

# 顯示疊加後的最終結果
cv2.namedWindow("Combined Result", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Combined Result", 800, 600)
cv2.imshow("Combined Result", final_img)
cv2.waitKey(0)
cv2.destroyAllWindows()