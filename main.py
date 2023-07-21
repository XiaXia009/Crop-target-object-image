from ultralytics import YOLO
import cv2

model = YOLO('Model_4.pt')

input_width = 640
input_height = 640
fps = 30

video_capture = cv2.VideoCapture("video.mp4")

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter("output.mp4", fourcc, fps, (input_width, input_height))

background = cv2.imread("background.jpg")

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    results = model(frame)

    composite_frame = background.copy()

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = box.xyxy.tolist()[0]

            obj_img = frame[int(y1):int(y2), int(x1):int(x2)]
            obj_height, obj_width, _ = obj_img.shape

            if obj_height > background.shape[0] or obj_width > background.shape[1]:
                continue

            pos_x1 = max(int(x1), 0)
            pos_y1 = max(int(y1), 0)
            pos_x2 = min(int(x1) + obj_width, input_width)
            pos_y2 = min(int(y1) + obj_height, input_height)

            obj_pos_x1 = max(-int(x1), 0)
            obj_pos_y1 = max(-int(y1), 0)
            obj_pos_x2 = obj_pos_x1 + pos_x2 - pos_x1
            obj_pos_y2 = obj_pos_y1 + pos_y2 - pos_y1

            composite_frame[pos_y1:pos_y2, pos_x1:pos_x2] = obj_img[obj_pos_y1:obj_pos_y2, obj_pos_x1:obj_pos_x2]

            class_name = result.names[int(box.cls)]
            cv2.imwrite(f'{class_name}_{box.id}.png', obj_img)

    out.write(composite_frame)

    cv2.imshow("output", composite_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
out.release()
cv2.destroyAllWindows()
