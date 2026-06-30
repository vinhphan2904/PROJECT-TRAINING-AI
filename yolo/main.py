from lib import *

model = YOLO("yolo/runs/detect/train/weights/best.pt")

url = "http://192.168.1.22/capture"  


def white_balance(img):
    img = img.astype(np.float32)

    b, g, r = cv2.split(img)

    avg_b = np.mean(b)
    avg_g = np.mean(g)
    avg_r = np.mean(r)

    avg = (avg_b + avg_g + avg_r) / 3

    b = b * avg / avg_b
    g = g * avg / avg_g
    r = r * avg / avg_r

    img = cv2.merge([b, g, r])

    img = np.clip(img, 0, 255).astype(np.uint8)

    return img

def clahe(img):

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8,8)
    )

    l = clahe.apply(l)

    lab = cv2.merge((l,a,b))

    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def sharpen(img):

    kernel = np.array([
        [0,-1,0],
        [-1,5,-1],
        [0,-1,0]
    ])

    return cv2.filter2D(img,-1,kernel)

while True:
    time_date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    response = requests.get(url)

    img_array = np.frombuffer(response.content, np.uint8)

    frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    if frame is None:
        continue


    # frame = white_balance(frame)
    # frame = clahe(frame)
    # frame = sharpen(frame)

    results = model.predict(
        frame,
        conf=0.3,
        verbose=False
    )

    img_show = frame.copy()

    for r in results:

        for box in r.boxes:

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            cls = int(box.cls[0])

            conf = float(box.conf[0])

            # Crop ROI
            roi = frame[y1:y2, x1:x2]

            if roi.size != 0 and model.names[cls] != None:
                filename = f'yolo/data/{time_date}.jpg'
                cv2.imwrite(filename,roi)
                cv2.imshow("ROI", roi)
                

            # Vẽ bounding box
            cv2.rectangle(
                img_show,
                (x1,y1),
                (x2,y2),
                (0,255,0),
                2
            )

            cv2.putText(
                img_show,
                f"{model.names[cls]} {conf:.2f}",
                (x1,y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0,255,0),
                2
            )

    cv2.imshow("YOLO", img_show)
    time.sleep(2)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()