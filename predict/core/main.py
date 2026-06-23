from lib import *
from camera import ESP32CAMERA

camera = ESP32CAMERA("192.168.1.38")
cnt = 1

while True:
    img = camera.capture()
    if img is None:
        continue
    time_now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    frame = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
    filename = f'predict/data/{time_now}.jpg'
    img.save(filename)
    cv2.imshow("ESP32 Camera",frame)
    if cv2.waitKey(5000) & 0xFF == ord('q'):
        break
    cnt += 1
cv2.destroyAllWindows()