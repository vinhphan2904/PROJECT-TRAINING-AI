from lib import *

class ESP32CAMERA:
    def __init__(self,ip):
        self.url = "http://%s/capture" % ip

    def capture(self):
        try:
            respond = requests.get(self.url)

            if respond.status_code != 200:
                print("HTTP error",respond.status_code)
                return None
            
            img = Image.open(BytesIO(respond.content))

            return img
        
        except Exception as e: 
            print('Camera error', e)
            return None