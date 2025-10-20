import cv2
import threading

class VideoStream:
    def __init__(self, source=0):
        print("🎥 Initializing video stream...")
        self.cap = cv2.VideoCapture(source, cv2.CAP_DSHOW)
        if not self.cap.isOpened():
            raise IOError("❌ Cannot open webcam")
        self.ret, self.frame = self.cap.read()
        self.stopped = False
        print("✅ Webcam opened successfully!")

    def start(self):
        threading.Thread(target=self.update, daemon=True).start()
        return self

    def update(self):
        while not self.stopped:
            self.ret, self.frame = self.cap.read()

    def read(self):
        return self.ret, self.frame

    def stop(self):
        self.stopped = True
        self.cap.release()
        cv2.destroyAllWindows()
        print("✅ Stream released and windows closed.")
