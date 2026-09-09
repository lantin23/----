import cv2

def list_cameras(max_check=5):
    for idx in range(max_check):
        cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
        if cap.isOpened():
            print(f"ID {idx} →可用")
            cap.release()
        else:
            print(f"ID {idx} →不可用")

list_cameras()