import cv2

for i in range(5):  # try 0–4; increase if you have many devices
    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    ok = cap.isOpened()
    ret, frame = cap.read() if ok else (False, None)
    cap.release()
    print(f"index {i}: opened={ok}, read_ok={ret}, shape={None if frame is None else frame.shape}")