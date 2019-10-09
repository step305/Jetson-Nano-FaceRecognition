import cv2
import platform
from PIL import Image
import face_recognition
from edgetpu.detection.engine import DetectionEngine
from queue import Queue
from multiprocessing import Process
import multiprocessing
import time
from time import sleep

def camThread(frameBuffer, results, stop_prog):
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FPS, 15)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    window_name = 'Video'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    t0 = time.monotonic()
    last_result=None
    frames_cnt = 0
    while True:
        ret, frame = cam.read()
        if not ret:
            continue
        if frameBuffer.full():
            frameBuffer.get()
        frameBuffer.put(frame.copy())
        res = None
        if not results.empty():
            res = results.get(False)
            imdraw = overlay_on_image(frame, res)
            last_result = res
        else:
            imdraw = overlay_on_image(frame,last_result)
        cv2.imshow('Video', imdraw)
        frames_cnt += 1
        if frames_cnt >= 15:
            t1 = time.monotonic()
            print('FPS={d:.1f}'.format(d = frames_cnt/(t1-t0)))
            frames_cnt = 0
            t0 = t1
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            stop_prog.set()
            break
    cam.release()    

def overlay_on_image(frame, result):
    if isinstance(result, type(None)):
        return frame.copy()
    img = frame.copy()
    boxes = result["boxes"]
    encod = result["encodings"]
    k = 1
    for box in boxes:
        y0, x1, y1, x0 = box
        cv2.rectangle(img, (x0,y0), (x1,y1), (255,0,0), 3)
        cv2.putText(img, '{d}'.format(d=k), (x0+6,y1-6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
        k += 1
    return img

def recognition(frameBuffer, objsBuffer, stop_prog, path):
    engine = DetectionEngine('mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite')
    face_cnt = 0
    while True:
        if stop_prog.is_set():
            break
        if frameBuffer.empty():
            continue
        t0 = time.monotonic()
        bgr_img = frameBuffer.get()
        rgb_img = bgr_img[:, :, ::-1].copy()
        arr_img = Image.fromarray(rgb_img)
        t1 = time.monotonic()
        objs = engine.DetectWithImage(arr_img, threshold = 0.1, keep_aspect_ratio = True, relative_coord = False, top_k = 5)
        t2 = time.monotonic()
        coral_boxes = []
        m = 0
        for obj in objs:
            x0, y0, x1, y1 = obj.bounding_box.flatten().tolist()
            w = x1-x0
            h = y1-y0
            x0 = int(x0+w/10)
            y0 = int(y0+h/4)
            x1 = int(x1-w/10)
            y1 = int(y1)
            coral_boxes.append((y0, x1, y1, x0))
            face_mini = bgr_img[y0:y1,x0:x1]
            cv2.imwrite(path + "face_" + str(face_cnt) + "_" + str(m) + ".jpg", face_mini)
            print(path + "face_" + str(face_cnt) + "_" + str(m) + ".jpg")
            m += 1
            face_cnt += 1
        sleep(1)
        enc=[]
        objsBuffer.put({"boxes": coral_boxes, "encodings": enc})


if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
    prog_stop = multiprocessing.Event()
    prog_stop.clear()
    recImage = multiprocessing.Queue(2)
    resultRecogn = multiprocessing.Queue(2)
    camProc = Process(target=camThread, args=(recImage, resultRecogn, prog_stop), daemon=True)
    camProc.start()
    frecogn = Process(target=recognition, args=(recImage, resultRecogn, prog_stop, "/home/step305/face_database/Maxim/"), daemon=True)
    frecogn.start()

    while True:
        if prog_stop.is_set():
            camProc.terminate()
            frecogn.terminate()
            break
        sleep(1)
    cv2.destroyAllWindows()
