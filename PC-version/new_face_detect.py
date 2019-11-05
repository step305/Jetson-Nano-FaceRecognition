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
import math
from sklearn import neighbors
import os
import os.path
import pickle
from http import server
import socketserver
import logging
import numpy as np
from datetime import datetime, timedelta

HTML_PAGE="""\
<html>
<head>
<title>Face recognition</title>
</head>
<body>
<center><h1>Cam</h1></center>
<center><img src="stream.mjpg" width="1280" height="720" /></center>
</body>
</html>
"""

def camThread(frameBuffer, results, MJPEGQueue, persBuffer, stop_prog):
    cam = cv2.VideoCapture(1)
    cam.set(cv2.CAP_PROP_FPS, 10)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    window_name = 'Video'
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)
    t0 = time.monotonic()
    last_result=None
    frames_cnt = 0
    persons = None
    while True:
        ret, frame = cam.read()
        if not ret:
            continue
        if frameBuffer.empty():
            frameBuffer.put(frame.copy())
        res = None
        if not results.empty():
            res = results.get(False)
            imdraw = overlay_on_image(frame, res)
            last_result = res
        else:
            imdraw = overlay_on_image(frame,last_result)
        if not persBuffer.empty():
            persons = persBuffer.get(False)
        imdraw = overlay_faces(imdraw, persons)
        cv2.imshow('Video', imdraw)
        frames_cnt += 1
        if frames_cnt >= 15:
            t1 = time.monotonic()
            print('FPS={d:.1f}'.format(d = frames_cnt/(t1-t0)))
            frames_cnt = 0
            t0 = t1
        if not MJPEGQueue.full():
            MJPEGQueue.put(imdraw.copy())
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            stop_prog.set()
            break
    cam.release()    

def overlay_faces(frame, persons):
    img = frame#.copy()
    if isinstance(persons, type(None)):
        return img
    x = 1280-288
    y = 1
    for person in persons:
        img[y:y+216, x:x+288] = person["face_image"]
        if person["seen_count"] == 1:
            visit_label = "First visit"
        else:
            visit_label = "{} visits".format(person["seen_count"])
        cv2.putText(img, visit_label, (x, y+184), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
        cv2.putText(img, person["name"], (x, y+200), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
        y += 216
        if y >= 648:
            break
    return img

def overlay_on_image(frame, result):
    if isinstance(result, type(None)):
        return frame#.copy()
    img = frame#.copy()
    boxes = result["boxes"]
    encod = result["names"]
    for box, name in zip(boxes,encod):
        y0, x1, y1, x0 = box
        cv2.rectangle(img, (x0,y0), (x1,y1), (255,0,0), 3)
        cv2.putText(img, '{d}'.format(d=name), (x0+6,y1-6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255,0,0), 2)
    return img

def recognition(frameBuffer, objsBuffer, persBuffer, stop_prog):
    engine = DetectionEngine('mobilenet_ssd_v2_face_quant_postprocess_edgetpu.tflite')
    with open("trained_knn_model.clf", 'rb') as f:
        knn_clf = pickle.load(f)
    known_persons={}
    for class_dir in os.listdir("/home/step305/face_database/"):
        face_image = cv2.imread(os.path.join("/home/step305/face_database/", class_dir, "face_ID.jpg"))
        face_image = cv2.resize(face_image, (288,216))
        known_persons[class_dir]={
            "first_seen": datetime(1,1,1),
            "name": class_dir,
            "first_seen_this_interaction": datetime(1,1,1),
            "last_seen": datetime(1,1,1),
            "seen_count": 0,
            "seen_frames": 0,
            "face_image": face_image
        }
    while True:
        if stop_prog.is_set():
            break
        if frameBuffer.empty():
            continue
        #t0 = time.monotonic()
        bgr_img = frameBuffer.get()
        rgb_img = bgr_img[:, :, ::-1].copy()
        arr_img = Image.fromarray(rgb_img)
        #t1 = time.monotonic()
        objs = engine.DetectWithImage(arr_img, threshold = 0.1, keep_aspect_ratio = True, relative_coord = False, top_k = 5)
        #t2 = time.monotonic()
        coral_boxes = []
        for obj in objs:
            x0, y0, x1, y1 = obj.bounding_box.flatten().tolist()
            w = x1-x0
            h = y1-y0
            x0 = int(x0+w/10)
            y0 = int(y0+h/4)
            x1 = int(x1-w/10)
            y1 = int(y1)
            coral_boxes.append((y0, x1, y1, x0))
        #t3 = time.monotonic()
        if coral_boxes:
            enc = face_recognition.face_encodings(rgb_img, coral_boxes)
            closest_distances = knn_clf.kneighbors(enc, n_neighbors=1)
            are_matches = [closest_distances[0][i][0] <= 0.55 for i in range(len(coral_boxes))]
            predR = []
            locR = []
            for pred, loc, rec in zip(knn_clf.predict(enc), coral_boxes, are_matches):
                if rec:
                    person_found = known_persons.get(pred)
                    if person_found != None:
                        if known_persons[pred]["first_seen"] != datetime(1,1,1):
                            known_persons[pred]["last_seen"] = datetime.now()
                            known_persons[pred]["seen_frames"] += 1
                            if datetime.now() - known_persons[pred]["first_seen_this_interaction"] > timedelta(minutes=5):
                                known_persons[pred]["first_seen_this_interaction"] = datetime.now()
                                known_persons[pred]["seen_count"] += 1
                                known_persons[pred]["seen_frames"] = 0
                        else:
                            known_persons[pred]["first_seen"] = datetime.now()
                            known_persons[pred]["last_seen"] = datetime.now()
                            known_persons[pred]["seen_count"] += 1
                            known_persons[pred]["first_seen_this_interaction"] = datetime.now()
                    predR.append(pred)
                else:
                    predR.append("unknown")
                locR.append(loc)
            if objsBuffer.empty():
                objsBuffer.put({"boxes": locR, "names": predR})
        else:
            if objsBuffer.empty():
                objsBuffer.put(None)
        dtnow = datetime.now()
        visi_faces = []
        for pers in known_persons:
            if dtnow-known_persons[pers]["last_seen"] < timedelta(seconds=10) and known_persons[pers]["seen_frames"] > 5:
                visi_faces.append(known_persons[pers])
        if persBuffer.empty():
            if len(visi_faces) > 0:
                persBuffer.put(visi_faces)
            else:
                persBuffer.put(None)
        del bgr_img
        del rgb_img
        del arr_img
        del objs
        #del visi_faces
        del dtnow
        #t4 = time.monotonic()
        #print('Prep time = {dt1:.1f}ms, Infer time = {dt2:.1f}ms, Face enc time = {dt3:.1f}ms, Overall time = {dt4:.1f}ms'.format(
        #    dt1=(t1-t0)*1000, dt2=(t2-t1)*1000, dt3=(t4-t3)*1000, dt4 = (t4-t0)*1000))

class StreamingHandler(server.BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == '/':
            self.send_response(301)
            self.send_header('Location', '/index.html')
            self.end_headers()
        elif self.path == '/index.html':
            stri = HTML_PAGE
            content = stri.encode('utf-8')
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.send_header('Conent-Length', len(content))
            self.end_headers()
            self.wfile.write(content)
        # elif self.path == '/data.html':
        #     stri = coral_engine.result_str
        #     content = stri.encode('utf-8')
        #     self.send_response(200)
        #     self.send_header('Content-Type', 'text/html')
        #     self.send_header('Conent-Length', len(content))
        #     self.end_headers()
        #     self.wfile.write(content)
        elif self.path == '/stream.mjpg':
            self.send_response(200)
            self.send_header('Age', 0)
            self.send_header('Cache-Control', 'no-cache, private')
            self.send_header('Pragma', 'no-cache')
            self.send_header('Content-Type', 'multipart/x-mixed-replace; boundary=FRAME')
            self.end_headers()
            try:
                while True:
                    if not self.server.MJPEGQueue.empty():
                        frame = self.server.MJPEGQueue.get()
                        ret, buf = cv2.imencode('.jpg', frame)
                        frame = np.array(buf).tostring()
                        self.wfile.write(b'-FRAME\r\n')
                        self.send_header('Content-Type', 'image/jpeg')
                        self.send_header('Content-Length', len(frame))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b'\r\r')
            except Exception as e:
                logging.warning('Removed streaming clients %s: %s', self.client_address, str(e))
        else:
            self.send_error(404)
            self.end_headers()

class StreamingServer(socketserver.ThreadingMixIn, server.HTTPServer):
    allow_reuse_address = True
    daemon_threads = True

def server_start(frameQueue, exit_key):
    try:
        address = ('', 8000)
        server = StreamingServer(address, StreamingHandler)
        server.MJPEGQueue = frameQueue
        print('Started server')
        server.serve_forever()
    finally:
        # Release handle to the webcam
        exit_key.set()

if __name__ == '__main__':
    multiprocessing.set_start_method('forkserver')
    prog_stop = multiprocessing.Event()
    prog_stop.clear()
    prog_stop1 = multiprocessing.Event()
    prog_stop1.clear()
    recImage = multiprocessing.Queue(2)
    resultRecogn = multiprocessing.Queue(2)
    persBuffer = multiprocessing.Queue(2)
    MJPEGQueue = multiprocessing.Queue(10)
    camProc = Process(target=camThread, args=(recImage, resultRecogn,MJPEGQueue, persBuffer,  prog_stop), daemon=True)
    camProc.start()
    frecogn = Process(target=recognition, args=(recImage, resultRecogn, persBuffer, prog_stop), daemon=True)
    frecogn.start()
    serverProc = Process(target=server_start, args=(MJPEGQueue, prog_stop1), daemon=True)
    serverProc.start()

    while True:
        if prog_stop.is_set():
            camProc.terminate()
            frecogn.terminate()
            serverProc.terminate()
            break
        sleep(1)
    cv2.destroyAllWindows()
