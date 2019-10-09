# Jetson-Nano-FaceRecognition
Face recognition on Jetson Nano (Python version)

PC-version
Usage:
1. Run python3 get_face_person.py for every person. Pesron's photos must be placed in unique directory. Directory name = person's name.
2. Run python3 face_train.py for generating database with face encodings of known persons.
3. Run python3 face_detect.py for real-time face recognition and persons identification.
Requirements:
PC-version uses Google Coral USB Accelerator. Default camera used is /dev/video0.
