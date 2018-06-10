import face_recognition
import cv2
import os
import sys

class VideoProcesser():
    def __init__(self, framerate=2):
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_max_id = 0
        self.frame_count = framerate
        self.framerate = framerate
        self.face_locations = []
        self.face_names = []
        self.face_encoding_distance = 0
        
    def add(self, dir_path, im_name):
        im = face_recognition.load_image_file(dir_path+'/'+im_name)
        self.known_face_encodings.append(face_recognition.face_encodings(im)[0])
        self.known_face_names.append(im_name.split('.')[0])
    
    def process_frame(self, frame, sens=0.65, scale=2):
        small_frame = cv2.resize(frame, (0, 0), fx=1/scale, fy=1/scale)

        rgb_small_frame = small_frame[:, :, ::-1]
        
        if self.frame_count % self.framerate == 0:
            self.face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, self.face_locations)

            self.face_names = []

            for face_encoding in face_encodings:
                if len(self.known_face_encodings) == 0:
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_names.append("Id:"+str(self.face_max_id))
                    self.face_names.append("Id:"+str(self.face_max_id))
                    self.face_max_id += 1
                    continue
                else:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding, sens)

                if True in matches:
                    first_match_index = matches.index(True)
                    name = self.known_face_names[first_match_index]
                else:
                    self.known_face_encodings.append(face_encoding)
                    self.known_face_names.append("Id:"+str(self.face_max_id))
                    name = str(self.face_max_id)
                    self.face_max_id += 1
                    first_match_index = self.face_max_id - 1

                self.face_names.append(name)

                face_distance = face_recognition.face_distance([self.known_face_encodings[first_match_index]], face_encoding)
                self.face_encoding_distance = (1 - face_distance[0]) * 100
                
        self.frame_count += 1
        for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
            top *= scale
            right *= scale
            bottom *= scale
            left *= scale

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name + " " + str(self.face_encoding_distance)[0:2] + "%", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
            
        return frame
    
    
def main():
    print("Enter video file [Webcam]: ", end='')
    sys.stdout.flush()
    file = sys.stdin.readline()
    if file == "\n":
        file = 0 #if webcam doesn't work, change to 1
    else:
        file = file[:-1]
    video_capture = cv2.VideoCapture(file)
    
    print("Enter sensitivity [0.65]: ", end='')
    sys.stdout.flush()
    sens = sys.stdin.readline()
    if sens == "\n":
        sens=0.65
    else:
        sens = float(sens)
    
    print("Enter scale [2]: ", end='')
    sys.stdout.flush()
    scale = sys.stdin.readline()
    if scale=="\n":
        scale = 2
    else:
        scale = int(scale)
        
    print("Enter framerate [2]: ", end='')
    sys.stdout.flush()
    framerate = sys.stdin.readline()
    if framerate=="\n":
        framerate = 2
    else:
        framerate = int(framerate)
    proc = VideoProcesser(framerate)
    
    print("Preload images from directory [Don't]: ", end='')
    sys.stdout.flush()
    im_dir = sys.stdin.readline()
    if im_dir != "\n":
        formats = ["jpg", "png"]
        im_dir = im_dir[:-1]
        files = os.listdir(im_dir)
        for file in files:
            if file.split('.')[-1] in formats:
                proc.add(im_dir, file)
                print("Loaded:", file)
            sys.stdout.flush()

    print("Press <q> to exit.")
    sys.stdout.flush()
    
    while True:
        ret, frame = video_capture.read()
        frame = proc.process_frame(frame, sens, scale)
        cv2.imshow('Video', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
