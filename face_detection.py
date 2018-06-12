import face_recognition
import cv2
import os
import sys
from moviepy.editor import *

 
class VideoProcesser():
    def __init__(self, framerate=2):
        self.known_face_encodings = []
        self.known_face_names = []
        self.face_max_id = 0
        self.frame_count = framerate
        self.framerate = framerate
        self.face_locations = []
        self.face_names = []
        self.face_encoding_distances = []
 
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
            self.face_encoding_distances = []
 
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
                    first_match_index = len(self.known_face_encodings) - 1
 
                self.face_names.append(name)
                face_distance = face_recognition.face_distance([self.known_face_encodings[first_match_index]], face_encoding)
                self.face_encoding_distances.append((1 - face_distance[0]) * 100)
 
        self.frame_count += 1
        for (top, right, bottom, left), name, face_encoding_distance in zip(self.face_locations, self.face_names, self.face_encoding_distances):
            top *= scale
            right *= scale
            bottom *= scale
            left *= scale
 
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
 
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
 
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name + " " + str(face_encoding_distance)[0:2] + "%", (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
 
        return frame


def stick_audio(input_file_name, output_file_name):
    input_video = VideoFileClip(input_file_name)
    output_video = VideoFileClip(output_file_name)
    audio = AudioFileClip(input_file_name)

    os.rename(output_file_name, "output_without_audio.avi")

    expected_duration = output_video.duration * output_video.fps / input_video.fps
    if expected_duration > input_video.duration:
        expected_duration = input_video.duration

    audio = vfx.speedx(clip=audio.subclip(0, expected_duration), final_duration=output_video.duration)

    output_video = output_video.set_audio(audio)
    output_video.write_videofile(output_file_name, codec='mpeg4', bitrate="14400k")
 
 
def main():
    print("Enter video file [Webcam]: ", end='')
    sys.stdout.flush()
    file = sys.stdin.readline()
    is_webcam = False
    if file == "\n":
        file = 0 # if webcam doesn't work, change it to 1
        is_webcam = True
    else:
        file = file[:-1]
    video_capture = cv2.VideoCapture(file)
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
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
        formats = ["jpg", "jpeg", "png"]
        im_dir = im_dir[:-1]
        files = os.listdir(im_dir)
        for file in files:
            if file.split('.')[-1] in formats:
                proc.add(im_dir, file)
                print("Loaded:", file)
            sys.stdout.flush()
 
    print("Record video [n]/y? ", end='')
    sys.stdout.flush()
    rec = sys.stdin.readline()
    if rec == "y\n":
        print("Enter fps [24]: ", end='')
        sys.stdout.flush()
        fps = sys.stdin.readline()
        if fps == "\n":
            fps = 24
        else:
            fps = int(fps)
        rec = True
        recorder = cv2.VideoWriter('output.avi', cv2.VideoWriter_fourcc('X','V','I','D'), fps, (frame_width, frame_height))
    else:
        rec = False
 
    print("Press <q> to exit.")
    sys.stdout.flush()
 
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if ret:
            frame = proc.process_frame(frame, sens, scale)
            cv2.imshow('Video', frame)
            if rec:
                recorder.write(frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
 
    video_capture.release()
    if rec:
        recorder.release()

    cv2.destroyAllWindows()

    if rec and not is_webcam:
        stick_audio(file, "output.avi")

 
if __name__ == "__main__":
    main()
