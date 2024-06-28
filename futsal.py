import cv2
import numpy as np
import pytesseract
import torch
from deep_sort_realtime.deepsort_tracker import DeepSort
import tkinter as tk
from tkinter import ttk, filedialog
import threading
import mysql.connector
from dotenv import load_dotenv
import os

class FutsalTracker:
    def __init__(self):
        # Charger les variables d'environnement à partir du fichier .env
        load_dotenv()
        mysql_host = os.getenv('MYSQL_HOST')
        mysql_user = os.getenv('MYSQL_USER')
        mysql_password = os.getenv('MYSQL_PASSWORD')
        mysql_db = os.getenv('MYSQL_DB')

        self.cap = None
        self.model = self.load_model()
        self.tracker = DeepSort(max_age=30, n_init=3)
        self.players_stats = {}
        self.field_width = 40.0  # meters
        self.field_height = 20.0  # meters
        self.frame_width = None
        self.frame_height = None

        # Connexion à MySQL
        self.conn = mysql.connector.connect(
            host=mysql_host,
            user=mysql_user,
            password=mysql_password,
            database=mysql_db
        )
        self.cursor = self.conn.cursor()

        self.create_gui()

    def load_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        return model

    def detect_objects(self, frame):
        results = self.model(frame)
        detections = results.pandas().xyxy[0].to_dict(orient="records")
        return detections

    def annotate_frame(self, frame, detections, tracks):
        for track in tracks:
            bbox = track.to_tlbr()
            box = [bbox[1] / self.frame_height, bbox[0] / self.frame_width, bbox[3] / self.frame_height, bbox[2] / self.frame_width]
            player_num = self.extract_player_number(frame, box)
            if player_num:
                player_id = track.track_id
                if player_id not in self.players_stats:
                    self.players_stats[player_id] = {'number': player_num, 'goals': 0, 'passes': 0, 'touches': 0, 'distance': 0, 'sprints': 0}

                x, y, w, h = bbox.astype(int)
                self.update_player_stats(player_id, x, y)
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(frame, f'#{player_num}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

        return frame

    def extract_player_number(self, frame, box):
        ymin, xmin, ymax, xmax = box
        xmin, xmax, ymin, ymax = int(xmin * frame.shape[1]), int(xmax * frame.shape[1]), int(ymin * frame.shape[0]), int(ymax * frame.shape[0])
        roi = frame[ymin:ymax, xmin:xmax]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        player_num = pytesseract.image_to_string(gray, config='--psm 7').strip()
        return player_num if player_num.isdigit() else None

    def update_player_stats(self, player_id, x, y):
        stats = self.players_stats[player_id]
        last_position = stats.get('last_position', (x, y))
        distance_moved = np.sqrt((x - last_position[0]) ** 2 + (y - last_position[1]) ** 2)
        stats['distance'] += distance_moved / self.frame_width * self.field_width  # Convert to meters
        stats['last_position'] = (x, y)

        # Simple heuristics for touches and sprints detection
        if distance_moved > 5:
            stats['touches'] += 1
        if distance_moved > 15:
            stats['sprints'] += 1

    def convert_detections_to_bboxes(self, detections):
        bboxes = []
        for det in detections:
            if det['confidence'] < 0.5 or det['class'] != 0:  # 0 corresponds to 'person' class in COCO
                continue
            xmin, ymin = det['xmin'], det['ymin']
            width, height = det['xmax'] - det['xmin'], det['ymax'] - det['ymin']
            confidence = det['confidence']
            # Ajouter un vecteur de caractéristiques factice pour chaque détection
            feature_vector = np.random.rand(128).tolist()  # ou utilisez une extraction de caractéristiques si disponible
            bbox = [xmin, ymin, width, height]
            bboxes.append((bbox, confidence, feature_vector))
        return bboxes

    def save_to_db(self):
        for player_id, stats in self.players_stats.items():
            self.cursor.execute(
                "INSERT INTO player_stats (number, goals, passes, touches, distance, sprints) VALUES (%s, %s, %s, %s, %s, %s)",
                (stats['number'], stats['goals'], stats['passes'], stats['touches'], stats['distance'], stats['sprints'])
            )
        self.conn.commit()

    def create_gui(self):
        self.root = tk.Tk()
        self.root.title("Futsal Tracker")

        self.tree = ttk.Treeview(self.root, columns=("number", "goals", "passes", "touches", "distance", "sprints"), show='headings')
        self.tree.heading("number", text="Player Number")
        self.tree.heading("goals", text="Goals")
        self.tree.heading("passes", text="Passes")
        self.tree.heading("touches", text="Touches")
        self.tree.heading("distance", text="Distance (meters)")
        self.tree.heading("sprints", text="Sprints")
        self.tree.pack(fill=tk.BOTH, expand=True)

        self.button_save = ttk.Button(self.root, text="Save to Database", command=self.save_to_db)
        self.button_save.pack()

        self.button_load_video = ttk.Button(self.root, text="Load Video", command=self.load_video)
        self.button_load_video.pack()

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.update_gui()

    def update_gui(self):
        for item in self.tree.get_children():
            self.tree.delete(item)

        for player_id, stats in self.players_stats.items():
            self.tree.insert("", tk.END, values=(
                stats['number'],
                stats['goals'],
                stats['passes'],
                stats['touches'],
                stats['distance'],
                stats['sprints']
            ))

        self.root.after(1000, self.update_gui)

    def load_video(self):
        video_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
        if video_path:
            self.cap = cv2.VideoCapture(video_path)
            self.frame_width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
            self.frame_height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
            self.run_in_thread()

    def run_in_thread(self):
        if self.cap is not None:
            threading.Thread(target=self.run, daemon=True).start()

    def on_closing(self):
        if self.cap:
            self.cap.release()
        self.conn.close()
        self.root.destroy()

    def run(self):
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                break

            detections = self.detect_objects(frame)
            bboxes = self.convert_detections_to_bboxes(detections)
            tracks = self.tracker.update_tracks(bboxes, frame)  # Passe le frame original et non un array vide
            annotated_frame = self.annotate_frame(frame, detections, tracks)

            cv2.imshow('Futsal Tracker', annotated_frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    tracker = FutsalTracker()
    tracker.root.mainloop()