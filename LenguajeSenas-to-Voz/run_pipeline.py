

import sys, cv2, numpy as np, os, traceback
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from PyQt5.QtGui      import QImage, QPixmap
from PyQt5.QtCore     import QTimer, Qt
from PyQt5.uic        import loadUi
import sys, cv2, numpy as np, os, traceback, pathlib

ROOT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
from tensorflow.keras.models import load_model
from mediapipe.python.solutions.holistic import Holistic
from src import helpers, constants, text_to_speech
from src.helpers   import (mediapipe_detection, extract_keypoints, there_hand,
                       get_word_ids, draw_keypoints, words_text)
from src.constants import *
from src.text_to_speech import text_to_speech

# ---------- utilidades -----------------------------------------------------
def interpolate_or_sample(frames, target=MODEL_FRAMES):
    n = len(frames)
    idx = np.linspace(0, n - 1, target)
    out = []
    for i in idx:
        lo, hi = int(np.floor(i)), int(np.ceil(i))
        w = i - lo
        if lo == hi:
            out.append(frames[lo])
        else:
            out.append(cv2.addWeighted(frames[lo], 1 - w, frames[hi], w, 0))
    return out

def seq_to_keypoints(frames, holistic):
    kp_seq = []
    for f in frames:
        res = mediapipe_detection(f, holistic)
        kp_seq.append(extract_keypoints(res))
    return np.array(kp_seq, dtype="float32")

# ---------- ventana --------------------------------------------------------
class VideoRecorder(QMainWindow):
    def __init__(self):
        super().__init__()
        try:
            loadUi('src\mainwindow.ui', self)
        except Exception:
            QMessageBox.critical(self, "UI", "No se pudo cargar mainwindow.ui")
            sys.exit(1)

        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            QMessageBox.critical(self, "Cam", "No se encontró webcam")
            sys.exit(1)

        self.holistic = Holistic(model_complexity=1)
        self.model    = load_model(MODEL_PATH)
        self.word_ids = get_word_ids(WORDS_JSON_PATH)
        print("word_ids:", self.word_ids)

        self.frames_buf, self.no_hand = [], 0
        self.MIN_FRAMES, self.NO_HAND_MAX = 10, 6

        self.timer = QTimer(self, interval=30, timeout=self.loop)
        self.timer.start()

    # -------------------------------------------------------------------
    def loop(self):
        ok, frame = self.cap.read()
        if not ok:
            return
        try:
            self.process_frame(frame)
        except Exception as e:
            # logea sin cerrar la app
            print("\n===== EXCEPCIÓN EN LOOP =====")
            traceback.print_exc()
            print("===== CONTINÚA RUN =====\n")

    # -------------------------------------------------------------------
    def process_frame(self, frame_bgr):
        res = mediapipe_detection(frame_bgr, self.holistic)
        hand = there_hand(res)

        if hand:
            self.frames_buf.append(frame_bgr.copy())
            self.no_hand = 0
        else:
            if self.frames_buf:
                self.no_hand += 1
                if self.no_hand >= self.NO_HAND_MAX:
                    self.handle_gesture()
                    self.frames_buf.clear()
                    self.no_hand = 0

        # draw
        vis = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        draw_keypoints(vis, res)
        pix = QPixmap.fromImage(
            QImage(vis.data, vis.shape[1], vis.shape[0],
                   vis.shape[1]*3, QImage.Format_RGB888)
            .scaled(self.lbl_video.size(), Qt.KeepAspectRatio,
                    Qt.SmoothTransformation)
        )
        self.lbl_video.setPixmap(pix)

    # -------------------------------------------------------------------
    def handle_gesture(self):
        n = len(self.frames_buf)
        if n < self.MIN_FRAMES:
            print(f"gesto descartado ({n} frames)")
            return

        print(f"gesto capturado ({n} frames)")
        frames15 = interpolate_or_sample(self.frames_buf)
        kp_seq   = seq_to_keypoints(frames15, self.holistic)
        pred     = self.model.predict(kp_seq[None, ...], verbose=0)[0]
        print({w: f"{p:.2f}" for w, p in zip(self.word_ids, pred)})

        if pred.max() > 0.6:
            wid  = self.word_ids[pred.argmax()].split('-')[0]
            sent = words_text.get(wid)
            self.lbl_output.setText(sent)
            text_to_speech(sent)
        else:
            print("- Confianza baja, sin voz")

    # -------------------------------------------------------------------
    def closeEvent(self, e):
        self.cap.release()
        self.holistic.close()
        e.accept()

# ---------- main -----------------------------------------------------------
if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
    app = QApplication(sys.argv)
    w = VideoRecorder()
    w.show()
    sys.exit(app.exec_())
