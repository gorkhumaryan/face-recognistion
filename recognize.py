from pathlib import Path
import numpy as np

from face_embedder import FaceEmbedder

class Recognizer:
    def __init__(self, gallery_path: str = "model/gallery.npz"):
        data = np.load(gallery_path, allow_pickle=True)
        self.names = data["names"]
        self.templates = data["templates"]
        self.embedder = FaceEmbedder()

    def recognize(self, img_path: str, threshold: float = 0.48):
        q = self.embedder.embed_one(Path(img_path))
        sims = self.templates @ q
        best_id = int(np.argmax(sims))
        best_score = float(sims[best_id])

        if best_score < threshold:
            return "Unknown", best_score
        return str(self.names[best_id]), best_score

if __name__ == "__main__":
    r = Recognizer()
    name, score = r.recognize("data/Faces/Faces/Amitabh Bachchan_44.jpg", threshold=0.48)
    print("Prediction:", name)
    print("Similarity:", score)