from pathlib import Path
import numpy as np

from face_embedder import FaceEmbedder

def main():
    
    unknown_ids = {
    "Alia Bhatt",
    "Camila Cabello",
    "Hugh Jackman",
    "Marmik",
    "Virat Kohli"}

    thr = 0.48
    val_dir = Path("data/split/val")
    test_dir = Path("data/split/test")

    data = np.load("model/gallery.npz", allow_pickle=True)
    names = data["names"]
    templates = data["templates"]

    embedder = FaceEmbedder()

    def run_split(root_dir: Path):
        embs = []
        labels = []
        for person_dir in sorted([p for p in root_dir.iterdir() if p.is_dir()]):
            true_name = person_dir.name
            for img_path in person_dir.glob("*.jpg"):
                try:
                    embs.append(embedder.embed_one(img_path))
                    labels.append(true_name)
                except Exception:
                    pass
        embs = np.stack(embs)
        labels = np.array(labels)

        sims = embs @ templates.T

        known_total = known_correct = 0
        unk_total = unk_correct = 0

        for i, true_name in enumerate(labels):
            best_id = int(np.argmax(sims[i]))
            best_score = float(sims[i, best_id])
            pred = "Unknown" if best_score < thr else str(names[best_id])

            if true_name in unknown_ids:
                unk_total += 1
                if pred == "Unknown":
                    unk_correct += 1
            else:
                known_total += 1
                if pred == true_name:
                    known_correct += 1

        known_acc = known_correct / known_total if known_total else 0.0
        unk_acc = unk_correct / unk_total if unk_total else 0.0
        return known_acc, unk_acc, known_total, unk_total

    val_known, val_unk, _, _ = run_split(val_dir)
    test_known, test_unk, kt, ut = run_split(test_dir)

    print("VAL known acc:", val_known, "VAL unknown detect:", val_unk)
    print("TEST known acc:", test_known, "on", kt, "images")
    print("TEST unknown detect:", test_unk, "on", ut, "images")

if __name__ == "__main__":
    main()