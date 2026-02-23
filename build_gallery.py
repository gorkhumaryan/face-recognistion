from pathlib import Path
import numpy as np
from tqdm import tqdm

from face_embedder import FaceEmbedder

def main():
    train_dir = Path("data/split/train")
    out_path = Path("model")
    out_path.mkdir(exist_ok=True)
    gallery_file = out_path / "gallery.npz"

    embedder = FaceEmbedder()

    unknown_ids = {
    "Alia Bhatt",
    "Camila Cabello",
    "Hugh Jackman",
    "Marmik",
    "Virat Kohli"
}

    names = []
    templates = []

    for person_dir in tqdm(sorted([p for p in train_dir.iterdir() if p.is_dir()]),
                           desc="Building gallery"):
        if person_dir.name in unknown_ids:
            continue
        embs = []
        for img_path in person_dir.glob("*.jpg"):
            try:
                embs.append(embedder.embed_one(img_path))
            except Exception:
                pass

        if not embs:
            continue

        template = np.mean(np.stack(embs), axis=0)
        template /= (np.linalg.norm(template) + 1e-12)

        names.append(person_dir.name)
        templates.append(template)

    names = np.array(names, dtype=object)
    templates = np.stack(templates)

    np.savez(gallery_file, names=names, templates=templates)
    print("Saved:", gallery_file)
    print("Gallery identities:", len(names), "| template shape:", templates.shape)

if __name__ == "__main__":
    main()