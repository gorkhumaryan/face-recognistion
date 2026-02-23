from pathlib import Path
import shutil

from sklearn.model_selection import train_test_split

def copy_split(X, y, root: Path):
    for p, person in zip(X, y):
        (root / person).mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, root / person / p.name)

def main():
    data_dir = Path("data/Faces/Faces")
    out_dir = Path("data/split")

    if out_dir.exists():
        shutil.rmtree(out_dir)

    train_dir, val_dir, test_dir = out_dir/"train", out_dir/"val", out_dir/"test"
    for d in (train_dir, val_dir, test_dir):
        d.mkdir(parents=True, exist_ok=True)

    files = sorted(data_dir.glob("*.jpg"))
    labels = [p.stem.split("_")[0] for p in files]

    X_train, X_temp, y_train, y_temp = train_test_split(
        files, labels, test_size=0.2, random_state=42, stratify=labels
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    copy_split(X_train, y_train, train_dir)
    copy_split(X_val, y_val, val_dir)
    copy_split(X_test, y_test, test_dir)

    print("Total:", len(files), "People:", len(set(labels)))
    print("Train:", len(X_train), "Val:", len(X_val), "Test:", len(X_test))

if __name__ == "__main__":
    main()