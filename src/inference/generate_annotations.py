"""
generate_annotations.py

Gera os CSVs de anotação a partir da estrutura de pastas:

data/raw/videos/<sinal>_<repeticao>.mp4      -> ex: oi_1.mp4, oi_2.mp4, oi_3.mp4
data/raw/alphabet_images/<LETRA>/*.jpg|png   -> ex: A/a_01.jpg

Saída:
data/annotations/videos_annotations.csv
data/annotations/alphabet_annotations.csv
data/annotations/labels_map.json
"""

import os
import re
import json
import csv
import cv2

BASE_DIR = "libras-recognition"
VIDEOS_DIR = os.path.join(BASE_DIR, "data", "raw", "videos")
ALPHABET_DIR = os.path.join(BASE_DIR, "data", "raw", "alphabet_images")
ANNOTATIONS_DIR = os.path.join(BASE_DIR, "data", "annotations")

VIDEO_PATTERN = re.compile(r"^(.+)(\d+)\.\w+$")


def get_video_metadata(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration = round(num_frames / fps, 2) if fps else 0
    cap.release()
    return fps, num_frames, duration


def generate_video_annotations():
    rows = []
    labels = set()

    if not os.path.isdir(VIDEOS_DIR):
        print(f"[aviso] pasta não encontrada: {VIDEOS_DIR}")
        return rows, labels

    for filename in sorted(os.listdir(VIDEOS_DIR)):
        match = VIDEO_PATTERN.match(filename)
        if not match:
            print(f"[aviso] nome fora do padrão, ignorado: {filename}")
            continue

        label, repetition = match.group(1), int(match.group(2))
        full_path = os.path.join(VIDEOS_DIR, filename)
        fps, num_frames, duration = get_video_metadata(full_path)

        rows.append({
            "video_path": full_path.replace("\\", "/"),
            "label": label,
            "repetition": repetition,
            "duration_sec": duration,
            "fps": fps,
            "num_frames": num_frames,
        })
        labels.add(label)

    return rows, labels


def generate_alphabet_annotations():
    rows = []

    if not os.path.isdir(ALPHABET_DIR):
        print(f"[aviso] pasta não encontrada: {ALPHABET_DIR}")
        return rows

    for letter in sorted(os.listdir(ALPHABET_DIR)):
        letter_dir = os.path.join(ALPHABET_DIR, letter)
        if not os.path.isdir(letter_dir):
            continue

        for filename in sorted(os.listdir(letter_dir)):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            full_path = os.path.join(letter_dir, filename)
            rows.append({
                "image_path": full_path.replace("\\", "/"),
                "label": letter,
                "source": "alphabet_images",
            })

    return rows


def write_csv(rows, path, fieldnames):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] {path} ({len(rows)} linhas)")


def main():
    video_rows, video_labels = generate_video_annotations()
    write_csv(
        video_rows,
        os.path.join(ANNOTATIONS_DIR, "videos_annotations.csv"),
        ["video_path", "label", "repetition", "duration_sec", "fps", "num_frames"],
    )

    alphabet_rows = generate_alphabet_annotations()
    write_csv(
        alphabet_rows,
        os.path.join(ANNOTATIONS_DIR, "alphabet_annotations.csv"),
        ["image_path", "label", "source"],
    )

    all_labels = sorted(video_labels) + [chr(c) for c in range(ord("A"), ord("Z") + 1)]
    labels_map = {label: idx for idx, label in enumerate(sorted(set(all_labels)))}

    labels_map_path = os.path.join(ANNOTATIONS_DIR, "labels_map.json")
    with open(labels_map_path, "w", encoding="utf-8") as f:
        json.dump(labels_map, f, ensure_ascii=False, indent=2)
    print(f"[ok] {labels_map_path} ({len(labels_map)} classes)")


if __name__ == "__main__":
    main()