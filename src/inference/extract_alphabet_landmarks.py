"""
extract_alphabet_landmarks.py

Lê data/annotations/alphabet_annotations.csv, roda MediaPipe Hands
em cada imagem estática (letras do alfabeto) e salva os landmarks
de mão normalizados em um único CSV, pronto para treinar um
classificador simples (ex: MLP ou RandomForest).

Saída:
data/processed/alphabet_landmarks/alphabet_landmarks.csv
  colunas: label, x0, y0, z0, x1, y1, z1, ..., x20, y20, z20

Requer: pip install mediapipe opencv-python
"""

import os
import csv
import cv2
import mediapipe as mp

BASE_DIR = "libras-recognition"
ANNOTATIONS_CSV = os.path.join(BASE_DIR, "data", "annotations", "alphabet_annotations.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed", "alphabet_landmarks")
OUTPUT_CSV = os.path.join(OUTPUT_DIR, "alphabet_landmarks.csv")

mp_hands = mp.solutions.hands

N_HAND_POINTS = 21


def extract_hand_landmarks(image_path, hands):
    image = cv2.imread(image_path)
    if image is None:
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    if not results.multi_hand_landmarks:
        return None

    # usa a primeira mão detectada
    landmarks = results.multi_hand_landmarks[0]
    coords = []
    for lm in landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])

    return coords


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isfile(ANNOTATIONS_CSV):
        print(f"[erro] não encontrei {ANNOTATIONS_CSV}. Rode generate_annotations.py antes.")
        return

    with open(ANNOTATIONS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    header = ["label"]
    for i in range(N_HAND_POINTS):
        header += [f"x{i}", f"y{i}", f"z{i}"]

    written = 0
    skipped = 0

    with mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5,
    ) as hands, open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as out_f:

        writer = csv.writer(out_f)
        writer.writerow(header)

        for row in rows:
            image_path = row["image_path"]
            label = row["label"]

            if not os.path.isfile(image_path):
                print(f"[aviso] imagem não encontrada: {image_path}")
                skipped += 1
                continue

            coords = extract_hand_landmarks(image_path, hands)
            if coords is None:
                print(f"[aviso] mão não detectada: {image_path}")
                skipped += 1
                continue

            writer.writerow([label] + coords)
            written += 1

    print(f"[ok] {OUTPUT_CSV} -> {written} linhas escritas, {skipped} ignoradas")


if __name__ == "__main__":
    main()