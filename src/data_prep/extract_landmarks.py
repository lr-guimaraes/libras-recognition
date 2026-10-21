"""
extract_landmarks.py

Lê data/annotations/videos_annotations.csv, roda o MediaPipe Holistic
(mp.solutions, mediapipe==0.10.21) em cada vídeo e salva a sequência de
landmarks (pose + rosto + mãos) como um array .npy por vídeo.

RETOMÁVEL: pode ser interrompido (Ctrl+C, queda de energia, terminal fechado)
e, ao rodar de novo, continua de onde parou:
  - cada .npy é gravado de forma atômica (arquivo temporário -> renomeia), então
    nunca fica um arquivo pela metade com cara de "pronto";
  - os .npy já existentes são VALIDADOS (formato e tamanho); os corrompidos são refeitos;
  - um vídeo com erro não derruba a execução: é registrado e a extração segue.

Opções:  --force (refaz tudo)   --limit N (processa só N vídeos, para testar)

Saída:
data/processed/landmarks/<label>_<repeticao>.npy
  shape: (num_frames, 1629)  = (33 pose + 468 rosto + 21 mão esq + 21 mão dir) * (x, y, z)
  Pontos não detectados em um frame ficam zerados.

Requer: pip install mediapipe==0.10.21 opencv-python numpy
(Alternativa com a Tasks API: extract_landmarks_tasks.py)
"""

import os
import csv
import sys
import time
import argparse
import numpy as np
import cv2
import mediapipe as mp

def _find_base_dir():
    """Acha a raiz do projeto (pasta que contém data/raw ou data/processed), de qualquer pasta onde o script rodar."""
    env = os.environ.get("LIBRAS_BASE_DIR")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    for start in (here, os.getcwd(), os.path.join(os.getcwd(), "libras-recognition")):
        d = start
        while True:
            if os.path.isdir(os.path.join(d, "data", "raw")) or os.path.isdir(os.path.join(d, "data", "processed")):
                return d
            parent = os.path.dirname(d)
            if parent == d:
                break
            d = parent
    return os.path.join(os.getcwd(), "libras-recognition")


BASE_DIR = _find_base_dir()
ANNOTATIONS_CSV = os.path.join(BASE_DIR, "data", "annotations", "videos_annotations.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "processed", "landmarks")

mp_holistic = mp.solutions.holistic

N_POSE, N_FACE, N_HAND = 33, 468, 21
RAW_DIM = (N_POSE + N_FACE + 2 * N_HAND) * 3  # 1629


def resolve_path(p):
    """Aceita caminho absoluto ou relativo (CSVs antigos guardavam 'libras-recognition/data/...')."""
    p = os.path.normpath(p)
    if os.path.isfile(p):
        return p
    for root in (BASE_DIR, os.path.dirname(BASE_DIR)):
        cand = os.path.join(root, p)
        if os.path.isfile(cand):
            return cand
    return p


def extract_keypoints(results):
    """Vetor 1D fixo; zeros onde a parte do corpo não foi detectada no frame."""

    def to_array(landmarks, n_points):
        if landmarks is None:
            return np.zeros(n_points * 3)
        return np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark]).flatten()

    return np.concatenate([
        to_array(results.pose_landmarks, N_POSE),
        to_array(results.face_landmarks, N_FACE),
        to_array(results.left_hand_landmarks, N_HAND),
        to_array(results.right_hand_landmarks, N_HAND),
    ])


def process_video(video_path, holistic):
    cap = cv2.VideoCapture(video_path)
    sequence = []
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        sequence.append(extract_keypoints(holistic.process(frame_rgb)))
    cap.release()
    return np.array(sequence, dtype=np.float32)


def is_valid_npy(path):
    """True só se o .npy está completo: cabeçalho legível, 1629 colunas e tamanho em bytes batendo com o esperado."""
    try:
        with open(path, "rb") as f:
            major, _ = np.lib.format.read_magic(f)
            reader = np.lib.format.read_array_header_1_0 if major == 1 else np.lib.format.read_array_header_2_0
            shape, _, dtype = reader(f)
            data_start = f.tell()
        expected = data_start + int(np.prod(shape)) * dtype.itemsize
        return len(shape) == 2 and shape[0] > 0 and shape[1] == RAW_DIM and os.path.getsize(path) == expected
    except Exception:
        return False


def save_atomic(path, array):
    """Grava em arquivo temporário e só então renomeia: ou o .npy fica completo, ou não existe."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        np.save(f, array)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def fmt_time(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}h{seconds % 3600 // 60:02d}m" if seconds >= 3600 else f"{seconds // 60}m{seconds % 60:02d}s"


def main():
    ap = argparse.ArgumentParser(description="Extrai landmarks dos vídeos (retomável)")
    ap.add_argument("--force", action="store_true", help="refaz todos os vídeos, mesmo os já extraídos")
    ap.add_argument("--limit", type=int, default=0, help="processa no máximo N vídeos nesta execução")
    args = ap.parse_known_args()[0]

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.isfile(ANNOTATIONS_CSV):
        print(f"[erro] não encontrei {ANNOTATIONS_CSV}. Rode generate_annotations.py antes.")
        sys.exit(1)

    for name in os.listdir(OUTPUT_DIR):  # restos de uma execução interrompida
        if name.endswith(".tmp"):
            os.remove(os.path.join(OUTPUT_DIR, name))
            print(f"[limpeza] temporário removido: {name}")

    with open(ANNOTATIONS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    todo, done = [], 0
    for row in rows:
        out_name = f"{row['label']}_{row['repetition']}.npy"
        out_path = os.path.join(OUTPUT_DIR, out_name)
        if not args.force and os.path.exists(out_path):
            if is_valid_npy(out_path):
                done += 1
                continue
            print(f"[refazendo] arquivo incompleto/corrompido: {out_name}")
        todo.append((row, out_name, out_path))
    if args.limit:
        todo = todo[:args.limit]

    print(f"{len(rows)} vídeos na anotação | já concluídos: {done} | a processar agora: {len(todo)}")
    print(f"saída: {OUTPUT_DIR}")
    if not todo:
        print("[ok] nada a fazer: todos os vídeos já foram extraídos.")
        return

    errors, processed, t0 = [], 0, time.time()
    try:
        for i, (row, out_name, out_path) in enumerate(todo, 1):
            video_path = resolve_path(row["video_path"])
            tag = f"[{i}/{len(todo)}]"
            if not os.path.isfile(video_path):
                errors.append((out_name, "vídeo não encontrado"))
                print(f"{tag}[aviso] vídeo não encontrado: {video_path}")
                continue
            try:
                # um Holistic novo por vídeo: o tracking não "vaza" do fim de um vídeo para o início do próximo
                with mp_holistic.Holistic(static_image_mode=False, model_complexity=1,
                                          min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
                    sequence = process_video(video_path, holistic)
                if sequence.size == 0:
                    raise ValueError("nenhum frame lido")
                save_atomic(out_path, sequence)
            except Exception as e:  # um vídeo ruim não pode derrubar os outros
                errors.append((out_name, f"{type(e).__name__}: {e}"))
                print(f"{tag}[erro] {out_name}: {type(e).__name__}: {e}")
                continue

            processed += 1
            hands = int((np.abs(sequence[:, 1503:]).sum(1) > 0).sum())
            eta = (time.time() - t0) / processed * (len(todo) - i)
            print(f"{tag}[ok] {out_name} -> {sequence.shape} | mão detectada: {hands}/{len(sequence)} | restante ~{fmt_time(eta)}")
    except KeyboardInterrupt:
        print(f"\n[interrompido] {processed} vídeos salvos nesta execução. Rode o mesmo comando para continuar de onde parou.")
        sys.exit(130)

    print(f"\n[fim] {processed} extraídos nesta execução | {done + processed}/{len(rows)} no total | erros: {len(errors)} | tempo: {fmt_time(time.time() - t0)}")
    if errors:
        err_path = os.path.join(os.path.dirname(OUTPUT_DIR), "extraction_errors.txt")
        with open(err_path, "w", encoding="utf-8") as f:
            f.writelines(f"{n}\t{msg}\n" for n, msg in errors)
        print(f"[aviso] lista de erros em {err_path}. Rode de novo para tentar apenas esses.")
        sys.exit(1)


if __name__ == "__main__":
    main()