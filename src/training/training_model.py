#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
train_video_model.py
====================
Treina e testa o modelo de SINAIS (vídeo) de Libras a partir dos landmarks
gerados pelo extract_landmarks.py (MediaPipe Holistic).

Entrada : data/processed/landmarks/<sinal>_<repeticao>.npy   (shape: frames x 1629)
Saída   : models_output/video_model.pt  +  models_output/video_classes.json

Como usar (um único arquivo, vários modos):

  python train_video_model.py                      # treina + avalia (modo padrão)
  python train_video_model.py train --epochs 120   # treina com mais épocas
  python train_video_model.py train --final        # treina com TODAS as repetições (para uso real)
  python train_video_model.py eval                 # reavalia um modelo já treinado
  python train_video_model.py predict --video "C:\\caminho\\Pôr o anel_Articulador3.mp4"
  python train_video_model.py webcam               # legenda em tempo real pela webcam (q sai, c limpa)

Ideia central do modelo
-----------------------
As MÃOS e o ROSTO têm encoders próprios e maiores que o da pose:
  * mãos : formato dos dedos (relativo ao punho), posição no corpo, velocidade, flags de presença
  * rosto: só pontos com valor gramatical em Libras (sobrancelhas, olhos, boca) + âncoras de cabeça
  * pose : tronco/braços (referência de posição) em tamanho pequeno
Depois os três fluxos são fundidos e passam por uma BiGRU com pooling por atenção.
"""

import argparse
import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:  # acentos no terminal do Windows
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# --------------------------------------------------------------------------------------
# Layout do vetor bruto salvo pelo extract_landmarks.py: [pose 33 | rosto 468 | mão esq 21 | mão dir 21] x (x,y,z)
# --------------------------------------------------------------------------------------
N_POSE, N_FACE, N_HAND = 33, 468, 21
POSE_SL = slice(0, N_POSE * 3)
FACE_SL = slice(N_POSE * 3, (N_POSE + N_FACE) * 3)
LH_SL = slice((N_POSE + N_FACE) * 3, (N_POSE + N_FACE + N_HAND) * 3)
RH_SL = slice((N_POSE + N_FACE + N_HAND) * 3, (N_POSE + N_FACE + 2 * N_HAND) * 3)
RAW_DIM = (N_POSE + N_FACE + 2 * N_HAND) * 3  # 1629
LH_PTS = (N_POSE + N_FACE, N_POSE + N_FACE + N_HAND)  # posições na grade de 543 pontos
RH_PTS = (N_POSE + N_FACE + N_HAND, N_POSE + N_FACE + 2 * N_HAND)

# Pontos do rosto com função gramatical em Libras (índices do MediaPipe Face Mesh)
FACE_LIPS_OUTER = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
FACE_LIPS_INNER = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 191]
FACE_EYE_R = [33, 7, 163, 144, 145, 153, 154, 155, 133, 246, 161, 160, 159, 158, 157, 173]
FACE_EYE_L = [263, 249, 390, 373, 374, 380, 381, 382, 362, 466, 388, 387, 386, 385, 384, 398]
FACE_BROW_R = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
FACE_BROW_L = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
FACE_ANCHORS = [1, 10, 152, 234, 454]  # ponta do nariz, testa, queixo, laterais (inclinação/aceno de cabeça)
FACE_IDX = sorted(set(FACE_LIPS_OUTER + FACE_LIPS_INNER + FACE_EYE_R + FACE_EYE_L
                      + FACE_BROW_R + FACE_BROW_L + FACE_ANCHORS))
POSE_IDX = [0, 11, 12, 13, 14, 15, 16]  # nariz, ombros, cotovelos, punhos

HAND_DIM = 63 * 2 + 42 * 2 + 42 * 2 + 2      # formato(2 mãos) + posição + velocidade + flags
POSE_DIM = len(POSE_IDX) * 2 + 1
FACE_DIM = len(FACE_IDX) * 3 + 1
FEAT_DIM = HAND_DIM + POSE_DIM + FACE_DIM


# --------------------------------------------------------------------------------------
# Engenharia de features
# --------------------------------------------------------------------------------------
def split_blocks(raw):
    t = raw.shape[0]
    return (raw[:, POSE_SL].reshape(t, N_POSE, 3), raw[:, FACE_SL].reshape(t, N_FACE, 3),
            raw[:, LH_SL].reshape(t, N_HAND, 3), raw[:, RH_SL].reshape(t, N_HAND, 3))


def present(block):
    """True nos frames em que a parte do corpo foi detectada (o extrator grava zeros quando não detecta)."""
    return np.abs(block).sum(axis=(1, 2)) > 0


def hand_features(h, ok, sc, sw):
    t = h.shape[0]
    scale = np.maximum(np.linalg.norm(h[:, 9, :] - h[:, 0, :], axis=1), 1e-4)
    shape = ((h - h[:, :1, :]) / scale[:, None, None]).reshape(t, -1)          # formato da mão
    pos = ((h[:, :, :2] - sc[:, None, :2]) / sw[:, None, None]).reshape(t, -1)  # posição no corpo
    shape[~ok] = 0
    pos[~ok] = 0
    delta = np.zeros_like(pos)                                                  # velocidade
    if t > 1:
        both = ok[1:] & ok[:-1]
        delta[1:] = np.where(both[:, None], pos[1:] - pos[:-1], 0.0)
    return shape, pos, delta


def build_features(raw):
    """(T,1629) landmarks brutos -> (T,FEAT_DIM) features normalizadas (invariantes a posição/escala do corpo)."""
    raw = np.asarray(raw, dtype=np.float32)
    t = raw.shape[0]
    pose, face, lh, rh = split_blocks(raw)
    pose_ok, face_ok, lh_ok, rh_ok = (present(b) for b in (pose, face, lh, rh))

    # referência corporal: centro e largura dos ombros
    sc = ((pose[:, 11, :] + pose[:, 12, :]) / 2.0).astype(np.float32)
    sw = np.linalg.norm(pose[:, 11, :2] - pose[:, 12, :2], axis=1).astype(np.float32)
    bad = (~pose_ok) | (sw < 1e-3)
    if bad.all():
        sc[:] = (0.5, 0.5, 0.0)
        sw[:] = 0.25
    elif bad.any():
        sc[bad] = np.median(sc[~bad], axis=0)
        sw[bad] = np.median(sw[~bad])

    lh_s, lh_p, lh_d = hand_features(lh, lh_ok, sc, sw)
    rh_s, rh_p, rh_d = hand_features(rh, rh_ok, sc, sw)
    hand = np.concatenate([lh_s, rh_s, lh_p, rh_p, lh_d, rh_d,
                           lh_ok[:, None].astype(np.float32), rh_ok[:, None].astype(np.float32)], axis=1)

    pp = ((pose[:, POSE_IDX, :2] - sc[:, None, :2]) / sw[:, None, None]).reshape(t, -1)
    pp[~pose_ok] = 0
    posef = np.concatenate([pp, pose_ok[:, None].astype(np.float32)], axis=1)

    iod = np.maximum(np.linalg.norm(face[:, 33, :2] - face[:, 263, :2], axis=1), 1e-4)  # distância entre os olhos
    fn = ((face[:, FACE_IDX, :] - face[:, 1:2, :]) / iod[:, None, None]).reshape(t, -1)
    fn[~face_ok] = 0
    facef = np.concatenate([fn, face_ok[:, None].astype(np.float32)], axis=1)

    return np.concatenate([hand, posef, facef], axis=1).astype(np.float32)


def trim_to_activity(raw, margin=4, min_len=8):
    """Corta o início/fim do vídeo em que não há nenhuma mão detectada."""
    act = np.where((np.abs(raw[:, LH_SL]).sum(1) > 0) | (np.abs(raw[:, RH_SL]).sum(1) > 0))[0]
    if len(act) == 0:
        return raw
    a, b = max(0, act[0] - margin), min(len(raw), act[-1] + margin + 1)
    return raw[a:b] if b - a >= min_len else raw


def time_indices(n, T, rng=None):
    """Índices dos frames reamostrados para T frames. Com rng: corte aleatório + variação de velocidade."""
    if n <= 1:
        return np.zeros(T, dtype=int)
    if rng is None:
        return np.round(np.linspace(0, n - 1, T)).astype(int)
    start = rng.uniform(0, 0.12) * (n - 1)
    end = (n - 1) * (1 - rng.uniform(0, 0.12))
    u = np.linspace(0, 1, T) ** rng.uniform(0.8, 1.25)
    return np.clip(np.round(start + u * (end - start)), 0, n - 1).astype(int)


def augment_raw(seq, rng):
    """Augmentation espacial nos landmarks brutos (T,1629). Mantém zeros (= parte não detectada)."""
    t = seq.shape[0]
    pts = seq.reshape(t, -1, 3).astype(np.float32).copy()
    ok = np.abs(pts).sum(-1) > 0

    ang = np.deg2rad(rng.uniform(-10, 10))
    c, s = np.cos(ang), np.sin(ang)
    scl = rng.uniform(0.9, 1.1)
    sxy = np.array([scl * rng.uniform(0.95, 1.05), scl * rng.uniform(0.95, 1.05)])
    shift = rng.uniform(-0.05, 0.05, size=2)
    xy = pts[..., :2] - 0.5
    xy = np.stack([xy[..., 0] * c - xy[..., 1] * s, xy[..., 0] * s + xy[..., 1] * c], axis=-1)
    pts[..., :2] = xy * sxy + 0.5 + shift
    pts[..., 2] *= scl

    for a, b in (LH_PTS, RH_PTS):  # jitter independente de cada mão em torno do punho
        wrist = pts[:, a:a + 1, :2]
        ah = np.deg2rad(rng.uniform(-10, 10))
        ch, sh = np.cos(ah), np.sin(ah)
        rel = pts[:, a:b, :2] - wrist
        rot = np.stack([rel[..., 0] * ch - rel[..., 1] * sh, rel[..., 0] * sh + rel[..., 1] * ch], axis=-1)
        pts[:, a:b, :2] = wrist + rot * rng.uniform(0.92, 1.08)
        ok[rng.random(t) < 0.04, a:b] = False          # simula falhas de detecção da mão

    ok[rng.random(t) < 0.02, N_POSE:N_POSE + N_FACE] = False  # e, raramente, do rosto
    pts += rng.normal(0, 0.002, pts.shape).astype(np.float32)
    pts[~ok] = 0
    return pts.reshape(t, -1)


def make_input(raw, T, train, rng):
    raw = trim_to_activity(raw)
    raw = raw[time_indices(len(raw), T, rng if train else None)]
    if train:
        raw = augment_raw(raw, rng)
    return build_features(raw)


class SignDataset(Dataset):
    def __init__(self, items, T, mean, std, train=False, repeat=1, seed=0):
        self.items, self.T, self.mean, self.std = items, T, mean, std
        self.train, self.repeat = train, repeat
        self.rng = np.random.default_rng(seed)

    def __len__(self):
        return len(self.items) * self.repeat

    def __getitem__(self, i):
        raw, y = self.items[i % len(self.items)]
        x = (make_input(raw, self.T, self.train, self.rng) - self.mean) / self.std
        return torch.from_numpy(x.astype(np.float32)), int(y)


def compute_stats(items, T, seed=0):
    rng = np.random.default_rng(seed)
    feats = [make_input(raw, T, k > 0, rng) for k in range(4) for raw, _ in items]  # 1 limpa + 3 aumentadas
    feats = np.concatenate(feats, axis=0)
    return feats.mean(0).astype(np.float32), np.maximum(feats.std(0), 1e-2).astype(np.float32)


# --------------------------------------------------------------------------------------
# Modelo: encoders separados (mãos e rosto maiores que a pose) -> fusão -> BiGRU -> atenção
# --------------------------------------------------------------------------------------
class StreamEncoder(nn.Module):
    def __init__(self, in_dim, d, dropout):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(in_dim, d), nn.LayerNorm(d), nn.GELU(), nn.Dropout(dropout))
        self.conv = nn.Conv1d(d, d, kernel_size=5, padding=2)
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                                   # (B,T,in)
        h = self.proj(x)
        c = self.conv(h.transpose(1, 2)).transpose(1, 2)    # contexto temporal local
        return self.norm(h + self.drop(F.gelu(c)))


class SignNet(nn.Module):
    def __init__(self, n_classes, hidden=128, dropout=0.35):
        super().__init__()
        self.hand = StreamEncoder(HAND_DIM, 160, dropout)
        self.pose = StreamEncoder(POSE_DIM, 32, dropout)
        self.face = StreamEncoder(FACE_DIM, 96, dropout)
        self.fuse = nn.Sequential(nn.Linear(160 + 32 + 96, hidden * 2), nn.LayerNorm(hidden * 2),
                                  nn.GELU(), nn.Dropout(dropout))
        self.gru = nn.GRU(hidden * 2, hidden, batch_first=True, bidirectional=True)
        self.att = nn.Linear(hidden * 2, 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden * 4, n_classes))

    def forward(self, x):                                   # (B,T,FEAT_DIM)
        h = x[:, :, :HAND_DIM]
        p = x[:, :, HAND_DIM:HAND_DIM + POSE_DIM]
        f = x[:, :, HAND_DIM + POSE_DIM:]
        z = self.fuse(torch.cat([self.hand(h), self.pose(p), self.face(f)], dim=-1))
        z, _ = self.gru(z)                                  # (B,T,2H)
        w = torch.softmax(self.att(z), dim=1)
        pooled = torch.cat([(w * z).sum(1), z.max(1).values], dim=-1)
        return self.head(pooled)


# --------------------------------------------------------------------------------------
# Dados / checkpoint
# --------------------------------------------------------------------------------------
NPY_PATTERN = re.compile(r"^(.+)_(\d+)\.npy$")


def find_base_dir():
    try:
        here = Path(__file__).resolve().parent
    except NameError:  # Jupyter/Colab
        here = Path.cwd()
    for c in [here, *here.parents, Path.cwd(), Path.cwd() / "libras-recognition"]:
        if (c / "data" / "processed" / "landmarks").is_dir():
            return c
    return Path.cwd() / "libras-recognition"


def load_items(lm_dir):
    items = []
    for f in sorted(Path(lm_dir).glob("*.npy")):
        m = NPY_PATTERN.match(f.name)
        if not m:
            print(f"[aviso] nome fora do padrão <sinal>_<n>.npy, ignorado: {f.name}")
            continue
        raw = np.load(f)
        if raw.ndim != 2 or raw.shape[1] != RAW_DIM or raw.shape[0] < 5:
            print(f"[aviso] shape inesperado {raw.shape} em {f.name} (esperado: frames x {RAW_DIM}), ignorado")
            continue
        items.append({"label": m.group(1), "rep": int(m.group(2)), "raw": raw.astype(np.float32)})
    if not items:
        sys.exit(f"[erro] nenhum .npy válido em {lm_dir}. Rode o extract_landmarks.py antes.")
    return items


def save_checkpoint(path, model, classes, mean, std, T):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "classes": classes, "mean": mean.tolist(), "std": std.tolist(),
                "T": T, "feat_dim": FEAT_DIM}, path)
    with open(Path(path).with_name("video_classes.json"), "w", encoding="utf-8") as f:
        json.dump({c: i for i, c in enumerate(classes)}, f, ensure_ascii=False, indent=2)


def load_checkpoint(path, device):
    if not Path(path).is_file():
        sys.exit(f"[erro] modelo não encontrado: {path}. Treine primeiro (python train_video_model.py).")
    ck = torch.load(path, map_location=device, weights_only=False)
    if ck["feat_dim"] != FEAT_DIM:
        sys.exit("[erro] o modelo foi treinado com outro layout de features; treine novamente.")
    ck["mean"] = np.array(ck["mean"], dtype=np.float32)
    ck["std"] = np.array(ck["std"], dtype=np.float32)
    model = SignNet(len(ck["classes"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


# --------------------------------------------------------------------------------------
# Avaliação / inferência
# --------------------------------------------------------------------------------------
@torch.no_grad()
def predict_probs(model, items, T, mean, std, device):
    ds = SignDataset(items, T, mean, std, train=False)
    model.eval()
    out = [torch.softmax(model(xb.to(device)), dim=-1).cpu() for xb, _ in DataLoader(ds, batch_size=64)]
    return torch.cat(out).numpy()


def evaluate(model, items, classes, T, mean, std, device, title="Avaliação", verbose=True):
    probs = predict_probs(model, items, T, mean, std, device)
    y = np.array([y for _, y in items])
    pred = probs.argmax(1)
    k = min(3, len(classes))
    top1 = float((pred == y).mean())
    top3 = float(np.mean([y[i] in np.argsort(-probs[i])[:k] for i in range(len(y))]))
    if verbose:
        print(f"\n===== {title} ({len(y)} amostras) =====")
        print(f"top-1: {top1:.1%} | top-{k}: {top3:.1%}")
        wrong = [(classes[y[i]], classes[pred[i]], probs[i, pred[i]]) for i in range(len(y)) if pred[i] != y[i]]
        if wrong:
            print(f"Erros ({len(wrong)}):")
            for t_, p_, c_ in wrong[:30]:
                print(f"  verdadeiro: {t_!r:35} previsto: {p_!r} (conf {c_:.0%})")
            pairs = Counter((a, b) for a, b, _ in wrong).most_common(5)
            print("Confusões mais frequentes:", "; ".join(f"{a} -> {b} ({n}x)" for (a, b), n in pairs))
    return top1, top3


@torch.no_grad()
def infer(model, ck, raw, device):
    x = (make_input(raw, ck["T"], False, None) - ck["mean"]) / ck["std"]
    xb = torch.from_numpy(x.astype(np.float32))[None].to(device)
    model.eval()
    return torch.softmax(model(xb), dim=-1)[0].cpu().numpy()


# --------------------------------------------------------------------------------------
# MediaPipe (mesmas configurações do extract_landmarks.py)
# --------------------------------------------------------------------------------------
def get_holistic_module():
    try:
        import mediapipe as mp
        return mp.solutions.holistic
    except (ImportError, AttributeError):
        sys.exit("[erro] mediapipe.solutions indisponível. Use: pip install mediapipe==0.10.21 (Python 3.9-3.12).")


def extract_keypoints(results):
    def arr(lms, n):
        return np.zeros(n * 3, dtype=np.float32) if lms is None else \
            np.array([[p.x, p.y, p.z] for p in lms.landmark], dtype=np.float32).flatten()
    return np.concatenate([arr(results.pose_landmarks, N_POSE), arr(results.face_landmarks, N_FACE),
                           arr(results.left_hand_landmarks, N_HAND), arr(results.right_hand_landmarks, N_HAND)])


def new_holistic(mp_holistic):
    return mp_holistic.Holistic(static_image_mode=False, model_complexity=1,
                                min_detection_confidence=0.5, min_tracking_confidence=0.5)


def extract_video_landmarks(video_path):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        sys.exit(f"[erro] não consegui abrir o vídeo: {video_path}")
    frames = []
    with new_holistic(get_holistic_module()) as h:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(extract_keypoints(h.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
    cap.release()
    if len(frames) < 5:
        sys.exit("[erro] vídeo com poucos frames.")
    return np.stack(frames)


# --------------------------------------------------------------------------------------
# Comandos
# --------------------------------------------------------------------------------------
def cmd_train(args, paths, device):
    items_all = load_items(paths["landmarks"])
    classes = sorted({it["label"] for it in items_all})
    cidx = {c: i for i, c in enumerate(classes)}
    reps = sorted({it["rep"] for it in items_all})
    final = args.final
    if not final and len(reps) < 2:
        print("[aviso] só existe 1 repetição; treinando com tudo (sem conjunto de teste).")
        final = True
    test_rep = args.test_rep if args.test_rep in reps else reps[-1]

    if final:
        train_items, test_items = items_all, []
    else:
        train_items = [it for it in items_all if it["rep"] != test_rep]
        test_items = [it for it in items_all if it["rep"] == test_rep]
    tr = [(it["raw"], cidx[it["label"]]) for it in train_items]
    te = [(it["raw"], cidx[it["label"]]) for it in test_items]

    print(f"Classes: {len(classes)} | repetições: {reps} | dispositivo: {device}")
    print(f"Treino: {len(tr)} vídeos | Teste: {len(te)} vídeos" + ("" if final else f" (repetição {test_rep} reservada)"))
    print(f"Features por frame: mãos={HAND_DIM}, pose={POSE_DIM}, rosto={FACE_DIM} (total {FEAT_DIM}) | frames/amostra: {args.frames}")
    missing = set(range(len(classes))) - {y for _, y in tr}
    if missing:
        print("[aviso] classes sem nenhum vídeo no treino:", [classes[i] for i in sorted(missing)])

    mean, std = compute_stats(tr, args.frames, args.seed)
    ds = SignDataset(tr, args.frames, mean, std, train=True, repeat=args.repeat, seed=args.seed)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=0)

    model = SignNet(len(classes)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-2)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.epochs * len(loader), pct_start=0.15)
    crit = nn.CrossEntropyLoss(label_smoothing=0.1)

    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        model.train()
        loss_sum, correct, total = 0.0, 0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            out = model(xb)
            loss = crit(out, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            loss_sum += loss.item() * len(yb)
            correct += (out.argmax(1) == yb).sum().item()
            total += len(yb)
        if ep % args.log_every == 0 or ep == 1 or ep == args.epochs:
            msg = f"época {ep:03d}/{args.epochs} | loss {loss_sum / total:.3f} | acc treino {correct / total:.1%}"
            if te:
                msg += f" | acc teste {evaluate(model, te, classes, args.frames, mean, std, device, verbose=False)[0]:.1%}"
            print(msg + f" | {time.time() - t0:.0f}s")

    save_checkpoint(paths["model"], model, classes, mean, std, args.frames)
    print(f"\n[ok] modelo salvo em {paths['model']}")
    if te:
        evaluate(model, te, classes, args.frames, mean, std, device, title=f"TESTE (repetição {test_rep}, nunca vista no treino)")
    else:
        evaluate(model, tr, classes, args.frames, mean, std, device, title="Treino (sem teste independente)")
        print("Obs.: modo --final usa todas as repetições; não há estimativa honesta de acurácia.")


def cmd_eval(args, paths, device):
    model, ck = load_checkpoint(paths["model"], device)
    classes = ck["classes"]
    cidx = {c: i for i, c in enumerate(classes)}
    items = [it for it in load_items(paths["landmarks"]) if it["label"] in cidx and (args.test_rep == 0 or it["rep"] == args.test_rep)]
    if not items:
        sys.exit("[erro] nenhuma amostra para avaliar (confira --test-rep; use 0 para todas).")
    data = [(it["raw"], cidx[it["label"]]) for it in items]
    evaluate(model, data, classes, ck["T"], ck["mean"], ck["std"], device,
             title="Avaliação (todas as repetições)" if args.test_rep == 0 else f"Avaliação (repetição {args.test_rep})")


def cmd_predict(args, paths, device):
    if not args.video:
        sys.exit('[erro] informe o vídeo: python train_video_model.py predict --video "caminho.mp4"')
    model, ck = load_checkpoint(paths["model"], device)
    print("Extraindo landmarks do vídeo...")
    probs = infer(model, ck, extract_video_landmarks(args.video), device)
    print(f"\nTop-{args.topk} para {Path(args.video).name}:")
    for i in np.argsort(-probs)[:args.topk]:
        print(f"  {ck['classes'][i]:35} {probs[i]:.1%}")


def _font(size):
    from PIL import ImageFont
    for name in ("arial.ttf", "C:/Windows/Fonts/arial.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def draw_text(frame, text, xy, size=30, color=(255, 255, 255)):
    """Desenha texto com acentos (cv2.putText não suporta ô, ã, ç...)."""
    import cv2
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        cv2.putText(frame, text, xy, cv2.FONT_HERSHEY_SIMPLEX, size / 40, color[::-1], 2)
        return frame
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(img).text(xy, text, font=_font(size), fill=color)
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)


def cmd_webcam(args, paths, device):
    import cv2
    model, ck = load_checkpoint(paths["model"], device)
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        sys.exit("[erro] não consegui abrir a webcam (tente --camera 1).")
    seg, idle, subtitle, conf = [], 0, "", 0.0
    print("Webcam ativa. Faça o sinal; ao baixar as mãos o sinal é reconhecido. 'q' sai, 'c' limpa.")
    with new_holistic(get_holistic_module()) as h:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            kp = extract_keypoints(h.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            hands = bool(np.abs(kp[LH_SL]).sum() > 0 or np.abs(kp[RH_SL]).sum() > 0)
            if hands:
                seg.append(kp)
                idle = 0
            elif seg:
                seg.append(kp)
                idle += 1
            if seg and (idle >= args.idle_frames or len(seg) >= args.max_frames):
                raw = np.stack(seg)
                n_hand = int(((np.abs(raw[:, LH_SL]).sum(1) + np.abs(raw[:, RH_SL]).sum(1)) > 0).sum())
                if n_hand >= args.min_frames:
                    probs = infer(model, ck, raw, device)
                    k = int(probs.argmax())
                    if probs[k] >= args.threshold:
                        subtitle, conf = ck["classes"][k], float(probs[k])
                        print(f"-> {subtitle} ({conf:.0%})")
                seg, idle = [], 0
            h_img, w_img = frame.shape[:2]
            cv2.rectangle(frame, (0, h_img - 70), (w_img, h_img), (0, 0, 0), -1)
            frame = draw_text(frame, f"{subtitle}  ({conf:.0%})" if subtitle else "...", (15, h_img - 58), 32)
            frame = draw_text(frame, "capturando sinal" if seg else ("mãos visíveis" if hands else "aguardando"), (15, 8), 22, (0, 255, 0))
            cv2.imshow("Libras - tempo real (q sai, c limpa)", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("c"):
                subtitle, conf = "", 0.0
    cap.release()
    cv2.destroyAllWindows()


def parse_args():
    p = argparse.ArgumentParser(description="Treino e teste do modelo de sinais (vídeo) de Libras")
    p.add_argument("mode", nargs="?", default="train", choices=["train", "eval", "predict", "webcam"])
    p.add_argument("--base-dir", help="raiz do projeto (auto-detectada)")
    p.add_argument("--landmarks-dir")
    p.add_argument("--model", help="caminho do .pt (padrão: models_output/video_model.pt)")
    p.add_argument("--frames", type=int, default=64, help="frames por amostra após reamostragem")
    p.add_argument("--epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--repeat", type=int, default=12, help="cópias aumentadas de cada vídeo por época")
    p.add_argument("--test-rep", type=int, default=3, help="repetição reservada para teste (0 = todas, no modo eval)")
    p.add_argument("--final", action="store_true", help="treina com todas as repetições (sem teste)")
    p.add_argument("--log-every", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--video", help="vídeo para o modo predict")
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--camera", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.5, help="confiança mínima para mostrar a legenda")
    p.add_argument("--idle-frames", type=int, default=12, help="frames sem mãos que encerram um sinal")
    p.add_argument("--max-frames", type=int, default=150)
    p.add_argument("--min-frames", type=int, default=10, help="mínimo de frames com mãos para classificar")
    return p.parse_known_args()[0]


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    base = Path(args.base_dir) if args.base_dir else find_base_dir()
    paths = {"landmarks": Path(args.landmarks_dir) if args.landmarks_dir else base / "data" / "processed" / "landmarks",
             "model": Path(args.model) if args.model else base / "models_output" / "video_model.pt"}
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Projeto: {base}")
    {"train": cmd_train, "eval": cmd_eval, "predict": cmd_predict, "webcam": cmd_webcam}[args.mode](args, paths, device)


if __name__ == "__main__":
    main()