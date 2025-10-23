#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Évaluation panoptique THINGS-only pour SemanticKITTI (propre et robuste).
# - Décode sem/inst depuis le panoptic uint32
# - Remap RAW->TRAIN via learning_map
# - Inst = 0 pour les classes non-THINGS (d'après YAML: "instances")
# - Sanity-check: détecte si Pred == GT sur un échantillon
# - Moyennes THINGS via IDs train (pas de KeyError sur des noms)

import argparse
import os
import sys
import time
import yaml
import numpy as np
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from auxiliary.eval_np import PanopticEval  # noqa: E402

SPLITS = ["train", "valid", "test"]

def _load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def _build_lut_raw2train(learning_map: dict) -> np.ndarray:
    # Keys in YAML are str; cast to int
    items = {int(k): int(v) for k, v in learning_map.items()}
    max_raw = max(items.keys()) if items else 0
    lut = np.zeros(max_raw + 1 + 1024, dtype=np.int32)  # marge large
    for raw, tid in items.items():
        lut[raw] = tid
    return lut

def _list_label_files(root, seqs):
    out = []
    for s in seqs:
        s = f"{int(s):02d}"
        d = os.path.join(root, "sequences", s, "labels")
        if not os.path.isdir(d):
            continue
        out.extend(sorted(
            os.path.join(d, fn) for fn in os.listdir(d) if fn.endswith(".label")
        ))
    return out

def _list_pred_files(root, seqs):
    out = []
    for s in seqs:
        s = f"{int(s):02d}"
        d = os.path.join(root, "sequences", s, "predictions")
        if not os.path.isdir(d):
            continue
        out.extend(sorted(
            os.path.join(d, fn) for fn in os.listdir(d) if fn.endswith(".label")
        ))
    return out

def main():
    ap = argparse.ArgumentParser("./evaluate_panoptic_things.py")
    ap.add_argument("-d", "--dataset", required=True, type=str,
                    help="Racine du dataset SemanticKITTI.")
    ap.add_argument("-p", "--predictions", default=None, type=str,
                    help="Racine des prédictions (même orga que dataset). Défaut: = --dataset.")
    ap.add_argument("-s", "--split", default="valid", choices=SPLITS)
    ap.add_argument("-dc", "--data_cfg", default="config/semantic-kitti.yaml", type=str)
    ap.add_argument("-l", "--limit", default=None, type=int)
    ap.add_argument("--min_inst_points", default=50, type=int)
    args = ap.parse_args()

    if args.predictions is None:
        args.predictions = args.dataset

    print("*" * 70)
    print("INTERFACE (THINGS-only)")
    print("Data      :", args.dataset)
    print("Preds     :", args.predictions)
    print("Split     :", args.split)
    print("Config    :", args.data_cfg)
    print("Limit     :", args.limit)
    print("Min inst  :", args.min_inst_points)
    print("*" * 70)

    cfg = _load_yaml(args.data_cfg)
    labels = {int(k): str(v) for k, v in cfg["labels"].items()}                # train_id -> name
    learning_map = {int(k): int(v) for k, v in cfg["learning_map"].items()}    # raw -> train
    learning_ignore = {int(k): int(v) for k, v in cfg["learning_ignore"].items()}  # train -> 0/1
    learning_map_inv = {int(k): int(v) for k, v in cfg["learning_map_inv"].items()} # train -> raw
    raw_instances = set(int(x) for x in cfg.get("instances", []))              # RAW ids with instances

    # Déduire les IDs train des THINGS à partir des RAW "instances"
    thing_train_ids = sorted({learning_map.get(rid, 0) for rid in raw_instances})
    # Nettoyage: enlever 0 et ceux ignorés
    thing_train_ids = [tid for tid in thing_train_ids if tid > 0 and learning_ignore.get(tid, 0) == 0]
    if not thing_train_ids:
        raise RuntimeError("Impossible de déduire des classes THINGS depuis le YAML (champ 'instances').")

    # LUT RAW->TRAIN
    lut = _build_lut_raw2train(cfg["learning_map"])

    # Séquences du split
    seqs = cfg["split"][args.split]
    seqs = list(seqs) if isinstance(seqs, (list, tuple)) else [seqs]

    # Fichiers
    gt_files = _list_label_files(args.dataset, seqs)
    pr_files = _list_pred_files(args.predictions, seqs)
    if len(gt_files) == 0:
        raise RuntimeError("Aucun fichier GT trouvé.")
    if len(pr_files) == 0:
        raise RuntimeError("Aucune prédiction trouvée.")
    if len(gt_files) != len(pr_files):
        raise RuntimeError(f"Compte différent GT={len(gt_files)} vs Pred={len(pr_files)}.")

    # Garde-fou: vérifie rapidement si Pred == GT pour quelques scans
    try:
        same_count = 0
        sample_ids = np.linspace(0, len(gt_files)-1, num=min(5, len(gt_files)), dtype=int)
        for i in sample_ids:
            g = np.fromfile(gt_files[i], dtype=np.uint32)
            p = np.fromfile(pr_files[i], dtype=np.uint32)
            if g.shape == p.shape and np.array_equal(g, p):
                same_count += 1
        if same_count >= max(2, len(sample_ids)//2):
            print("[ALERTE] Plusieurs fichiers Pred sont identiques aux GT. "
                  "Vérifie que tu n'évalues pas GT vs GT.")
    except Exception:
        pass

    # Prépare l'évaluateur
    nr_classes = max(learning_map_inv.keys()) + 1  # borne sûre
    ignore_train_ids = [tid for tid, ign in learning_ignore.items() if int(ign) == 1]
    peval = PanopticEval(nr_classes, None, ignore_train_ids, min_points=args.min_inst_points)

    # Éval boucle
    t0 = time.time()
    for gt_path, pr_path in tqdm(zip(gt_files, pr_files), total=len(gt_files), desc="Evaluating scans", ncols=0):
        gt_pan = np.fromfile(gt_path, dtype=np.uint32)
        pr_pan = np.fromfile(pr_path, dtype=np.uint32)

        if args.limit is not None:
            n = int(args.limit)
            gt_pan = gt_pan[:n]
            pr_pan = pr_pan[:n]

        # Décode RAW sem + inst
        gt_sem_raw = (gt_pan & 0xFFFF).astype(np.int32)
        pr_sem_raw = (pr_pan & 0xFFFF).astype(np.int32)
        gt_inst = (gt_pan >> 16).astype(np.int32)
        pr_inst = (pr_pan >> 16).astype(np.int32)

        # Remap RAW -> TRAIN
        gt_sem = lut[gt_sem_raw]
        pr_sem = lut[pr_sem_raw]

        # Inst=0 hors THINGS (côté GT et Pred)
        gt_non_thing = ~np.isin(gt_sem, thing_train_ids)
        pr_non_thing = ~np.isin(pr_sem, thing_train_ids)
        gt_inst[gt_non_thing] = 0
        pr_inst[pr_non_thing] = 0

        peval.addBatch(pr_sem, pr_inst, gt_sem, gt_inst)

    dt = time.time() - t0

    # Récupère les métriques
    class_PQ, class_SQ, class_RQ, class_all_PQ, class_all_SQ, class_all_RQ = peval.getPQ()
    class_IoU, class_all_IoU = peval.getSemIoU()
    class_all_PQ = np.asarray(class_all_PQ).reshape(-1)
    class_all_SQ = np.asarray(class_all_SQ).reshape(-1)
    class_all_RQ = np.asarray(class_all_RQ).reshape(-1)
    class_all_IoU = np.asarray(class_all_IoU).reshape(-1)

    # Filtre THINGS par IDs train
    th_idx = [tid for tid in thing_train_ids if tid < len(class_all_PQ)]
    if not th_idx:
        raise RuntimeError("Aucun ID train THINGS valide dans les métriques retournées.")

    PQ_th = float(np.nanmean(class_all_PQ[th_idx]))
    RQ_th = float(np.nanmean(class_all_RQ[th_idx]))
    SQ_th = float(np.nanmean(class_all_SQ[th_idx]))

    # Affichage global
    print("\n" + "=" * 60)
    print(" Résultats panoptiques — THINGS uniquement")
    print("=" * 60)
    print(f" Scans   : {len(gt_files)} | Temps: {dt:.2f} s")
    print(f" Things (train IDs) : {th_idx}")
    print(f" PQ_th   : {PQ_th:.4f}")
    print(f" RQ_th   : {RQ_th:.4f}")
    print(f" SQ_th   : {SQ_th:.4f}")
    print("=" * 60 + "\n")

    # Détail par classe thing (nom + valeurs)
    print("Détail par classe (THINGS):")
    for tid in th_idx:
        name = labels.get(int(tid), f"class_{int(tid)}")
        print(f" - {name:<15} PQ={class_all_PQ[tid]:.4f} | RQ={class_all_RQ[tid]:.4f} | SQ={class_all_SQ[tid]:.4f}")

if __name__ == "__main__":
    main()
