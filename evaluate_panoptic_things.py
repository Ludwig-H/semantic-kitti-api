#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Évaluation panoptique THINGS-only pour SemanticKITTI.
# - Utilise PanopticEval de l'API officielle
# - Résume uniquement les métriques THINGS (PQ, RQ, SQ) + temps total
# - Robuste aux variantes noms/IDs (s’appuie sur le YAML pour mapper)

import argparse
import os
import sys
import time
import yaml
import numpy as np
from tqdm import tqdm

# S'assurer que l'import "auxiliary" fonctionne quand on lance ce script directement
REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from auxiliary.eval_np import PanopticEval  # noqa: E402

SPLITS = ["train", "valid", "test"]

THING_NAMES = [
    "car", "bicycle", "motorcycle", "truck", "other-vehicle",
    "person", "bicyclist", "motorcyclist",
]

def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_class_lut(learning_map):
    maxkey = max(int(k) for k in learning_map.keys())
    lut = np.zeros((maxkey + 100), dtype=np.int32)  # marge pour labels inconnus
    keys = [int(k) for k in learning_map.keys()]
    vals = [int(v) for v in learning_map.values()]
    lut[keys] = vals
    return lut

def list_label_files(root, seqs):
    out = []
    for s in seqs:
        s = f"{int(s):02d}"
        d = os.path.join(root, "sequences", s, "labels")
        out.extend(sorted(os.path.join(d, fn) for fn in os.listdir(d) if fn.endswith(".label")))
    return out

def list_pred_files(root, seqs):
    out = []
    for s in seqs:
        s = f"{int(s):02d}"
        d = os.path.join(root, "sequences", s, "predictions")
        out.extend(sorted(os.path.join(d, fn) for fn in os.listdir(d) if fn.endswith(".label")))
    return out

def main():
    parser = argparse.ArgumentParser("./evaluate_panoptic_things.py")
    parser.add_argument("--dataset", "-d", type=str, required=True,
                        help="Racine du dataset SemanticKITTI.")
    parser.add_argument("--predictions", "-p", type=str, required=False, default=None,
                        help="Racine des prédictions (même orga que dataset). "
                             "Par défaut: = --dataset.")
    parser.add_argument("--split", "-s", type=str, choices=SPLITS, default="valid",
                        help=f"Split à évaluer. Défaut: valid.")
    parser.add_argument("--data_cfg", "-dc", type=str, default="config/semantic-kitti.yaml",
                        help="Chemin du YAML SemanticKITTI. Défaut: config/semantic-kitti.yaml")
    parser.add_argument("--limit", "-l", type=int, default=None,
                        help="Limiter au N premiers points de chaque scan (debug).")
    parser.add_argument("--min_inst_points", type=int, default=50,
                        help="Taille min d’une instance pour l’éval panoptique.")
    args = parser.parse_args()

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

    # Charger config
    cfg = load_yaml(args.data_cfg)
    learning_map = {int(k): int(v) for k, v in cfg["learning_map"].items()}
    learning_ignore = {int(k): int(v) for k, v in cfg["learning_ignore"].items()}
    labels = {int(k): str(v) for k, v in cfg["labels"].items()}
    inv_map = {int(k): int(v) for k, v in cfg["learning_map_inv"].items()}

    # classes à ignorer pour PanopticEval (IDs train où ignore=1)
    ignore_train_ids = [cid for cid, ign in learning_ignore.items() if int(ign) == 1]

    # LUT pour remapper RAW->TRAIN
    lut = build_class_lut(learning_map)

    # Séquences du split
    seqs = cfg["split"][args.split]
    if not isinstance(seqs, list):
        seqs = list(seqs)

    # Lister fichiers
    label_files = list_label_files(args.dataset, seqs)
    pred_files = list_pred_files(args.predictions, seqs)
    if len(label_files) == 0:
        raise RuntimeError("Aucun fichier GT trouvé.")
    if len(pred_files) == 0:
        raise RuntimeError("Aucune prédiction trouvée.")
    if len(label_files) != len(pred_files):
        raise RuntimeError(f"Compte différent GT={len(label_files)} vs Pred={len(pred_files)}.")

    # Panoptic evaluator
    nr_classes = 1 + max(inv_map.keys())  # upper bound safe
    peval = PanopticEval(nr_classes, None, ignore_train_ids, min_points=args.min_inst_points)

    # Boucle d’éval avec barre de progression
    t0 = time.time()
    for gt_path, pr_path in tqdm(zip(label_files, pred_files),
                                 total=len(label_files),
                                 desc="Evaluating scans", ncols=0):
        # GT
        gt = np.fromfile(gt_path, dtype=np.uint32)
        gt_sem_train = lut[gt & 0xFFFF]
        gt_inst = gt

        # Pred
        pr = np.fromfile(pr_path, dtype=np.uint32)
        pr_sem_train = lut[pr & 0xFFFF]
        pr_inst = pr

        if args.limit is not None:
            n = int(args.limit)
            gt_sem_train = gt_sem_train[:n]
            gt_inst = gt_inst[:n]
            pr_sem_train = pr_sem_train[:n]
            pr_inst = pr_inst[:n]

        peval.addBatch(pr_sem_train, pr_inst, gt_sem_train, gt_inst)

    dt = time.time() - t0

    # Récupérer métriques
    class_PQ, class_SQ, class_RQ, class_all_PQ, class_all_SQ, class_all_RQ = peval.getPQ()
    class_IoU, class_all_IoU = peval.getSemIoU()

    # Convertir en listes python
    class_all_PQ = class_all_PQ.flatten().tolist()
    class_all_SQ = class_all_SQ.flatten().tolist()
    class_all_RQ = class_all_RQ.flatten().tolist()
    class_all_IoU = class_all_IoU.flatten().tolist()

    # Construire dictionnaire {nom_de_classe: métriques}
    output_dict = {}
    for idx, (pq, rq, sq, iou) in enumerate(zip(class_all_PQ, class_all_RQ, class_all_SQ, class_all_IoU)):
        # idx est un ID train dense [0..C-1]; on passe par inv_map pour retrouver le "train id" vrai
        # puis labels[train_id] pour obtenir le nom.
        train_id = idx  # dans l’implémentation PanopticEval, l’ordre colle aux IDs train
        name = labels.get(train_id, f"class_{train_id}")
        output_dict[name] = {"PQ": pq, "RQ": rq, "SQ": sq, "IoU": iou}

    # THINGS only
    present_things = [n for n in THING_NAMES if n in output_dict]
    if len(present_things) == 0:
        raise RuntimeError("Aucune classe THINGS présente selon le YAML. Vérifie 'labels' et learning_map_inv.")

    def _mean(key):
        vals = []
        for n in present_things:
            v = output_dict[n].get(key, None)
            if v is not None:
                try:
                    vals.append(float(v))
                except Exception:
                    pass
        return float(np.mean(vals)) if len(vals) else float("nan")

    PQ_th = _mean("PQ")
    RQ_th = _mean("RQ")
    SQ_th = _mean("SQ")

    # Affichage résumé
    print("\n" + "=" * 60)
    print(" Résultats panoptiques — THINGS uniquement")
    print("=" * 60)
    print(f" Scans   : {len(label_files)} | Temps: {dt:.2f} s")
    print(f" Classes : {present_things}")
    print(f" PQ_th   : {PQ_th:.4f}")
    print(f" RQ_th   : {RQ_th:.4f}")
    print(f" SQ_th   : {SQ_th:.4f}")
    print("=" * 60 + "\n")

    # Affichage par classe (utile, et sans fantômes)
    print("Détail par classe (THINGS):")
    for n in present_things:
        e = output_dict[n]
        print(f" - {n:<15} PQ={e['PQ']:.4f} | RQ={e['RQ']:.4f} | SQ={e['SQ']:.4f}")

if __name__ == "__main__":
    main()
