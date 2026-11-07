# ParkVision : expérience B (YOLOv8s)

Branche : `experience-b-yolov8s`. Cette expérience mesure l'effet du passage de
YOLOv8n (expérience A) à YOLOv8s, **à protocole identique** : mêmes données
(`data/raw.dvc`, manifeste `split_manifest.csv`), mêmes lots (train 8 613,
valid 2 500, test 1 029), résolution 640, batch 16, AdamW, graine 42, 50 époques
maximum, patience 10.

## Ce qui change par rapport à A

| Élément | Fichier | Différence |
|---|---|---|
| Configuration | `configs/train_b.yaml` | uniquement `model: yolov8s.pt` et `name: B_yolov8s_baseline` |
| Auteur MLflow | `src/models/train.py` | le tag `member` lit `PARKVISION_MEMBER` (défaut : `Nourchenne`, qui préserve l'attribution de A) |

## Outils ajoutés

- `notebooks/ParkVision_Training_B.ipynb` : notebook Colab complet (clone de la
  branche, données, DVC, vérification, sauvegarde des poids sur Drive, essai
  court, entraînement, export MLflow, analyse). Livré sans sortie ni jeton.
- `src/analysis/analyze_errors.py` : analyse du lot **valid** défini dans le
  manifeste (jamais `data/raw/valid`, ancien découpage). Il écrit
  `comptage.csv`, `comptage_resume.json`, `erreurs_detail.csv` et des images
  d'exemple dans `reports/experience_b/`.
- `tests/test_error_analysis.py` : 12 tests de la logique d'appariement et de
  comptage, sans GPU ni données.

## Définitions de l'analyse

- **Seuils de comptage** : confiance 0,25 et NMS IoU 0,7 (valeurs par défaut,
  à aligner avec A et C avant la comparaison). Appariement vérité/prédiction :
  IoU ≥ 0,5.
- **Détection correcte** : même classe et IoU ≥ 0,5 (appariement glouton par IoU décroissant).
- **Confusion libre/occupée** : bon emplacement, mauvaise classe.
- **Oubli** : place annotée sans prédiction.
- **Double détection** : prédiction supplémentaire sur une place déjà détectée.
- **Fausse détection** : prédiction sans place annotée correspondante. Elle peut
  correspondre à une vraie place visible mais non annotée dans PKLot : ces cas
  sont à relire visuellement et à signaler à part dans `erreurs.md`.
- **MAE de comptage** : moyenne, sur les images évaluées, de
  |prédit − annoté|, calculée séparément pour les libres et les occupées.

## Reproduire

Tests locaux (Windows, sans GPU) :

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Entraînement et analyse : exécuter dans l'ordre les cellules de
`notebooks/ParkVision_Training_B.ipynb` sur un runtime GPU. Les poids sont
écrits sur Google Drive (`ParkVision_B/runs_detect/`) ; le ZIP MLflow
téléchargé est importé sur le serveur MLflow du groupe avec
`src/utils/transfer_run.py import`.

## État

- [x] Configuration B, tag membre configurable, outils d'analyse et tests.
- [ ] Essai court sur Colab et vérification des poids sur Drive.
- [ ] Entraînement complet, ZIP MLflow, import par Nourchenne.
- [ ] Analyse du lot valid et rapports `resultats.md` / `erreurs.md`.

Limites : `train.py` n'offre pas de reprise après coupure ; une session Colab
interrompue obligerait à reprendre depuis `last.pt` avec un code à ajouter.
Les journées sont séparées, pas les parkings ni les caméras : les scores ne
prouvent pas la généralisation à un parking inconnu.
