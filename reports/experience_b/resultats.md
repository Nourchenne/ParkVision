# Expérience B · YOLOv8s : résultats

## Identification

| Élément | Valeur |
|---|---|
| Modèle | YOLOv8s (`yolov8s.pt`, 11,1 M de paramètres, 28,4 GFLOPs) |
| Configuration | `configs/train_b.yaml` (seuls `model` et `name` diffèrent de A) |
| Branche / commit Git de l'entraînement | `experience-b-yolov8s` / `d438560e179d6691079825ee8250ef6445fb9022` |
| Identifiant du run MLflow (Colab, lu dans `metadata.json` du ZIP) | `faf38527c8e34d8c9f5d4672500affba` |
| Identifiant du run sur le serveur du groupe | **À compléter après l'import du ZIP par Nourchenne** |
| Statut du run | `FINISHED` |
| Nom du ZIP MLflow | `B_yolov8s_baseline_faf38527c8e34d8c9f5d4672500affba.zip` |
| Poids | `best.pt` (22,5 Mo) sur Google Drive : `ParkVision_B/runs_detect/B_yolov8s_baseline_faf38527/weights/` |
| SHA-256 de `best.pt` | `d1f0e97c360394871f2afe92c3bfea6e5192fc82e5d236c968bfcc4fe95657e9` |
| Empreinte du manifeste de découpage | `b51c7072a9276eff8c69d9023849979efaecf1aadd458f31c2e81651ee76630a` |
| Membre (tag MLflow `member`) | voir « Points à corriger » ci-dessous |

## Entraînement

| Élément | Valeur |
|---|---|
| Époques terminées | 50 sur 50 (l'arrêt anticipé, patience 10, ne s'est pas déclenché) |
| Meilleure époque (critère Ultralytics : 0,1·mAP50 + 0,9·mAP50-95) | 42 (mAP50-95 = 0,9326) |
| Durée | 2,574 h d'entraînement (9 266 s dans `results.csv`) ; 9 340 s entre le début et la fin du run MLflow, validation finale comprise |
| Matériel | Google Colab, GPU Tesla T4 (14 913 MiB) |
| Logiciels | Ultralytics 8.4.201, PyTorch 2.14.1+cu130, Python 3.13.16 |
| Données | train 8 613 images, valid 2 500, test 1 029 (non utilisé) ; découpage par journée de capture |
| Réglages | résolution 640, batch 16, AdamW, lr0 0,001, lrf 0,01, graine 42, mode déterministe |

L'évolution époque par époque est dans [`results.csv`](results.csv). Le mAP50 est
déjà à 0,994 dès la première époque : il ne départage plus les modèles. Le mAP50-95
passe de 0,780 (époque 1) à 0,933 (époque 42) et plafonne ensuite.

## Résultats sur le lot valid (2 500 images), poids `best.pt`

Valeurs de la validation finale d'Ultralytics. Elle utilise un seuil de confiance très bas,
propre au calcul du mAP : ce n'est pas le seuil de comptage.

| Classe | Images | Instances | Précision | Rappel | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| Toutes | 2 500 | 139 036 | 0,999 | 0,999 | 0,995 | 0,933 |
| `space-empty` | 2 265 | 73 367 | 0,999 | 0,999 | 0,995 | 0,948 |
| `space-occupied` | 2 162 | 65 669 | 1,000 (arrondi) | 0,999 | 0,995 | 0,917 |

Vitesse sur T4 : 0,1 ms de prétraitement, 2,6 ms d'inférence, 2,2 ms de post-traitement par image.

## Comptage (seuils fixés pour A, B et C : à confirmer avec Nourchenne)

- Seuil de confiance : **0,25**
- Seuil NMS (IoU) : **0,7**
- Seuil d'appariement vérité/prédiction (IoU) : 0,5
- Images évaluées : **2 500** (lot valid de `split_manifest.csv`)

| | Libres | Occupées |
|---|---:|---:|
| Places annotées | 73 367 | 65 669 |
| Places prédites | 73 527 | 65 688 |
| Écart total | +160 (+0,22 %) | +19 (+0,03 %) |
| **MAE par image** | **0,0784** | **0,0620** |

- 2 294 images sur 2 500 (91,8 %) ont un comptage exact pour les deux classes.
- 2 476 images (99,0 %) ont un écart d'au plus 1 place pour chaque classe.
- Écart maximal observé : 16 places libres et 5 places occupées (même image, 2012-10-31 à 15:03).
- Sans l'image `2013-03-02_12_50_07` (voir `erreurs.md`), les MAE sont de 0,0776 (libres) et 0,0616 (occupées) : cette image n'explique pas le MAE, car ses erreurs se compensent dans le comptage.
- Détail par image dans [`comptage.csv`](comptage.csv).

## Erreurs d'appariement (IoU 0,5)

| Type | Nombre |
|---|---:|
| Fausse détection | 196 |
| Confusion libre/occupée | 88 |
| Oubli | 40 |
| Double détection | 23 |
| **Total** | **347** (206 images concernées sur 2 500) |

Analyse détaillée et exemples : [`erreurs.md`](erreurs.md).

## Points à corriger avant la comparaison finale

1. **Tag `member`** : le run contient `member = Prenom Nom` (valeur d'exemple du notebook,
   non remplacée avant l'entraînement). L'auteur réel est la membre 2. Le ZIP ne doit pas être
   modifié (il est vérifié par empreintes) : le tag sera corrigé dans MLflow après l'import,
   ou ce défaut sera signalé tel quel.
2. **Identifiant MLflow du groupe** à ajouter dans le tableau ci-dessus après l'import.
3. **Seuils de comptage** à valider avec Nourchenne pour A et C.

## Limites

- Les journées sont séparées entre train, valid et test, mais pas les parkings ni les caméras :
  ces scores ne prouvent pas la généralisation à un parking inconnu.
- Les 1 029 images de test n'ont pas été utilisées.
- Aucune comparaison chiffrée avec A n'est donnée ici : elle sera faite avec Nourchenne,
  à seuils identiques.
