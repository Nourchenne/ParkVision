# Expérience B — Résultats de YOLOv8s

## Identification

| Élément | Valeur |
|---|---|
| Modèle | YOLOv8s (`yolov8s.pt`, 11,1 M de paramètres, 28,4 GFLOPs) |
| Configuration | `configs/train_b.yaml` ; seuls `model` et `name` diffèrent de la configuration A |
| Branche de l’entraînement | `experience-b-yolov8s` |
| Commit Git de l’entraînement | `d438560e179d6691079825ee8250ef6445fb9022` |
| Identifiant du run MLflow source sur Colab | `faf38527c8e34d8c9f5d4672500affba` |
| Identifiant du run importé dans MLflow local chez Nourchenne | `72e5fc398bf441929066040a3627a3af` |
| Expérience MLflow | `ParkVision` |
| Nom du run | `B_yolov8s_baseline` |
| Statut du run importé | `FINISHED` |
| Archive MLflow originale | `B_yolov8s_baseline_faf38527c8e34d8c9f5d4672500affba.zip` |
| Poids retenus | `best.pt`, environ 22,5 Mo |
| Emplacement des poids dans les artefacts MLflow | `training/weights/best.pt` et `training/weights/last.pt` |
| Copie des poids sur Google Drive | `ParkVision_B/runs_detect/B_yolov8s_baseline_faf38527/weights/` |
| SHA-256 de `best.pt` communiqué par le membre B | `d1f0e97c360394871f2afe92c3bfea6e5192fc82e5d236c968bfcc4fe95657e9` |
| SHA-256 du manifeste de découpage | `b51c7072a9276eff8c69d9023849979efaecf1aadd458f31c2e81651ee76630a` |
| Auteur | Membre B ; prénom et nom exacts à reporter depuis le tag `member` du run local |

## Entraînement

| Élément | Valeur |
|---|---|
| Époques terminées | 50 sur 50 |
| Arrêt anticipé | Patience de 10 époques ; non déclenché |
| Meilleure époque rapportée | 42, avec un mAP50-95 d’environ 0,9326 |
| Durée d’entraînement | 9 266 secondes, soit environ 2,574 heures, d’après `results.csv` |
| Durée totale du run MLflow | Environ 9 340 secondes, validation finale comprise |
| Matériel | Google Colab, GPU Tesla T4, 14 913 MiB |
| Environnement rapporté | Ultralytics 8.4.201, PyTorch 2.14.1+cu130, Python 3.13.16 |
| Données d’entraînement | 8 613 images |
| Données de validation | 2 500 images |
| Données de test | 1 029 images, non utilisées |
| Découpage | Séparation par journée de capture |
| Résolution | 640 |
| Taille de batch | 16 |
| Optimiseur | AdamW |
| Taux d’apprentissage initial `lr0` | 0,001 |
| Facteur final `lrf` | 0,01 |
| Graine aléatoire | 42 |
| Mode déterministe | Activé |

L’évolution des métriques par époque est disponible dans [results.csv](results.csv).

Le mAP50 atteint déjà environ 0,994 dès la première époque. Étant proche de son plafond, il est moins discriminant pour comparer les expériences. Le mAP50-95 passe d’environ 0,780 à la première époque à environ 0,933 à l’époque 42, puis varie peu.

La formule exacte du critère de sélection de `best.pt` doit être vérifiée dans la version d’Ultralytics utilisée avant d’être détaillée dans le rapport final.

## Détection sur la validation

Les résultats suivants correspondent à la validation finale des poids `best.pt` sur les 2 500 images de validation.

Le calcul du mAP utilise un seuil de confiance bas pour construire les courbes précision-rappel. Le seuil de comptage présenté plus loin correspond à une évaluation distincte.

| Classe | Images contenant la classe | Instances annotées | Précision | Rappel | mAP50 | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| Toutes | 2 500 | 139 036 | 0,999 | 0,999 | 0,995 | 0,933 |
| `space-empty` | 2 265 | 73 367 | 0,999 | 0,999 | 0,995 | 0,948 |
| `space-occupied` | 2 162 | 65 669 | 1,000 après arrondi | 0,999 | 0,995 | 0,917 |

Une même image peut contenir les deux classes. Les nombres d’images par classe ne s’additionnent donc pas pour obtenir le nombre total d’images.

Les temps moyens rapportés sur T4 sont :

- Prétraitement : 0,1 ms par image.
- Inférence : 2,6 ms par image.
- Post-traitement : 2,2 ms par image.

Ces mesures dépendent des conditions d’exécution. Une comparaison de vitesse entre A, B et C devra utiliser le même matériel et le même protocole.

## Comptage sur la validation

Les seuils effectivement utilisés pour B sont :

- Confiance : **0,25**.
- IoU NMS : **0,7**.
- IoU d’appariement entre annotations et prédictions pour l’analyse des erreurs : **0,5**.
- Ensemble évalué : **2 500 images de validation**, définies par `split_manifest.csv`.

Ces réglages constituent le protocole proposé pour comparer A, B et C. Leur utilisation effective pour A et C reste à vérifier et à documenter.

| Indicateur | Places libres | Places occupées |
|---|---:|---:|
| Places annotées | 73 367 | 65 669 |
| Places prédites | 73 527 | 65 688 |
| Écart total signé | +160, soit environ +0,22 % | +19, soit environ +0,03 % |
| **MAE par image** | **0,0784** | **0,0620** |

La MAE est la moyenne des écarts absolus de comptage par image, calculée séparément pour chaque classe. L’écart total signé peut masquer des erreurs qui se compensent entre les images.

Résultats complémentaires :

- 2 294 images sur 2 500, soit 91,8 %, ont un comptage exact pour les deux classes.
- 2 476 images, soit 99,0 %, présentent un écart d’au plus une place pour chaque classe.
- Les écarts absolus maximaux rapportés sont de 16 places libres et de 5 places occupées, sur la même image du 31 octobre 2012 à 15 h 03.

Le détail par image est disponible dans [comptage.csv](comptage.csv).

### Analyse complémentaire d’une image suspecte

L’image `2013-03-02_12_50_07` présente une suspicion de décalage des annotations, décrite dans [erreurs.md](erreurs.md).

Son exclusion dans une analyse complémentaire donne des MAE d’environ 0,0776 pour les places libres et de 0,0616 pour les places occupées. Son influence sur la MAE est donc faible, même si elle concentre de nombreuses erreurs de localisation ou d’appariement.

Des erreurs de détection peuvent se compenser dans le comptage. Un nombre correct de places ne garantit donc pas que toutes les places ont été correctement localisées.

**Les résultats principaux conservent les 2 500 images.** Aucune modification du jeu de validation commun n’est appliquée sur la seule base de cette suspicion.

## Analyse des erreurs de détection

Les catégories suivantes sont produites par le script d’analyse avec un seuil d’appariement IoU de 0,5.

| Type d’erreur | Nombre |
|---|---:|
| Fausse détection | 196 |
| Confusion entre place libre et place occupée | 88 |
| Oubli | 40 |
| Double détection | 23 |
| **Total** | **347** |

Ces événements concernent 206 images sur 2 500.

Les 347 événements de détection ne représentent pas une somme d’erreurs absolues de comptage. Une image peut présenter plusieurs événements, dont certains se compensent dans les nombres de places.

Les livrables associés sont :

- [erreurs.md](erreurs.md) pour l’interprétation.
- [erreurs_detail.csv](erreurs_detail.csv) pour le détail des événements.
- [comptage_resume.json](comptage_resume.json) pour les indicateurs agrégés.
- Le dossier `exemples/` pour les illustrations.

## Import MLflow et traçabilité

L’archive de B a été importée avec succès dans l’expérience `ParkVision` du serveur MLflow local de Nourchenne.

- Run source Colab : `faf38527c8e34d8c9f5d4672500affba`.
- Run local importé : `72e5fc398bf441929066040a3627a3af`.
- Statut local confirmé : `FINISHED`.
- Présence confirmée de `training/weights/best.pt`, `training/weights/last.pt` et `training/results.csv`.

L’import conserve les résultats de l’entraînement et ne nécessite aucun réentraînement.

L’archive originale contient la valeur d’exemple `member = Prenom Nom`. Le prénom et le nom réels doivent être renseignés dans le tag `member` du run local et reportés dans ce document. Cette correction de métadonnée ne modifie ni les poids ni les métriques.

L’archive originale est conservée comme référence du transfert.

## Points restant à finaliser

1. Reporter le prénom et le nom exacts du membre B après vérification du tag `member`.
2. Confirmer l’application du même protocole de comptage et d’analyse des erreurs à A et C.
3. Vérifier les versions logicielles des expériences à partir de leurs artefacts de provenance et documenter les différences éventuelles.
4. Vérifier le critère exact de sélection de la meilleure époque dans la version d’Ultralytics utilisée.
5. Examiner l’image suspecte et ses annotations, en conservant les résultats principaux sur la validation complète.
6. Compléter la comparaison A/B/C avec les métriques de détection, de comptage et de vitesse.

## Limites

- Les journées sont séparées entre entraînement, validation et test, mais les parkings et les caméras ne sont pas nécessairement distincts entre ces ensembles.
- Les scores obtenus ne démontrent pas la généralisation à une photo de parking inconnu prise avec un téléphone.
- Une annotation absente ou décalée peut influencer l’analyse des erreurs.
- Un comptage exact peut masquer des erreurs de localisation ou de classification qui se compensent.
- Les 1 029 images de test n’ont pas été utilisées. Elles restent réservées à l’évaluation finale selon le protocole commun.
- Ce rapport présente l’expérience B. La comparaison chiffrée avec A et C sera réalisée avec des conditions d’évaluation communes.