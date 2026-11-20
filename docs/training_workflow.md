# ParkVision : entraînement et transfert MLflow

## État de cette livraison

Ces scripts complètent le dépôt au commit `fa0b526`. Ils ne modifient pas les
scripts de préparation, le manifeste, les configurations ou les données brutes.
L'entraînement réel reste à vérifier sur la T4 ; aucun résultat de performance
n'est fourni par ce pack.

Validation du pack : quatre tests exécutés avec succès sous Linux avec MLflow
3.17.2. Ils couvrent l'isolation des copies de travail, le refus des journées
communes et des listes incohérentes, puis un aller-retour entre deux bases SQLite
avec historique, fichiers, détection de corruption et refus des doubles imports.
L'exécution sur Windows et l'entraînement Ultralytics restent à valider chez vous.

- `src/models/train.py` : vérification, essai court, entraînement et export final.
- `src/utils/transfer_run.py` : export/import portable d'un run MLflow.
- `tests/test_training_tools.py` : tests du transfert et de la préparation.

Les commandes ci-dessous se lancent depuis la racine du dépôt ParkVision.

## 1. Vérification Windows (maintenant)

```powershell
.\.venv\Scripts\python.exe src/models/train.py --check
```

Cette commande utilise PyYAML, sans charger PyTorch ou Ultralytics. Elle vérifie
les classes, les correspondances manifeste/listes YOLO, la présence des images
et annotations train/validation, l'absence de doublons et de journées communes.
Elle n'ouvre pas les images du test et n'entraîne aucun modèle. Les contrôles
complets des coordonnées et de la qualité des annotations ont été réalisés
dans la phase précédente ; `--check` ne les remplace pas.

Résultats attendus : train 8613, valid 2500, test 1029.

Tests logiciels facultatifs (aucun entraînement) :

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -p test_training_tools.py -v
```

## 2. Fichiers à ignorer dans Git

Ajouter ces lignes au `.gitignore` existant, sans remplacer son contenu :

```gitignore
# Sorties locales des experiences ParkVision
/runs/
/tracking/
/exports/
/yolov8*.pt
__pycache__/
```

Le serveur MLflow Windows conserve sa base et ses fichiers dans
`%LOCALAPPDATA%\ParkVision\mlflow`, comme configuré précédemment.
Les poids seront versionnés avec DVC dans une étape ultérieure : ne pas les
ajouter directement dans Git. Il n'y a pas encore de remote DVC configuré.

## 3. Colab : essai court (quand le GPU redevient disponible)

Récupérer d'abord le dépôt à jour et les données. Reproduire `prepare_yolo`
avec DVC puis vérifier le GPU. Le fichier `requirements-training.txt` fixe
les outils du projet ; PyTorch/CUDA restent ceux du runtime, à vérifier.

```python
%cd /content/ParkVision
%pip install -r requirements-training.txt
```

```python
import subprocess
import sys

subprocess.run([
    sys.executable, "src/models/train.py", "--smoke"
], check=True)
```

L'essai utilise une époque, 128 images d'entraînement et 32 de validation,
échantillonnées avec la graine de la configuration. Il conserve la résolution
640 et le batch 16 afin de tester le chargement, le GPU et la mémoire.
Warmup et fermeture de mosaic sont désactivés pour ce seul essai.
L'expérience MLflow est `ParkVision_Verifications` ; ses scores ne servent pas
à comparer les modèles. Le vrai entraînement repartira des poids préentraînés.

La copie des seules images/annotations utilisées est créée dans
`runs/datasets/<run_id>/`. Cela consomme de l'espace supplémentaire (environ la
taille des images sélectionnées), mais protège `data/raw` des caches et des
éventuelles corrections de fichiers effectuées par YOLO. Aucun test n'est copié.

Le script enregistre dans MLflow : paramètres effectifs, versions, commit Git,
empreinte du manifeste, code exécuté, historique des pertes/métriques par époque,
résultats CSV, graphiques et poids disponibles. La validation du meilleur modèle
est enregistrée séparément avec le préfixe `best_val/`.

Le suivi Colab se trouve dans `/content/parkvision_tracking` par défaut.
L'intégration MLflow automatique d'Ultralytics est désactivée : le script écrit
explicitement dans un unique run.

## 4. Télécharger l'export

Après l'essai, copier le chemin EXACT imprimé après `EXPORT A TELECHARGER` :

```python
from google.colab import files
files.download("/content/ParkVision/exports/NOM_EXACT_IMPRIME.zip")
```

L'export contient `metadata.json` et `artifacts/`. Il préserve toutes les étapes
et dates des métriques, les paramètres, les tags utiles et les fichiers avec
leurs empreintes SHA256. Il ne contient ni images du dataset ni base SQLite.

**Créer le ZIP dans Colab ne le sauvegarde pas sur le PC. Il faut le télécharger.**
La fermeture brutale du runtime ne permet pas d'exécuter l'export final. Les
résultats restent temporaires tant qu'ils n'ont pas été téléchargés. Ce workflow
ne promet pas de sauvegarde automatique hors de Colab à chaque époque.

Si l'entraînement échoue avec une exception Python, le script tente de conserver
les sorties déjà présentes et d'exporter le run en échec. Une suppression de la
machine ou un arrêt forcé du processus ne bénéficie pas de cette protection.

## 5. Importer sur Windows

Laisser le serveur MLflow Windows ouvert sur http://127.0.0.1:5000.
Dans une seconde fenêtre PowerShell :

```powershell
.\.venv\Scripts\python.exe src/utils/transfer_run.py inspect "$env:USERPROFILE\Downloads\NOM_EXACT_IMPRIME.zip"
```

`inspect` est en lecture seule : il valide le ZIP local, recherche toutes les
pages de runs ayant le même identifiant source et vérifie les paramètres,
métriques et empreintes des artefacts d'un import marqué terminé. Il peut donc
télécharger les artefacts distants, sans les modifier. `import` sans option est
également en lecture seule. Les identifiants de connexion sont chargés depuis
le `.env` à la racine du projet via `python-dotenv`, sans être affichés.

Après inspection, une écriture nécessite `import ... --write`. Un import terminé
et vérifié pour le même identifiant source **et** la même empreinte ZIP est
réutilisé. Un import incomplet bloque la création d'un nouveau run. Pour
reprendre explicitement celui-ci, utiliser `import ... --write --resume-run-id ID` :
les éléments déjà présents sont vérifiés, seuls les éléments manquants sont
envoyés, puis le run est finalisé. Un élément présent mais différent bloque la
reprise. Les écritures échouées ne sont pas retentées automatiquement ; relancer
`inspect` avant toute nouvelle action. Les opérations réseau ont des délais
bornés et seules les lectures temporaires sont retentées au maximum trois fois.
Un nouveau run créé avec `--write` apparaît dans l'expérience de même nom,
vue **Model training**.

Les poids transférés restent des artifacts. L'enregistrement d'un modèle
MLflow séparé dans le Model Registry est décrit dans le
[guide Phase 2](model_registry_teammate_guide.md). Ce transfert ne migre pas
un Model Registry et ne réalise pas d'export ONNX.

## 6. Entraînement complet (après validation de l'essai ET de l'import)

```python
subprocess.run([
    sys.executable, "src/models/train.py", "--config", "configs/train_a.yaml"
], check=True)
```

50 époques maximum et patience 10, expérience `ParkVision`. Après exécution,
télécharger le ZIP et l'importer sur Windows. Ne pas lancer cette commande avant
la validation de l'essai court. Aucun entraînement complet CPU n'est autorisé
par défaut dans ce script. Un essai CPU reste possible avec `--smoke --device cpu`
après installation locale des dépendances, mais il sera plus lent.

## Limites à garder dans le rapport

- Les journées sont séparées, pas nécessairement les parkings/caméras. Cela ne
  prouve pas la généralisation à un parking inconnu ou à toute photo de téléphone.
- Certaines places visibles dans PKLot ne sont pas annotées. Elles peuvent être
  considérées comme fond pendant l'entraînement. Les métriques de détection et
  le futur comptage doivent être interprétés dans ce périmètre d'annotation.
- Le MAE de comptage, l'analyse des erreurs, la latence, ONNX et le registre sont
  des étapes à réaliser ; ce script ne prétend pas encore les couvrir.
- A/B/C utilisent la même version de données et un protocole comparable ; la
  sélection se fait sur validation et l'évaluation finale sur test.
- La graine fixée facilite la répétition ; elle ne garantit pas des résultats
  identiques entre toutes les versions de matériel, pilotes et bibliothèques.
