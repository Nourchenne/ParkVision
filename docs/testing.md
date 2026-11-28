# Dossier de tests

Ce document décrit ce qui est testé dans ParkVision, comment le relancer et ce que la
couverture mesure vraiment. Les sorties brutes du dernier passage sont dans
`reports/testing/`.

## Résultat du dernier passage

| Partie | Tests | Résultat | Preuve |
|---|---|---|---|
| Backend et pipeline (pytest) | 225 collectés | 224 réussis, 1 ignoré | `reports/testing/pytest_coverage.txt` |
| Frontend (vitest) | 17 dans 5 fichiers | 17 réussis | `reports/testing/vitest.txt` |
| Lint frontend (eslint + tsc) | — | sans erreur | `npm run lint` |

Couverture des lignes (`src/` et `orchestration/`) : **81 %** (3 231 lignes, 628 non
couvertes). Aucun seuil n'est imposé dans la CI : la couverture est mesurée et publiée,
pas utilisée comme barrière.

Le test ignoré (`test_symlinked_checkpoint_is_rejected`) exige des liens symboliques,
interdits sans droits élevés sous Windows. Il s'exécute dans la CI Linux.

## Relancer les tests

Backend, depuis la racine, avec l'environnement du backend (fastapi 0.115.12 ; Prefect, qui demande
fastapi >= 0.139, reste dans `requirements-training.txt`, dans un environnement séparé) :

```bash
python -m pytest tests/ -q
python -m pytest tests/ -q --cov=src --cov=orchestration --cov-report=term-missing:skip-covered
```

Frontend :

```bash
cd frontend
npm ci
npm run test     # vitest
npm run lint     # eslint + tsc
```

Aucun test n'a besoin du réseau, d'un jeton DagsHub, d'un GPU ni d'un vrai modèle : le
modèle est remplacé par un faux modèle PyFunc local, et MLflow tourne sur un magasin
temporaire.

## Ce que couvrent les tests

| Domaine | Fichiers | Ce qui est vérifié |
|---|---|---|
| API de prédiction | `test_api.py` | `/health`, `/ready`, `/model-info`, `/predict`, `/predict/visualize`, rejet des images invalides, trop grosses ou absentes, indisponibilité du modèle, sortie de modèle invalide |
| Vidéo | `test_video_api.py`, `test_video_service.py`, `test_video_sessions.py`, `test_youtube_source.py` | envoi d'un MP4, sessions, lecture par plages d'octets, WebSocket, limites (durée, taille, images), annulation, source YouTube |
| Cache et chargement du modèle | `test_model_cache.py` | vérification de l'empreinte du paquet, rejet d'un point de contrôle altéré ou lié par symlink |
| Registre MLflow | `test_registry*.py` | alias `champion`, politique de promotion, verrou, contrat d'adaptateur, variables d'environnement |
| Porte qualité | `test_validate_model.py` | seuils indépendants, preuves de validation (provenance, empreintes, ZIP corrompu, comptes contradictoires) |
| Analyse d'erreurs | `test_error_analysis.py` | appariement par IoU, places manquées, fausses détections, doubles détections, confusions de classe, MAE par classe |
| Export et transfert | `test_export_onnx.py`, `test_transfer_run.py`, `test_training_tools.py` | export ONNX, aller-retour et détection de corruption d'un run transféré |
| Orchestration | `test_prefect_flow.py` | le flux lance `prepare_split` avant `prepare_yolo`, les tâches correspondent aux étapes de `dvc.yaml` |
| Tableau de bord | `test_dashboard.py` | les anciennes routes du tableau de bord ne sont plus servies |
| Frontend | `api.test.ts`, `useMonitoring.test.tsx`, `VideoPlayer.test.tsx`, `ImagePage.test.tsx`, `ModelPage.test.tsx` | client API, restauration de session vidéo, surimpression sur le MP4 ; page image (aperçu, analyse, comptes, tableau des détections, erreurs) ; page modèle (champion chargé, API hors ligne, chargement) |

## Intégration continue

`.github/workflows/ci.yml` lance sur chaque pull request et chaque push sur `main` :

1. travail `quality` (Python 3.10 et 3.12) : compilation, `ruff`, `mypy`, validation du
   schéma de configuration, `pytest` avec couverture, contrôle des espaces. Le rapport
   `coverage.xml` est publié comme artefact (Python 3.12) ;
2. travail `frontend` : `npm ci`, lint, tests, build ;
3. travail `docker-security` : construction de l'image backend et analyse Trivy
   (informative, sans blocage).

Dans la CI, `prefect` n'est pas installé (conflit de version de fastapi avec le backend) :
les 5 tests de `test_prefect_flow.py` y sont donc **ignorés** et la couverture de
`orchestration/` n'y est pas comptée. Ils se lancent en local avec l'environnement
d'entraînement.

## Limites connues

La couverture est inégale, et c'est voulu : l'effort a porté sur ce qui est servi.

| Module | Couverture | Pourquoi |
|---|---|---|
| `src/api/*` (service servi) | 77 à 96 % | cœur du produit, testé avec un faux modèle |
| `src/models/registry*.py`, `transfer_run.py` | 73 à 90 % | logique de promotion et d'intégrité |
| `src/analysis/analyze_errors.py` | 51 % | la partie rapport (tracés, écriture de fichiers) n'est pas testée |
| `src/models/train.py` | 43 % | la boucle d'entraînement YOLO exige des données et du calcul ; elle est validée par les expériences, pas par pytest |
| `src/api/smoke_test.py` | 0 % | script de vérification manuelle, jamais appelé par les tests |

Ce que les tests ne prouvent pas :

- la **qualité du modèle** : elle est mesurée par l'évaluation (`reports/experience_b/`),
  pas par les tests, qui utilisent un faux modèle ;
- le chargement du **vrai** modèle champion depuis DagsHub (il faut un jeton) ;
- le rendu réel : les tests frontend tournent sur jsdom, sans navigateur ni vrai backend, et le frontend n'a pas de mesure de couverture.
