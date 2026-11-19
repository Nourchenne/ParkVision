# Audit ParkVision — Registry et traçabilité

Date : 19 novembre 2026. Code audité : `feature/mlops-phase2-registry`, commit `3f46d6144c814dfeceaf43b64fa6ca22ceebd085`. Main observé : `dde7a418a5330c52ebed2622f3bf95041c18533d`.

## Conclusion

Le Registry est implémenté dans la branche du membre C. Le guide annonce `ParkVision-Detector` v1 pour C, sans alias champion. Cette présence distante reste à relire par API : les archives de preuves reçues couvrent les runs, pas le contenu actuel du Registry. La politique de promotion est désactivée. Le workflow GitHub exécute des tests ; il ne déploie pas une application.

## Ce qui a été vérifié

- 75 tests du code reçu passent sous Python 3.12. Ruff passe. Le test MLflow local utilise SQLite ; les tests Registry utilisent des clients simulés et un détecteur simulé. Aucun accès distant ni entraînement lors de cet audit.
- DVC : la première collecte du 19 novembre rapporte un état local et distant sans différence. La deuxième collecte utilise `--skip-dvc` et fournit les runs distants. Ce sont deux preuves complémentaires, pas un nouveau contrôle DVC en direct.
- Les trois manifests archivés ont le même SHA-256 ; les listes retenues correspondent à 8 613 images train et 2 500 valid. Le manifeste conserve 1 029 images test. Les jours sont disjoints entre les trois splits.
- Les trois CSV de comptage couvrent exactement les mêmes 2 500 noms de fichiers valid. Les MAE ci-dessous sont cohérentes avec les comptes par image ; les exactitudes conjointes ont été recalculées.
- Les trois runs observés sont FINISHED, import_complete=true, test_evaluated=false ; membres Nourchenne, Yomna et Mohamed. Le tag test constitue une déclaration de provenance, pas une preuve absolue de toutes les opérations passées.
- Les branches et commits de contribution existent. Une branche fictive d’entraînement A ou une réécriture des commits historiques n’est pas nécessaire.

## Résultats existants et seuils

| Modèle | mAP50-95 | MAE libres | MAE occupées | Comptage exact des deux classes | Seuils actuels |
|---|---:|---:|---:|---:|---|
| A — YOLOv8n | 0,91869 | 0,4920 | 0,3176 | 78,48 % | Échec des deux MAE |
| B — YOLOv8s | 0,93263 | 0,0784 | 0,0620 | 91,76 % | Respectés |
| C — YOLO26n | 0,91861 | 0,2652 | 0,1496 | 76,12 % | Respectés |

Les seuils restent mAP ≥ 0,85 ; MAE libres ≤ 0,35 ; MAE occupées ≤ 0,25 ; exactitude ≥ 0,74. Ils sont provisoires. B est le meilleur des trois sur les résultats de comptage observés, mais ce constat ne constitue pas une promotion ni un déploiement. A reste conservé comme baseline.

## Corrections livrées

1. **Python 3.10** : `datetime.UTC` empêchait l’import de registry.py sur l’environnement de Nourchenne. Utilisation de `timezone.utc`. La CI est étendue à Python 3.10 et 3.12 ; seule la version 3.12 a été exécutée ici.
2. **Validation complète** : avant correction, un test avec une seule image et un faux manifeste non CSV obtenait verified=true et comparison_verified=true. Le Registry vérifie maintenant les noms exacts de toutes les images valid, leur unicité et le découpage du manifeste.
3. **Protocole approuvé** : `configs/evaluation_protocol.yaml` fixe le manifeste, 2 500 images, conf=0,25, NMS IoU=0,7, appariement IoU=0,5 et imgsz=640. Le Registry refuse les preuves qui divergent. Le CLI historique validate_model reste un contrôle générique des métriques ; son succès seul ne signifie pas admissibilité au Registry.
4. **DVC Windows/Linux** : l’identité de comparaison utilise le hash du répertoire DVC et ses effectifs, au lieu du SHA du texte YAML. Le SHA exact du fichier reste conservé comme preuve. Les variantes LF/CRLF du même fichier sont acceptées pour la preuve d’environnement ; une autre identité de données ne l’est pas.
5. **Robustesse** : lecture de toutes les pages des versions du Registry, rejet des entrées ZIP dupliquées, création du dossier de sortie de prepare.
6. **Tests** : ajout de régressions pour les jeux partiels, images test/étrangères, doublons, mauvais seuils, mauvais manifeste, identité DVC, fins de ligne et pagination. Résultat final : 87 tests passent, 32 sous-tests ; Ruff et compilation passent.

Les changements sont fournis dans un patch. Ils ne sont pas encore appliqués au dépôt Windows, poussés sur GitHub ou validés par sa CI. Aucun run, poids, modèle enregistré, alias ni donnée n’a été modifié.

## Identité des données

```text
manifest_sha256 = b51c7072a9276eff8c69d9023849979efaecf1aadd458f31c2e81651ee76630a
raw_dvc_md5 = 7ebd4f8ce4ef27bdc80d05e92882a012.dir
raw_size = 1817499248
raw_nfiles = 24835
```

Les 16 775 objets transférés par DVC ne représentent pas le nombre d’images du dataset. Il y a 12 142 images retenues dans le manifeste. Le hash DVC enregistré et le manifeste sont cohérents ; l’état exact de chaque image/annotation au moment de chaque ancien entraînement n’est pas prouvé par un contrôle DVC effectué à cette date. Ne pas inventer cette preuve rétroactivement.

## Réutiliser les bons exports

Les runs distants existants proviennent des réexports MLflow locaux de Nourchenne. Le ZIP Colab original d’un collègue peut avoir un autre source_run_id et ne pas être réutilisable directement. Ne pas réimporter automatiquement : inspecter d’abord, puis réutiliser le ZIP exact dont le hash correspond. Un nouvel export peut avoir un hash ZIP différent même avec un contenu similaire.

| Modèle | Run DagsHub | Source du réexport local | SHA-256 du ZIP importé |
|---|---|---|---|
| A | `a0352db5ea4d48be859baab5f4ec2553` | `1261806fd2374ed2950dce8896dda45a` | `970f707ebcf813af986eff9e62f6df9629ea6ddf502ee35dcf10bc88fc1e624c` |
| B | `b80b9226dbc246bcae7fca13b902cf06` | `72e5fc398bf441929066040a3627a3af` | `8c336584f65cbf285e3f90c35d0b00d4478f7f6ecaced3696af55e071d16b7f0` |
| C | `ae1ab363823f48a78622d56959e0cdbc` | `57abc8010d604c09960d6c787c79e800` | `534bb0b7c4cf4834e0fea596e7797a782cd31136e051e46fe5682c9a9b0ca0fd` |

## Limites et suite de l’intégration

- La branche Registry part d’avant la configuration DVC distante et le jalon. Fusionner `origin/main` dans la nouvelle branche de correction, puis vérifier `.dvc/config`, les documents du jalon et les tests. Un `.dvc/config` vide dans le ZIP Registry n’est pas une suppression prouvée : c’est la base ancienne de cette branche.
- Le schéma de protocole devient version 2. Les anciens candidats doivent être recalculés. L’ancienne version C reste intacte ; le contrôle d’anti-duplication refusera un candidat différent pour le même run/poids. Inspecter sa preuve puis préparer une migration explicite avant toute promotion. Ne pas modifier ses tags pour prétendre qu’elle a passé les nouveaux contrôles.
- Les fichiers historiques d’environnement d’analyse ne sont pas tous au nouveau format. Celui d’A est `environment_analysis.json` ; B/C n’en ont pas dans la collecte. Ne pas fabriquer ces fichiers ni confondre environnement d’entraînement et d’évaluation. Sans fichier valide, l’enregistrement peut rester comparison_verified=false, et la promotion automatique doit rester bloquée.
- Si la comparaison impose un environnement et un script communs, une réévaluation de best.pt sur valid peut être nécessaire. C’est de l’inférence, pas du réentraînement ; garder les anciens résultats dans leurs dossiers.
- Les environnements d’entraînement diffèrent : A PyTorch 2.11.0+cu130 / T4 ; B 2.14.1+cu130 / T4 ; C 2.11.0+cu128 / RTX 4060 Laptop. Ne pas présenter leurs durées comme un benchmark matériel identique.
- Les scripts train/analyze historiques n’imposent pas encore un contrôle DVC avant exécution et train conserve un nom de membre par défaut. Ces corrections de traçabilité pour les prochaines opérations restent à intégrer après cette branche ; elles n’imposent pas de relancer les entraînements existants.
- Le Registry utilise un verrou fichier local. Il ne synchronise pas les ordinateurs de plusieurs membres. Avant automatisation, centraliser les écritures dans un seul workflow avec concurrence contrôlée et secrets GitHub, et interdire les écritures simultanées hors de ce workflow.
- La politique compare un candidat à un champion ; ce n’est pas encore un classement global automatique de toutes les versions. Définir les candidats éligibles, le critère et le départage avant l’activer.
- Le packaging ne fige pas entièrement PyTorch et pandas. Avant servir le modèle, produire un environnement d’inférence reproductible et tester le vrai checkpoint, pas seulement le détecteur simulé.
- Restent : lecture du Registry distant, harmonisation des preuves, sélection/alias, API et tests d’inférence réels, conteneur, cible de déploiement, puis workflow CD. La passation à B sera préparée avec les interfaces et responsabilités stabilisées.
- Aucun entraînement ni évaluation du lot test n’est nécessaire pour appliquer ce correctif.

## Archives utilisées

- `ParkVision_registry_audit.zip` — SHA-256 `bcb1898eb1a5791192cc55853b8eabb3b72277aa08adf01ec90c5212d84fe3fe`
- `ParkVision_preuves_20261119_172815_138067.zip` — SHA-256 `705b788cd0ddb3a057f078bd6e1f7b5e8dfdc1aec1a885810d5fffc0babf762a`
- `ParkVision_preuves_20261119_171731_463912.zip` — SHA-256 `7ce7d00240a42cc0cae96c80cf47761a0689513f9d4a945ff44467d501447c23`
