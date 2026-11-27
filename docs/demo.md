# Démonstration : fichiers à utiliser

Tout le matériel vient du **lot test de PKLot** (1 029 images jamais vues pendant
l'entraînement ni la validation) : le modèle est donc montré sur des données de sa
distribution. Il est dans `reports/demo/` et se reconstruit avec
`python reports/demo/build_demo_assets.py --ffmpeg <chemin>`.

## Avant la démo

Lancer l'API puis le frontend (voir le README), attendre les deux pastilles vertes
« API online » et « Model ready ».

## 1. Photos : page *Image Analysis*

Les 8 photos sont dans `reports/demo/images/`. `selection.csv` donne, pour chacune, le
nombre de places annotées (libres / occupées). Ordre conseillé :

| # | Fichier (début du nom) | Ce qu'il montre | Vérité terrain (libres / occupées) |
|---|---|---|---|
| 1 | `2012-10-13_06_58_36` | grand parking quasi vide, petit matin | 100 / 0 |
| 2 | `2012-09-14_09_31_12` | même parking presque plein, plein soleil | 7 / 93 |
| 3 | `2012-09-14_13_01_21` | occupation moyenne en milieu de journée | 59 / 41 |
| 4 | `2012-09-14_13_16_21` | lumière très forte, image surexposée | 61 / 39 |
| 5 | `2012-09-14_18_06_32` | fin de journée, lumière faible | 43 / 57 |
| 6 | `2013-03-15_12_00_07` | autre parking (40 places), presque plein | 7 / 33 |
| 7 | `2013-04-13_13_30_08` | 40 places, peu occupé, temps couvert | 23 / 17 |
| 8 | `2013-03-15_18_30_14` | crépuscule, lampadaires allumés | 12 / 28 |

Les colonnes `libres_predites` et `occupees_predites` de `selection.csv` sont à remplir
après un passage dans l'application, pour garder une trace des comptes du modèle.

## 2. Vidéo : page *Video Analysis*

`reports/demo/demo_2013-03-15.mp4` : les 145 photos de la journée du 2013-03-15 (6 h 35
à 18 h 35), du même point de vue, assemblées à 4 images par seconde (36 s,
1280×720, 11,7 Mo, dans les limites de l'API). On voit le parking **se remplir entre
l'aube et 9 h** (0 à 12 s environ), rester plein, puis **se vider en fin de journée**
(40 places annotées, de 1 à 40 occupées).

L'API analyse par défaut **1 image par seconde** (`PARKVISION_VIDEO_SAMPLE_FPS=1`) : la
vidéo donne donc environ 36 images analysées. Pour les analyser toutes, démarrer
l'API avec `PARKVISION_VIDEO_SAMPLE_FPS=4`.

## Journées écartées

Dans le lot test, **2012-10-30 et 2012-11-07** (155 images) sont volontairement
exclues : leurs fichiers de labels déclarent « libre » presque toutes les places alors
que les images montrent un parking plein (1 à 2 % d'occupation annotée sur des jours
de semaine). La vérité terrain y est donc fausse et ne sert pas à la démo.

## À ne pas montrer

Une photo prise hors PKLot (autre parking, autre caméra) : le modèle y détecte très peu
de places (voir la limite de généralisation dans le rapport).
