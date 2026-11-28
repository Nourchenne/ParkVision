# Audit des journées : qualité des labels

Le découpage de PKLot se fait par **journée**. Une journée dont les labels sont faux
fausse donc d'un coup toutes ses images (jusqu'à plus de 150). Cet audit calcule, pour
chacune des 98 journées, l'occupation annotée et signale celles qui sont anormales.

- Script : [`src/analysis/audit_days.py`](../src/analysis/audit_days.py) (lecture seule,
  aucun label n'est modifié) : `python src/analysis/audit_days.py`
- Résultat : [`reports/data_audit/journees.csv`](../reports/data_audit/journees.csv)
- Vérification visuelle : `reports/data_audit/verification_*.jpg`
- Test : `tests/test_audit_days.py` (données synthétiques)

## Critère

Pour chaque journée du manifeste (`data/processed/split_manifest.csv`, train, valid et
test confondus), on lit les fichiers de labels YOLO et on calcule
`occupation = places occupées / places annotées`.

Une journée est **suspecte** si c'est un jour de semaine (lundi à vendredi) dont
l'occupation annotée est **inférieure à 10 %**.

Pourquoi ce seuil : sur les 63 jours de semaine non signalés, l'occupation va de 15,6 %
à 97,8 % (médiane 79,9 %). Le plus bas des jours signalés est à 6,8 %. Il n'y a rien
entre 6,8 % et 15,6 %, donc le seuil de 10 % tombe dans un vide naturel et n'est pas
sensible à sa valeur exacte. Les week-ends ne sont pas jugés : ils sont naturellement
peu occupés (jusqu'à 0 %, et jusqu'à 69 %).

## Ce qui ressort

98 journées, 12 142 images (69 journées train, 19 valid, 10 test). **9 journées sont
signalées** (1 050 images). Une vérification visuelle (boîtes de labels dessinées sur
des images de chaque jour) les sépare en deux groupes :

| Journée | Jour | Split | Images | Occupation annotée | Verdict |
|---|---|---|---|---|---|
| 2012-10-30 | mardi | test | 53 | 1,4 % | **labels faux** : parking plein, places étiquetées « libre » |
| 2012-11-06 | mardi | train | 99 | 1,4 % | **labels faux** : idem |
| 2012-11-07 | mercredi | test | 102 | 1,7 % | **labels faux** : idem |
| 2012-10-12 | vendredi | train | 154 | 0,0 % | parking réellement vide (jour férié brésilien) |
| 2012-10-15 | lundi | train | 157 | 2,7 % | parking réellement vide (cause non établie) |
| 2012-12-24 | lundi | train | 156 | 0,0 % | parking réellement vide |
| 2012-12-25 | mardi | valid | 134 | 0,0 % | parking réellement vide (Noël) |
| 2012-12-26 | mercredi | valid | 137 | 2,1 % | parking réellement vide |
| 2012-12-27 | jeudi | train | 58 | 6,8 % | parking réellement vide |

Les trois journées à **labels faux** montrent un parking de 100 places rempli de
voitures, avec presque toutes les places annotées « libre » (98 à 99 libres sur 100). Les
boîtes tombent au bon endroit sur les places : le fichier de labels correspond bien à
l'image, ce sont les **classes** qui sont fausses. Ces jours sont des mardis et mercredis
ordinaires, et leurs voisins sur le même parking sont cohérents (lundi 29 octobre à
69 %, mercredi 31 octobre à 59 %).

Les six autres jours sont réellement vides : les images le montrent, et les labels sont
justes. Le critère les signale quand même, car il ne lit que les labels. C'est pourquoi la
colonne `verification` du CSV (`labels_faux`, `vide_reel`, ou `non_verifie` pour un
futur jour signalé) vient d'un contrôle visuel et non du calcul. Le pipeline se comporte
bien sur ces jours vides : le modèle prédit correctement les 25 et 26 décembre.

### Conséquences

| Lot | Images touchées | Part |
|---|---|---|
| test | 155 (2012-10-30 et 2012-11-07) | **15,1 %** des 1 029 images |
| train | 99 (2012-11-06) | 1,1 % des 8 613 images |
| valid | 0 | 0 % |

- **Les métriques du lot test sont affectées.** Pour l'évaluation finale, on donne deux
  chiffres : le lot test complet (1 029 images) et le **sous-ensemble propre de 874
  images** (sans les deux journées à labels faux), avec l'écart expliqué.
- **Le jeu de données n'est pas propre** : 99 images d'entraînement ont des labels faux.
  L'impact sur le modèle est probablement faible (1,1 %), mais on ne peut pas écrire que
  les données sont sans erreur.
- **Rien n'est corrigé.** Réannoter est impossible : 155 images à une centaine de places
  représentent environ 15 000 boîtes. Les labels d'origine sont conservés.

## Limites de l'audit

- Le critère ne détecte que les journées **presque vides**. Une journée à moitié fausse,
  ou des erreurs isolées dans les images, passent sous le radar.
- Le calcul seul ne sépare pas un label faux d'un jour réellement vide : il faut regarder
  les images, une fois, pour chaque journée signalée (9 contrôlées ici).
- Les jours fériés ne sont pas listés dans le script : les causes ci-dessus sont des
  observations, pas des règles.
