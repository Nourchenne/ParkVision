# Expérience B · YOLOv8s : analyse des erreurs sur le lot valid

Poids `best.pt`, 2 500 images du lot valid de `split_manifest.csv`, confiance 0,25, NMS 0,7,
appariement IoU 0,5. Liste complète des 347 erreurs : [`erreurs_detail.csv`](erreurs_detail.csv).

Les causes ci-dessous sont des **hypothèses tirées de l'observation des images**. Les annotations
n'ont pas été auditées une par une.

Légende des images d'exemple : trait fin vert/rouge = vérité (libre/occupée), trait épais
cyan/magenta = prédiction (libre/occupée), cadre jaune = erreur.

## Vue d'ensemble

| Type | Nombre | Confiance moyenne | Remarque |
|---|---:|---:|---|
| Fausse détection | 196 | 0,61 | 139 « libre » et 57 « occupée » ; 35 viennent d'une seule image |
| Confusion | 88 | 0,89 | 68 occupées prédites libres, 20 libres prédites occupées : erreurs très confiantes |
| Oubli | 40 | n/a | 36 libres, 4 occupées ; 38 viennent d'une seule image |
| Double détection | 23 | 0,36 | proches du seuil 0,25 |

Trois faits à retenir :

- **Une image concentre 73 erreurs** (21 % du total) : `2013-03-02_12_50_07` (exemple 4). Hors de cette image, il reste 274 erreurs.
- **La journée du 2012-10-31 concentre 89 erreurs** sur 135 images (32 % des 274 restantes), surtout dans la séquence de 14:48 à 15:08 (exemples 5, 6 et 8). La même caméra, avec les mêmes places posant problème d'une image à l'autre, indique une erreur **systématique** et non aléatoire.
- Les doubles détections sont peu confiantes (0,29 à 0,56) ; les confusions le sont beaucoup (0,93 à 0,95 dans les exemples 1 et 2).

## Exemples

### 1. Confusion libre/occupée · [confusion_2012-09-15_15_32_22](exemples/confusion_2012-09-15_15_32_22_jpg.rf.883dbe1fda5dd.jpg)
- **Erreur** : deux places annotées *libres* prédites *occupées* (confiance 0,945), au centre de l'image, dans la zone ombrée.
- **Effet sur le comptage** : libres −2, occupées +2.
- **Cause probable** : un objet sombre (voiture ou ombre portée) dans la place, sous une forte ombre de bâtiment. Le modèle y voit un véhicule. **À vérifier** : si une voiture est bien là, l'erreur vient de l'annotation et non du modèle.

### 2. Confusion libre/occupée · [confusion_2012-10-31_09_38_08](exemples/confusion_2012-10-31_09_38_08_jpg.rf.52e75214c92e5.jpg)
- **Erreur** : deux places annotées *occupées* prédites *libres* (0,942 et 0,931). L'une est un sol nu visible, l'autre une voiture partiellement masquée par les véhicules voisins.
- **Effet sur le comptage** : libres +2, occupées −2.
- **Cause probable** : pour la première, une annotation probablement erronée ou périmée (aucune voiture visible) ; pour la seconde, une occlusion par le véhicule voisin. **À vérifier** sur l'annotation d'origine.

### 3. Oubli · [oubli_2012-10-27_06_35_35](exemples/oubli_2012-10-27_06_35_35_jpg.rf.99e9d59f32a59.jpg)
- **Erreur** : une place libre annotée (à l'extrémité droite de la 1ʳᵉ rangée, la plus lointaine) n'est pas détectée.
- **Effet sur le comptage** : libres −1 (99 places libres prédites sur 100).
- **Cause probable** : brouillard dense, contraste très faible, place éloignée située en bout de rangée. Les 99 autres places sont détectées malgré le brouillard : l'oubli reste isolé.

### 4. Oublis et fausses détections simultanés · [oubli_2013-03-02_12_50_07](exemples/oubli_2013-03-02_12_50_07_jpg.rf.c4387345a8537.jpg)
- **Erreur** : 38 oublis et 35 fausses détections dans la même image. Les prédictions « libre » (confiance 0,70 à 0,96) tombent sur des emplacements vides bien visibles, mais pas sur les cadres annotés.
- **Effet sur le comptage** : presque nul (libres −2, occupées −1), car les oublis et les fausses détections s'annulent. Le MAE ne détecte pas ce cas ; seul l'appariement par IoU le révèle.
- **Cause probable** : l'image ressemble à un assemblage de deux vues de perspective différente (couture verticale vers x = 768) ; les annotations semblent **décalées par rapport à l'image**. Il ne s'agit probablement pas d'une faiblesse du modèle. **À signaler à part** : à relire dans le fichier d'annotation, car les prédictions peuvent correspondre à de vraies places visibles hors de la zone annotée. Aucune autre image du lot ne montre ce profil : les 151 autres images de ce jour comptent 40 erreurs en tout.

### 5. Fausse détection · [fausse_detection_2012-10-31_14_48_21](exemples/fausse_detection_2012-10-31_14_48_21_jpg.rf.ead80622d1522.jpg)
- **Erreur** : 7 prédictions « libre » (0,46 à 0,75) sans place annotée correspondante, sur des voitures sombres des rangées 3 et 4.
- **Effet sur le comptage** : libres +7 (38 annotées, 45 prédites).
- **Cause probable** : voitures sombres sous un éclairage dur avec ombres d'arbres, vues par l'arrière, que le modèle prend pour des places libres. **À signaler à part** : certaines places visibles de cette image ne sont pas annotées (les mêmes emplacements sont annotés *occupés* dans les images suivantes, 14:58 et 15:03, exemples 6 et 8) ; ces prédictions sont donc à la fois des erreurs de classe et des places hors annotation.

### 6. Fausses détections et confusions · [fausse_detection_2012-10-31_15_03_22](exemples/fausse_detection_2012-10-31_15_03_22_jpg.rf.36b6975588ea1.jpg)
- **Erreur** : 11 fausses détections « libre » (0,45 à 0,69) sur des voitures de la rangée centrale, et 5 places annotées *occupées* prédites *libres* (0,70 à 0,92), sur les mêmes voitures sombres que dans l'exemple 5.
- **Effet sur le comptage** : libres +16, occupées −5, soit la plus grosse erreur de comptage du lot.
- **Cause probable** : même scène que l'exemple 5 cinq minutes plus tard. Les mêmes places sont fausses dans plusieurs images consécutives (14:48, 14:53, 14:58, 15:03, 15:08), ce qui indique une difficulté propre à cette plage horaire (voitures sombres, ombres d'arbres) et non un hasard.

### 7. Double détection · [double_detection_2012-10-29_16_43_25](exemples/double_detection_2012-10-29_16_43_25_jpg.rf.f79f128a0272e.jpg) et [double_detection_2012-10-29_16_38_25](exemples/double_detection_2012-10-29_16_38_25_jpg.rf.e3c06c1319e4c.jpg)
- **Erreur** : trois prédictions « libre » supplémentaires (0,29 à 0,56) sur des places déjà détectées, à l'extrémité gauche de la 1ʳᵉ rangée (et, pour la deuxième image, au bout de la 2ᵉ rangée).
- **Effet sur le comptage** : libres +3 dans chacune des deux images.
- **Cause probable** : places lointaines, étroites et qui se chevauchent à cause de la perspective, sur un sol surexposé dont les lignes sont délavées. Le modèle propose plusieurs cadres décalés pour la même place, avec des scores proches du seuil.

### 8. Écart de comptage · [ecart_comptage_2012-10-31_14_58_22](exemples/ecart_comptage_2012-10-31_14_58_22_jpg.rf.657154cf93eba.jpg)
- **Erreur** : 2 places annotées *occupées* prédites *libres* (0,81 et 0,76) et 4 fausses détections « libre » (0,54 à 0,85).
- **Effet sur le comptage** : libres +6, occupées −2.
- **Cause probable** : identique aux exemples 5 et 6 (même séquence du 2012-10-31). L'image a été retenue parce qu'elle fait partie des plus grands écarts de comptage.

## Cas à signaler à part (places visibles non annotées)

Selon la consigne du projet, les prédictions qui correspondent à une vraie place visible hors de la zone annotée sont à distinguer des erreurs du modèle. Deux situations sont repérées :

- **Exemple 4** : prédictions confiantes sur des emplacements vides visibles, probablement dues à un décalage ou à une absence d'annotation.
- **Exemple 5** : voitures visibles sans annotation dans cette image, annotées dans l'image suivante.

Le nombre exact de ces cas dans les 196 fausses détections **n'est pas établi** : il demande une relecture visuelle image par image.

## Pistes (non testées)

- Relire les annotations des images concernées (exemples 1, 2, 4 et 5) avant de conclure à une erreur du modèle.
- Les doubles détections ayant une confiance basse (0,36 en moyenne), un seuil de confiance plus élevé les réduirait, mais ferait aussi augmenter les oublis. À évaluer avec Nourchenne sur A, B et C avant de modifier les seuils communs.
- Hypothèse pour la séquence du 2012-10-31 : peu de voitures sombres sous ombre portée dans les données d'entraînement. Non vérifiée : elle demande de regarder les journées d'entraînement de cette caméra.
