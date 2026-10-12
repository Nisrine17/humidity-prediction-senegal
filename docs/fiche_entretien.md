# Fiche de révision — Projet « Humidité au Sénégal »

## 1. Le projet en 30 secondes

« J'ai repris la partie Data Science d'un projet de hackathon : estimer l'humidité relative dans les départements du Sénégal à partir de la température, du vent, des conditions météo et de la localisation. J'ai comparé six modèles avec trois protocoles de validation, dont une validation où chaque date est tour à tour laissée de côté. Le modèle retenu est un Ridge, avec un RMSE moyen d'environ 6 points d'humidité sur une date jamais vue. Je l'ai déployé dans une application Streamlit, en précisant ses limites : seulement cinq journées de données. »

## 2. Notions clés

| Notion | À savoir |
|---|---|
| Problème métier | Estimer l'humidité à partir de conditions observées. Ce n'est pas une prévision du temps futur. |
| Cible | `humidity` (% , variable continue) |
| Variables explicatives | `region`, `departement`, `weather` (catégorielles) ; `temperature`, `wind_speed` (numériques) |
| Régression vs classification | Régression : prédire une valeur continue. Classification : prédire une classe (ex. « sec / humide »). |
| MAE | Erreur absolue moyenne, en points d'humidité, lisible directement. |
| RMSE | Racine de l'erreur quadratique moyenne ; pénalise davantage les grosses erreurs. |
| R² | Part de variance expliquée ; peut être négatif si le modèle fait pire que la moyenne. |
| `StandardScaler` | Centre et réduit les variables numériques ; utile pour les modèles linéaires, pas nécessaire pour les arbres. |
| `OneHotEncoder` | Transforme une catégorie en colonnes 0/1 ; `handle_unknown="ignore"` évite une erreur sur une catégorie jamais vue. |
| `Pipeline` / `ColumnTransformer` | Enchaînent prétraitement et modèle ; garantissent que les transformations sont apprises uniquement sur l'entraînement. |

## 3. Les modèles

- **Régression linéaire / Ridge** : combinaison linéaire des variables ; Ridge ajoute une pénalité L2 qui stabilise les coefficients.
- **Random Forest** : beaucoup d'arbres entraînés en parallèle sur des échantillons différents, prédiction moyennée (réduit la variance).
- **Gradient Boosting** : arbres construits séquentiellement, chacun corrigeant les erreurs du précédent.
- **XGBoost** : implémentation optimisée du gradient boosting (régularisation, parallélisme, gestion fine de la complexité).

## 4. Validation

- **Séparation aléatoire** : des observations proches (même date, même région) sont en train et en test → score optimiste.
- **Séparation temporelle** : on teste sur une date jamais vue → plus proche d'un usage réel, mais une seule date.
- **Une date laissée de côté** : 5 plis, chacun testé sur une date ; donne une moyenne et une dispersion. Limite : seulement 5 plis.
- **Fuite de données** : ajuster scaler/encodeur sur tout le jeu avant la séparation, utiliser un identifiant de date comme variable, choisir les hyperparamètres sur le test.

## 5. Résultats à connaître (voir `README.md`)

- Séparation aléatoire : XGBoost meilleur (RMSE 4,25).
- Validation par date : Ridge meilleur (RMSE moyen 5,98 ± 1,42), XGBoost 6,81, Random Forest 8,53.
- Température seule : RMSE 10,86 ; avec toutes les variables : environ 6.
- Heure : pas d'amélioration (6,17 avec, 5,98 sans).

## 6. Questions probables et réponses types

**Pourquoi n'avez-vous pas retenu XGBoost, qui gagnait dans le script du hackathon ?**
« Il gagnait sur une séparation aléatoire, où des observations très proches sont en entraînement et en test. Quand je teste sur une date jamais vue, les modèles linéaires font mieux. Comme l'objectif est de généraliser, j'ai choisi sur la validation par date. »

**Votre R² est élevé, êtes-vous sûre qu'il n'y a pas de fuite ?**
« J'ai vérifié plusieurs points : le prétraitement est dans le pipeline, je n'utilise ni le jour ni le mois, et je compare trois protocoles. Le score baisse bien quand on teste sur une date non vue. Une grande partie du signal vient de la température : seule, elle donne déjà un R² de 0,74. »

**Pourquoi avoir retiré l'heure ?**
« Je l'ai testée avec et sans. Elle n'améliorait pas la validation par date, et elle est liée aux sessions de collecte, donc risquée à généraliser. J'ai gardé le modèle le plus simple. »

**Ridge ou régression linéaire ?**
« Les deux sont quasi identiques (5,98 contre 6,07 de RMSE moyen) ; avec 5 plis, la différence n'est pas significative. J'ai gardé Ridge pour sa stabilité. »

**Quelles sont les limites ?**
« Cinq journées d'octobre seulement : je ne peux rien affirmer sur d'autres saisons, ni sur la prévision. Peu de plis, pas de réglage d'hyperparamètres, et le modèle a été choisi sur les mêmes plis qui servent à le décrire, donc légèrement optimiste. »

**L'importance des variables prouve-t-elle que la température cause l'humidité ?**
« Non, c'est une association dans ce jeu de données. L'importance par permutation mesure de combien l'erreur augmente quand on mélange une variable. »

**Comment amélioreriez-vous le projet ?**
« Plus de données sur plusieurs saisons, une validation imbriquée pour régler les hyperparamètres, des variables supplémentaires (pression, altitude, distance à la côte) et des intervalles de prédiction. »

**Pourquoi un pipeline ?**
« Pour que scaler et encodeur soient ajustés uniquement sur l'entraînement, et pour sauvegarder un seul objet que l'application peut appeler directement. »
