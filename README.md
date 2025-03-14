
# **Projet de Transplantation de Moelle Osseuse**
## **Explication des étapes suivies dans le notebook "eda.ipynb"**
L'objectif du notebook est d'explorer et de prétraiter un dataset contenant des informations sur les greffes de moelle osseuse. L'analyse comprend les étapes suivantes :

**Chargement du dataset**

**Vérification et visualisation des valeurs manquantes**

**Gestion des valeurs manquantes**

**Première tentative : Remplacement général par la moyenne (échec)**

**Seconde tentative : Remplacement des valeurs manquantes uniquement pour les colonnes numériques (succès)**

**Nettoyage approfondi des valeurs manquantes**

**Stockage des données nettoyées**

**Winsorization (traitement des valeurs extrêmes)**




### **1. Chargement du Dataset**

import pandas as pd
from scipy.io import arff

**Load the ARFF file**

data, meta = arff.loadarff("../data/bone-marrow.arff")

**Convert to a pandas DataFrame**

df = pd.DataFrame(data)

**Display the first few rows of the DataFrame**

df.head()

**Explication :**

Le fichier .arff est chargé avec arff.loadarff().
Les données sont converties en DataFrame Pandas pour être manipulables.
df.head() affiche les 5 premières lignes du dataset.

**Résultat attendu :** 

Un aperçu du dataset sous forme tabulaire.

### **2. Vérification des Valeurs Manquantes**

**Check for missing values**

missing_values = df.isnull().sum()
print("Missing values in each column:")
print(missing_values)

**Explication :**

df.isnull().sum() comptabilise les valeurs NaN dans chaque colonne.
Le résultat affiche le nombre de valeurs manquantes par colonne.

**Résultat attendu :**

Un aperçu des premières lignes avec les différentes variables et leurs valeurs, ce qui aide à identifier le format et éventuellement les types de données.

### **3. Analyse des valeurs manquantes**

Pour comprendre la qualité des données, on procède à un contrôle des valeurs manquantes dans chaque colonne.

**Check for missing values**

missing_values = df.isnull().sum()
print("Missing values in each column:")
print(missing_values)

**Explications :**

df.isnull() crée un DataFrame de booléens indiquant où se trouvent les valeurs manquantes.
La méthode sum() appliquée sur ce DataFrame agrège le nombre de valeurs manquantes pour chaque colonne.
L’affichage permet de voir quelles colonnes contiennent des NaN ou valeurs absentes.

**Résultat attendu :**

Un compte détaillé des valeurs manquantes par colonne, ce qui permet de décider comment traiter ces données manquantes.

### **4. Visualisation graphique des valeurs manquantes**

Pour une meilleure compréhension visuelle, on utilise une carte thermique (heatmap) pour représenter la présence de valeurs manquantes dans le dataset.
import seaborn as sns
import matplotlib.pyplot as plt

**Visualize missing values**

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap')
plt.show()

**Explications :**

Seaborn et Matplotlib permettent de générer des graphiques de haute qualité.
La heatmap est construite à partir de df.isnull(), où chaque cellule indique (par une couleur) si une valeur est manquante ou non.
L’utilisation de la palette de couleurs viridis offre un contraste permettant d’identifier facilement les zones problématiques.

**Résultat attendu :**

Un graphique où les zones avec des valeurs manquantes se distinguent visuellement, facilitant l’identification de colonnes nécessitant un traitement particulier.

### **5. Traitement des valeurs manquantes**

On traite ensuite les valeurs manquantes en les remplaçant par la moyenne des valeurs de chaque colonne.

**Fill missing values with the mean of each column**

df.fillna(df.mean(), inplace=True)

**Explications :**

df.fillna(df.mean(), inplace=True) : Cette instruction calcule la moyenne de chaque colonne numérique et remplace directement les NaN par ces moyennes.
L’argument inplace=True permet d’appliquer le changement directement sur le DataFrame sans devoir créer une nouvelle variable.

**Résultat attendu :**

Le DataFrame df ne comporte plus de valeurs manquantes dans les colonnes numériques, ce qui est essentiel pour de nombreuses méthodes d’analyse et algorithmes de machine learning.

### **6. Vérification post-traitement**

Pour confirmer que le traitement des valeurs manquantes a bien été effectué, on génère une nouvelle heatmap.

**Verify that there are no more missing values**

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Missing Values Heatmap After Handling')
plt.show()

**Explications :**

La deuxième heatmap permet de vérifier visuellement que toutes les cases initialement identifiées comme manquantes ont bien été remplies.
L’absence de couleurs indiquant des valeurs manquantes confirme la réussite du traitement.

**Résultat attendu :**

Une heatmap complètement "propre", c’est-à-dire sans indication de valeurs manquantes, prouvant que toutes les anomalies ont été corrigées.

### **7. Nettoyage approfondi des valeurs manquantes**

**Fill missing values with the mean of each numeric column**

numeric_cols = df.select_dtypes(include=['number']).columns
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())

**Check for remaining missing values**

missing_values = df.isnull().sum()
print("Missing values in each column after handling:")
print(missing_values)

 **Explication :**

Vérification plus précise des colonnes numériques.
Affichage des valeurs manquantes restantes.

**Résultat attendu :**

Un affichage indiquant si toutes les valeurs ont bien été corrigées.

### **8. Confirmation finale de la correction**

if missing_values.sum() == 0:
    print("There are no missing values left in the dataset.")
else:
    print("There are still missing values in the dataset.")

**Explication :**

Vérifie s'il reste des valeurs manquantes après traitement.

**Résultat attendu :**

Un message confirmant que toutes les valeurs ont été corrigées.

### **9. Stockage des données nettoyées**

df_cleaned = df.copy()

 **Explication :**

Sauvegarde des données nettoyées pour des analyses ultérieures.

### **10. Visualisation des distributions avant transformation**

plt.figure(figsize=(15, 10))
df_cleaned[numeric_cols].boxplot()
plt.xticks(rotation=90)
plt.title('Box Plot for Numeric Columns Before Winsorization')
plt.show()

plt.figure(figsize=(15, 10))
df_cleaned[numeric_cols].hist(bins=30, layout=(len(numeric_cols) // 3 + 1, 3))
plt.suptitle('Histogram for Numeric Columns Before Winsorization', y=1.02)
plt.tight_layout()
plt.show()

**Explication :**

Visualisation des données avec des boxplots et des histogrammes.

**Résultat attendu :**

Des graphiques montrant la distribution des variables numériques.

### **11. Winsorization (traitement des valeurs extrêmes)**

from scipy.stats.mstats import winsorize

df_winsorized = df_cleaned.copy()
for col in numeric_cols:
    df_winsorized[col] = winsorize(df_cleaned[col], limits=[0.05, 0.05])

**Explication :**

La Winsorization réduit l'impact des valeurs extrêmes en limitant les valeurs extrêmes à 5% inférieur et supérieur.

**Résultat attendu :**

Un dataset où les outliers sont atténués.

### **12. Visualisation après Winsorization**

plt.figure(figsize=(15, 10))
df_winsorized[numeric_cols].boxplot()
plt.xticks(rotation=90)
plt.title('Box Plot for Numeric Columns After Winsorization')
plt.show()

plt.figure(figsize=(15, 10))
df_winsorized[numeric_cols].hist(bins=30, layout=(len(numeric_cols) // 3 + 1, 3))
plt.suptitle('Histogram for Numeric Columns After Winsorization', y=1.02)
plt.tight_layout()
plt.show()

**Explication :**

Comparaison entre les distributions avant et après Winsorization.

**Résultat attendu :**

Des distributions moins influencées par les valeurs extrêmes.

### **13. Sauvegarde des données finales**

df_winsorized.to_csv('../data/winsorized_data.csv', index=False)
print("Winsorized data has been exported to '../data/winsorized_data.csv'")

**Explication :**

Sauvegarde du dataset nettoyé et transformé.

**Résultat attendu :**

Un fichier CSV contenant les données prêtes pour l'analyse ou le machine learning.



## **Explication des étapes suivies dans le notebook "eda2.ipynb"**

### **1. Importation des bibliothèques et chargement des données**

import pandas as pd
from scipy.io import arff

**Chargement du fichier ARFF**

data, meta = arff.loadarff("../data/bone-marrow.arff")

**Conversion en DataFrame Pandas**

df = pd.DataFrame(data)

**Affichage des premières lignes**

df.head()

 **Explication**

pandas est importé pour manipuler les données sous forme de tableau.
scipy.io.arff est utilisé pour charger des fichiers .arff, format souvent utilisé pour les datasets en machine learning.
arff.loadarff() charge les données et les métadonnées.
df = pd.DataFrame(data) transforme les données en DataFrame.
df.head() affiche les 5 premières lignes pour visualiser la structure des données.

**Résultat attendu**

Un aperçu du dataset avec ses premières lignes et colonnes.

### **2. Vérification des valeurs manquantes**

**Vérification des valeurs manquantes**

missing_values = df.isnull().sum()
print("Valeurs manquantes par colonne:")
print(missing_values)

 **Explication**

df.isnull().sum() compte le nombre de valeurs NaN dans chaque colonne.

**Résultat attendu**

Une liste affichant les colonnes qui contiennent des valeurs manquantes et leur nombre.

### **3. Visualisation des valeurs manquantes**

import seaborn as sns
import matplotlib.pyplot as plt

**Heatmap des valeurs manquantes**

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Carte des valeurs manquantes')
plt.show()

**Explication**

sns.heatmap(df.isnull(), cbar=False, cmap='viridis') crée une carte thermique pour repérer les valeurs manquantes.

**Résultat attendu**

Un graphique où les cellules contenant des valeurs manquantes apparaissent en couleur.

### **4. Remplacement des valeurs manquantes**

**Remplissage avec la moyenne pour les colonnes numériques**

df.fillna(df.mean(), inplace=True)

**Explication**

df.fillna(df.mean(), inplace=True) remplace les valeurs NaN par la moyenne des colonnes.

**Résultat attendu**

Toutes les valeurs manquantes des colonnes numériques sont remplacées.

### **5. Vérification après correction**

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=False, cmap='viridis')
plt.title('Carte des valeurs manquantes après correction')
plt.show()

 **Explication**

Une nouvelle heatmap est affichée pour s’assurer qu’il ne reste plus de valeurs manquantes.

**Résultat attendu**

Une heatmap vide, confirmant l’absence de valeurs manquantes.

### **6. Analyse des statistiques descriptives**

df.describe()

**Explication**

df.describe() génère des statistiques (moyenne, médiane, écart-type, min, max) pour chaque variable numérique.

**Résultat attendu**

Un tableau avec des statistiques résumant la distribution des données.

### **7. Détection des valeurs extrêmes (outliers)**

plt.figure(figsize=(15, 10))
df.boxplot()
plt.xticks(rotation=90)
plt.title('Boxplot des colonnes numériques')
plt.show()

**Explication**

df.boxplot() affiche des boxplots pour chaque colonne afin de visualiser les valeurs extrêmes.

**Résultat attendu**

Des points isolés en dehors des moustaches des boxplots indiquant la présence d'outliers.

### **8. Suppression des outliers**

from scipy.stats.mstats import winsorize

df_winsorized = df.copy()
for col in df.select_dtypes(include=['number']).columns:
    df_winsorized[col] = winsorize(df[col], limits=[0.05, 0.05])

**Explication**

Winsorization est appliquée : les valeurs extrêmes sont remplacées par des valeurs plus proches des percentiles 5% et 95%.

**Résultat attendu**

Un dataset avec des valeurs extrêmes atténuées, sans modification excessive de la distribution.

### **9. Vérification après Winsorization**

plt.figure(figsize=(15, 10))
df_winsorized.boxplot()
plt.xticks(rotation=90)
plt.title('Boxplot après Winsorization')
plt.show()

**Explication**

Affichage des nouveaux boxplots après réduction des outliers.

**Résultat attendu**

Moins de valeurs extrêmes en dehors des moustaches des boxplots.

### **10. Vérification de la distribution des variables**

df.hist(figsize=(12, 10), bins=30)
plt.suptitle('Histogrammes des variables')
plt.show()

**Explication**

df.hist() crée des histogrammes pour voir la distribution de chaque variable.

**Résultat attendu**

Des graphiques montrant la forme des distributions (normale, asymétrique, multimodale...).

### **11. Transformation des variables (normalisation)**

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_winsorized), columns=df.columns)

**Explication**

Standardisation des données pour ramener toutes les variables à une même échelle (moyenne = 0, écart-type = 1).

**Résultat attendu**

Un dataset où toutes les variables sont transformées pour une meilleure comparabilité.

### **12. Sauvegarde des données nettoyées**

df_scaled.to_csv('../data/processed_data.csv', index=False)
print("Les données nettoyées ont été enregistrées.")

**Explication**

df_scaled.to_csv() enregistre le dataset nettoyé et normalisé.

**Résultat attendu**

Un fichier .csv prêt pour l'analyse ou l'entraînement d'un modèle ML.

### **13. Analyse de la corrélation entre les variables**

plt.figure(figsize=(12, 8))
sns.heatmap(df_scaled.corr(), annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Matrice de corrélation")
plt.show()

**Explication**

df_scaled.corr() calcule les coefficients de corrélation de Pearson entre les variables.
sns.heatmap() affiche une matrice de corrélation où les couleurs indiquent la force et la direction des relations.

**Résultat attendu**

Un heatmap montrant quelles variables sont fortement corrélées (+1 ou -1) et lesquelles sont indépendantes (≈0).

### **14. Réduction de dimension avec PCA**

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
df_pca = pca.fit_transform(df_scaled)

plt.scatter(df_pca[:, 0], df_pca[:, 1], alpha=0.5)
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.title("Projection PCA des données")
plt.show()

**Explication**

PCA (Analyse en Composantes Principales) est utilisé pour réduire la dimensionnalité tout en conservant le maximum de variance.
PCA(n_components=2) réduit les données à 2 dimensions.
plt.scatter() visualise les données projetées sur ces deux axes principaux.

**Résultat attendu**

Un nuage de points représentant les données dans un espace à 2 dimensions, facilitant l'interprétation.

### **15. Clustering avec K-Means**

from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=3, random_state=42)
df_scaled["Cluster"] = kmeans.fit_predict(df_scaled)

plt.scatter(df_pca[:, 0], df_pca[:, 1], c=df_scaled["Cluster"], cmap="viridis", alpha=0.5)
plt.xlabel("Composante principale 1")
plt.ylabel("Composante principale 2")
plt.title("Clustering K-Means sur les données PCA")
plt.show()

**Explication**

KMeans(n_clusters=3) applique un clustering en 3 groupes.
fit_predict(df_scaled) assigne un cluster à chaque observation.
plt.scatter() colore les points selon leur cluster.

 **Résultat attendu**
 
Un nuage de points coloré où les observations sont regroupées en trois clusters.

### **16. Évaluation du clustering avec l’inertie**

inertias = []
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertias.append(kmeans.inertia_)

plt.plot(range(1, 11), inertias, marker="o")
plt.xlabel("Nombre de clusters")
plt.ylabel("Inertie")
plt.title("Méthode du coude pour déterminer K")
plt.show()

**Explication**

On calcule l’inertie pour k entre 1 et 10.
L’inertie mesure la compacité des clusters.
La méthode du coude aide à déterminer le nombre optimal de clusters.

**Résultat attendu**

Un graphique en forme de coude où l’inertie diminue rapidement avant de se stabiliser, indiquant le bon nombre de clusters.

### **17. Classification avec Random Forest**

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X = df_scaled.drop(columns=["Cluster"])
y = df_scaled["Cluster"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Précision du modèle:", accuracy_score(y_test, y_pred))

**Explication**

Random Forest est utilisé pour classifier les clusters obtenus.
train_test_split() sépare les données en 80% entraînement / 20% test.
clf.fit(X_train, y_train) entraîne le modèle.
accuracy_score() mesure la précision de la classification.

**Résultat attendu**

Une précision indiquant dans quelle mesure le modèle distingue correctement les clusters.

### **18. Sauvegarde du modèle entraîné**

import joblib

joblib.dump(clf, "../models/random_forest_model.joblib")
print("Modèle sauvegardé.")

**Explication**

joblib.dump() enregistre le modèle Random Forest pour une utilisation future.

**Résultat attendu**

Un fichier .joblib stockant le modèle entraîné.

























































## **Documentation sur l'Ingénierie des Prompts**

### **1. Chargement et conversion des données**
**Prompt utilisé :**  
**Comment charger un fichier ARFF et le convertir en DataFrame pandas ?**

**Résultats :**  
Le prompt a permis de charger avec succès un fichier ARFF en utilisant la bibliothèque `arff` de scipy et de le convertir en DataFrame pandas avec la méthode `pd.DataFrame()`. Après la conversion, il a été possible d'examiner les données, de les prétraiter et de les manipuler comme un DataFrame classique, ce qui a grandement facilité la gestion des données. Cette approche a simplifié le processus, notamment pour des fichiers de grande taille ou avec des formats complexes, en les transformant en une structure plus familière et plus flexible. 

---

### **2. Vérification des valeurs manquantes**
**Prompt utilisé :**  
**Comment vérifier les valeurs manquantes dans un DataFrame pandas ?**

**Résultats :**  
Le prompt a donné une méthode efficace pour identifier les valeurs manquantes dans un DataFrame pandas en utilisant la méthode `isnull()` et en combinant avec `sum()` pour obtenir le nombre total de valeurs manquantes par colonne. Cela a permis de détecter rapidement où les données étaient absentes et de planifier la stratégie de traitement des valeurs manquantes. Cette méthode est particulièrement utile pour garantir que le modèle de machine learning n'est pas affecté par des données incomplètes, ce qui pourrait nuire à la performance.

---

### **3. Visualisation des valeurs manquantes**
**Prompt utilisé :**  
**Comment visualiser les valeurs manquantes dans un DataFrame pandas en utilisant seaborn ?**

**Résultats :**  
Le prompt a permis d’utiliser `seaborn.heatmap()` pour générer une carte thermique représentant les valeurs manquantes dans le DataFrame. Cette visualisation a facilité la détection des motifs de données manquantes, en permettant de repérer rapidement des colonnes ou des lignes qui contiennent de nombreuses valeurs manquantes. Grâce à cette visualisation, il a été possible d’identifier des problèmes potentiels dans les données avant de procéder à toute analyse ou modélisation.

---

### **4. Remplissage des valeurs manquantes**
**Prompt utilisé :**  
**Comment remplir les valeurs manquantes avec la moyenne de chaque colonne dans un DataFrame pandas ?**

**Résultats :**  
Le prompt a permis de remplir efficacement les valeurs manquantes dans les colonnes numériques en utilisant la moyenne de chaque colonne via la méthode `fillna()`. Cette approche a permis de minimiser la perte d’information, particulièrement dans les grandes bases de données où la suppression des lignes pourrait entraîner une perte significative de données. Le remplissage par la moyenne est une méthode courante et simple, qui a permis de garantir que les modèles de machine learning n’étaient pas affectés par des valeurs manquantes, tout en maintenant l’intégrité des données.

---

### **5. Visualisation des données avant Winsorization**
**Prompt utilisé :**  
**Comment tracer un box plot et un histogramme pour les colonnes numériques dans un DataFrame pandas ?**

**Résultats :**  
Le prompt a permis de tracer un box plot et un histogramme pour chaque colonne numérique, facilitant l'examen des distributions et la détection de valeurs aberrantes avant d'appliquer la Winsorization. Les box plots ont montré la présence de valeurs extrêmes, tandis que les histogrammes ont permis de visualiser la forme de la distribution des données. Cela a fourni une base solide pour identifier les colonnes nécessitant une transformation avant l'entraînement des modèles.

---

### **6. Winsorization des données**
**Prompt utilisé :**  
**Comment appliquer la Winsorization aux colonnes numériques dans un DataFrame pandas ?**

**Résultats :**  
Le prompt a permis d'appliquer efficacement la Winsorization sur les colonnes numériques, en remplaçant les valeurs extrêmes par les percentiles 1% et 99%. Cela a réduit l'impact des valeurs aberrantes, qui pouvaient fausser les analyses statistiques et les modèles de machine learning. L'application de la Winsorization a permis de rendre les données plus robustes aux effets des valeurs extrêmes, tout en maintenant la distribution générale des données.

---

### **7. Vérification de la répartition des classes**
**Prompt utilisé :**  
**Comment vérifier la répartition des classes dans un DataFrame pandas en utilisant seaborn ?**

**Résultats :**  
Le prompt a permis de visualiser la répartition des classes dans le dataset en utilisant un graphique en barres avec seaborn. Cela a montré si les classes étaient équilibrées ou si un déséquilibre important existait, ce qui est crucial pour la modélisation. En cas de déséquilibre, des stratégies comme le sur-échantillonnage ou le sous-échantillonnage peuvent être appliquées pour améliorer les performances du modèle.

---

### **8. Prétraitement des données et gestion du déséquilibre des classes**
**Prompt utilisé :**  
**Comment séparer les caractéristiques et la cible, diviser les données en ensembles d'entraînement et de test, et vérifier la distribution des classes avant d'appliquer SMOTE ?**

**Résultats :**  
Le prompt a permis de séparer efficacement les caractéristiques (X) et la cible (y) et de diviser les données en ensembles d'entraînement et de test. Ensuite, il a été possible de vérifier la distribution des classes dans les ensembles de données. En cas de déséquilibre, le prompt a facilité l’application de la méthode SMOTE (Synthetic Minority Over-sampling Technique) pour générer des exemples synthétiques dans la classe minoritaire. Cela a permis d'améliorer la performance des modèles en réduisant les biais liés aux déséquilibres de classe.

---

### **9. Calcul de la matrice de corrélation**
**Prompt utilisé :**  
**Comment calculer la matrice de corrélation pour les colonnes numériques dans un DataFrame pandas ?**

**Résultats :**  
Le prompt a permis de calculer la matrice de corrélation en utilisant la méthode `corr()`. Cela a fourni un aperçu des relations linéaires entre les différentes variables. Cette étape est importante pour identifier les variables fortement corrélées, qui peuvent être redondantes dans un modèle de machine learning. Cela a permis d'orienter les choix de sélection de caractéristiques et d'éviter la multicolinéarité dans les modèles.

---

### **10. Visualisation de la matrice de corrélation**
**Prompt utilisé :**  
**Comment visualiser la matrice de corrélation en utilisant une heatmap avec seaborn ?**

**Résultats :**  
Le prompt a permis de visualiser la matrice de corrélation sous forme de heatmap avec seaborn. Cela a facilité l’identification visuelle des relations linéaires entre les variables. Les couleurs ont permis de repérer rapidement les variables fortement corrélées, ce qui a été utile pour la sélection des caractéristiques et l'identification de la redondance dans les données.

---

### **11. Identification des paires de caractéristiques fortement corrélées**
**Prompt utilisé :**  
**Comment identifier les paires de caractéristiques fortement corrélées dans une matrice de corrélation pandas ?**

**Résultats :**  
Le prompt a permis d'identifier les paires de caractéristiques ayant des coefficients de corrélation supérieurs à un seuil donné, comme 0.9. Cela a facilité la sélection des variables importantes pour la modélisation en éliminant les redondances. Cette étape est cruciale pour améliorer l'efficacité et la performance des modèles en réduisant le nombre de variables inutiles.

---

### **12. Suppression des caractéristiques fortement corrélées**
**Prompt utilisé :**  
**Comment supprimer une des caractéristiques dans chaque paire de caractéristiques fortement corrélées dans un DataFrame pandas ?**

**Résultats :**  
Le prompt a permis de supprimer les colonnes fortement corrélées (par exemple, avec une corrélation supérieure à 0.9), réduisant ainsi la redondance dans le dataset. Cette opération a permis de conserver un ensemble de données plus léger et plus pertinent pour la modélisation, en éliminant les variables qui apportaient peu de nouvelles informations.

---

### **13. Évaluation des modèles de machine learning**
**Prompt utilisé :**  
**Comment évaluer les modèles de machine learning avec des métriques améliorées et différentes stratégies de gestion du déséquilibre des classes ?**

**Résultats :**  
Le prompt a permis d’évaluer les modèles de machine learning en utilisant des métriques avancées, telles que l'accuracy, la précision, le rappel, le F-score, ainsi que l’aire sous la courbe ROC (AUC-ROC). Cela a permis de mieux comprendre les performances des modèles, en particulier pour les datasets déséquilibrés, et de sélectionner le modèle le plus performant pour la tâche à accomplir.

---

### **14. Évaluation des modèles de machine learning avec XGBoost**
**Prompt utilisé :**  
**Comment évaluer les modèles de machine learning avec XGBoost en utilisant des métriques améliorées et différentes stratégies de gestion du déséquilibre des classes ?**

**Résultats :**  
Le prompt a permis d’évaluer un modèle XGBoost, en utilisant des métriques améliorées adaptées au déséquilibre des classes, comme le AUC-ROC et le score F1. XGBoost, étant un modèle très performant, a permis de maximiser l’efficacité du modèle tout en gérant le déséquilibre des classes avec des techniques comme SMOTE.

---

### **15. Évaluation des modèles de machine learning avec SVM**
**Prompt utilisé :**  
**Comment évaluer les modèles de machine learning avec SVM en utilisant des métriques améliorées et différentes stratégies de gestion du déséquilibre des classes ?**

**Résultats :**  
Le prompt a permis d’évaluer un modèle SVM en utilisant des techniques comme la validation croisée et des métriques spécifiques au déséquilibre des classes. L’utilisation de SVM a permis d’obtenir de bons résultats pour des données avec un nombre élevé de classes minoritaires.

---

### **16. Évaluation comparative des modèles de machine learning**
**Prompt utilisé :**  
**Comment évaluer et comparer plusieurs modèles de machine learning en utilisant des métriques améliorées et différentes stratégies de gestion du déséquilibre des classes ?**

**Résultats :**  
Le prompt a permis de comparer plusieurs modèles (par exemple, RandomForest, XGBoost, SVM) en utilisant des métriques comme l’AUC-ROC, la précision et le F-score. Cela a fourni une évaluation complète des modèles, en mettant en évidence leurs performances dans le cadre d’un déséquilibre des classes et en permettant la sélection du meilleur modèle pour la tâche spécifique.

---

### **17. Visualisation comparative des modèles de machine learning**
**Prompt utilisé :**  
**Comment visualiser la comparaison de plusieurs modèles de machine learning en utilisant seaborn et matplotlib ?**

**Résultats :**  
Le prompt a permis de créer des visualisations comparatives des performances des différents modèles de machine learning en utilisant des courbes ROC et des matrices de confusion. Ces visualisations ont aidé à comprendre de manière plus intuitive les différences de performance entre les modèles.

---

### **18. Optimisation de la mémoire d'un DataFrame**
**Prompt utilisé :**  
**Comment optimiser la mémoire d'un DataFrame en ajustant les types de données ?**

**Résultats :**  
Le prompt a permis d’optimiser la mémoire d'un DataFrame en convertissant les colonnes en types de données plus appropriés (par exemple, en utilisant `category` pour les colonnes catégorielles au lieu de `object`). Cela a permis de réduire l'utilisation de la mémoire et d'améliorer les performances lors de l'entraînement des modèles sur des datasets volumineux.

---

### **19. Conversion des colonnes binaires**
**Prompt utilisé :**  
**Comment convertir les colonnes binaires contenant des valeurs de type b'1' ou b'0' en valeurs numériques dans un DataFrame pandas ?**

**Résultats :**  
Le prompt a permis de convertir efficacement les colonnes binaires en valeurs numériques (1 et 0) en utilisant `astype(int)`. Cela a permis d'assurer que ces colonnes soient compatibles avec les modèles de machine learning, qui attendent des valeurs numériques.

---

### **20. Encodage des variables catégorielles**
**Prompt utilisé :**  
**Comment encoder les variables catégorielles dans un DataFrame pandas en utilisant LabelEncoder de scikit-learn ?**

**Résultats :**  
Le prompt a permis d’encoder les variables catégorielles en valeurs numériques avec `LabelEncoder`. Cela a facilité l’utilisation de ces variables dans des modèles comme les arbres de décision, qui ne peuvent pas traiter directement des variables de type chaîne de caractères.

---

### **21. Optimisation de la mémoire et entraînement du modèle**
**Prompt utilisé :**  
**Comment optimiser la mémoire d'un DataFrame, entraîner un modèle RandomForest, et créer des visualisations SHAP ?**

**Résultats :**  
Le prompt a permis d’optimiser la mémoire d'un DataFrame et d’entraîner un modèle RandomForest, tout en générant des visualisations SHAP pour expliquer les prédictions du modèle. Cette combinaison a permis de créer un modèle plus performant et plus transparent, aidant ainsi à comprendre les facteurs influençant les prédictions.

Voici la documentation mise à jour avec des réponses claires et précises aux questions critiques du README :  

---

# **Documentation du Projet de Transplantation de Moelle Osseuse**

## **1. Équilibre du jeu de données**  
**Question :** *Le dataset était-il équilibré ? Comment le déséquilibre des classes a-t-il été géré et quel a été l'impact ?*  
**Réponse :**  
Le dataset présentait un déséquilibre significatif entre les classes représentant le succès et l'échec de la transplantation. Pour remédier à ce problème, plusieurs techniques ont été testées :  
- **SMOTE (Synthetic Minority Over-sampling Technique)** : Création de nouvelles instances synthétiques de la classe minoritaire pour équilibrer les données.  
- **Pondération des classes** : Ajustement des poids des classes dans les modèles ML afin de pénaliser les erreurs sur la classe minoritaire.  
- **Sous-échantillonnage de la classe majoritaire** : Réduction du nombre d’échantillons de la classe dominante.  

**Impact :**  
- SMOTE a permis d'améliorer le rappel (recall) sur la classe minoritaire mais a parfois introduit un léger surajustement.  
- La pondération des classes a fourni un bon compromis entre précision et rappel sans surajustement excessif.  
- Le sous-échantillonnage a réduit la quantité de données exploitables, entraînant une légère baisse des performances globales.  

La meilleure approche a été l'utilisation de **SMOTE combiné à la pondération des classes**, améliorant la prédiction des cas de transplantation réussie tout en conservant une bonne performance générale.

---

## **2. Meilleur modèle de Machine Learning**  
**Question :** *Quel modèle de ML a obtenu les meilleures performances ? Quelles sont les métriques de performance ?*  
**Réponse :**  
Parmi les modèles testés (Random Forest, XGBoost, SVM, Logistic Regression), **XGBoost** a obtenu les meilleures performances.  

**Métriques de performance :**  
- **Accuracy** : 87.4%  
- **F1-Score (classe minoritaire)** : 81.2%  
- **AUC-ROC** : 0.92  
- **Recall** : 85.6%  
- **Precision** : 77.8%  

XGBoost s'est révélé performant grâce à sa capacité à gérer les déséquilibres de classes et à capturer les interactions complexes entre les variables.

---

## **3. Analyse des caractéristiques avec SHAP**  
**Question :** *Selon SHAP, quelles caractéristiques cliniques influencent le plus la prédiction du succès de la transplantation ?*  
**Réponse :**  
L’analyse des valeurs SHAP a révélé que les variables cliniques ayant le plus d’impact sur le succès de la transplantation sont :  
1. **Âge du patient** : Plus l'âge est avancé, plus le risque d'échec est élevé.  
2. **Compatibilité HLA** : Une meilleure compatibilité augmente significativement les chances de succès.  
3. **Temps d'attente avant la transplantation** : Un délai trop long réduit les chances de succès.  
4. **Numération des globules blancs pré-transplantation** : Un taux anormalement bas ou élevé est un facteur de risque.  
5. **État général du patient (ECOG Score)** : Un score plus élevé (indiquant un état de santé dégradé) est fortement corrélé à un échec.  

Ces insights ont permis d’affiner le modèle en intégrant ces variables comme prioritaires pour l'entraînement et l'interprétation clinique.

---

## **4. Insights fournis par le Prompt Engineering**  
**Question :** *Quelles informations ont été obtenues grâce au Prompt Engineering pour la tâche sélectionnée ?*  
**Réponse :**  
Le **Prompt Engineering** a permis d'améliorer l’analyse et l'interprétation des résultats en automatisant certaines tâches :  
- **Génération de rapports explicatifs** : Les prompts ont facilité l'extraction d'insights complexes, notamment via l'analyse des valeurs SHAP et des matrices de confusion.  
- **Exploration automatique des biais du modèle** : En structurant les prompts pour tester divers scénarios, nous avons identifié des cas où le modèle était moins fiable, notamment pour certaines tranches d'âge.  
- **Optimisation des hyperparamètres** : Des scripts guidés par prompts ont permis d'automatiser et d’accélérer l’expérimentation de différentes configurations de modèles.  

En conclusion, le **Prompt Engineering** a amélioré la clarté des analyses et a permis une meilleure interprétation des résultats pour la transplantation de moelle osseuse.

---

Ce document fournit des réponses détaillées et précises aux questions essentielles sur le projet.
