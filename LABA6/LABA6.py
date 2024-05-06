import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import silhouette_score

#1. Из датасета выберите наиболее важные параметры, характеризующие цель 
# исследования и сформируйте из них матрицу X.
data = pd.read_csv('Country-data.csv')
df = pd.DataFrame(data)
X = df[['child_mort', 'income', 'gdpp']]
print(X.head())

#2. Проверьте Х на пропуски и закодируйте категориальные данные, если это #
#необходимо. 

print(X.isnull().sum())

#3. Нормализуйте значения в матрице Х функцией MinMaxScaler().
scaler = MinMaxScaler()
X_normalized = scaler.fit_transform(X)
X_normalized_df = pd.DataFrame(X_normalized, columns=X.columns)
print(X_normalized_df.head())

#4. C помощью метода локтя определите оптимальное количество кластеров и 
#разделите данные на кластеры методом K-means.

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X_normalized)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10,6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Метод локтя')
plt.xlabel('Количество кластеров')
plt.ylabel('Сумма квадратов расстояний до ближайшего центроида (WCSS)')
plt.show()

optimal_clusters = 3

kmeans = KMeans(n_clusters=optimal_clusters, init='k-means++', random_state=42)
kmeans.fit(X_normalized)
clusters = kmeans.predict(X_normalized)
centroids = kmeans.cluster_centers_

#5. Визуализируйте результаты кластеризации, выбрав для визуализации два 
#параметра из матрицы Х.

plt.figure(figsize=(10,6))
for i in range(optimal_clusters):
    plt.scatter(X_normalized[clusters == i][:, 0], X_normalized[clusters == i][:, 1], label=f'Кластер {i+1}')
plt.scatter(centroids[:, 0], centroids[:, 1], marker='X', s=200, color='black', label='Центроиды')

plt.title('Результаты кластеризации')
plt.xlabel('Нормализованный child_mort')
plt.ylabel('Нормализованный income')
plt.legend()
plt.show()

#6. Разделите данные на кластеры методом иерархической кластеризации, 
#выберите с помощью дендрограммы оптимальное количество кластеров.

Z = linkage(X_normalized, method='ward')
plt.figure(figsize=(12, 6))
dendrogram(Z)
plt.title('Дендрограмма')
plt.xlabel('Индексы объектов')
plt.ylabel('Евклидово расстояние')
plt.show()

#7. Визуализируйте результаты кластеризации методом иерархической 
#кластеризации.

optimal_clusters_hierarchical = 3
clusters_hierarchical = fcluster(Z, optimal_clusters_hierarchical, criterion='maxclust')
plt.figure(figsize=(10,6))
for i in range(optimal_clusters_hierarchical):
    plt.scatter(X_normalized[clusters_hierarchical == i+1][:, 0], X_normalized[clusters_hierarchical == i+1][:, 1], label=f'Кластер {i+1}')

plt.title('Результаты кластеризации методом иерархической кластеризации')
plt.xlabel('Нормализованный child_mort')
plt.ylabel('Нормализованный income')
plt.legend()
plt.show()

#8. Оцените качество кластеризации методами K-means и иерархической 
#кластеризации, рассчитав пару метрик качества кластеризации (модуль 
#sklearn.metrics). Например, силуэт для выборки silhouette_score() и др.
silhouette_kmeans = silhouette_score(X_normalized, clusters)
print(f'Силуэт для метода K-means: {silhouette_kmeans}')
silhouette_hierarchical = silhouette_score(X_normalized, clusters_hierarchical)
print(f'Силуэт для метода иерархической кластеризации: {silhouette_hierarchical}')

#9 Из датасета выберите любой  конкретный объект (если вы делаете  модель на датасете Country-data.csv, 
#то выберите любую страну) и  визуализируйте этот объект в виде точки отличного цвета и размера на графике кластеров (пример на 
#рисунке, точка пурпурного цвета)

chosen_country = 'Afghanistan'
chosen_index = df[df['country'] == chosen_country].index[0]
plt.figure(figsize=(10,6))
for i in range(optimal_clusters):
    plt.scatter(X_normalized[clusters == i][:, 0], X_normalized[clusters == i][:, 1], label=f'Cluster {i+1}')
plt.scatter(X_normalized[chosen_index, 0], X_normalized[chosen_index, 1], marker='o', s=200, color='purple', label=f'{chosen_country}')

plt.title('Результаты кластеризации с выбранным объектом')
plt.xlabel('Нормализованный child_mort')
plt.ylabel('Нормализованный income')
plt.legend()
plt.show()