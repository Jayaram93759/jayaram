# Step 1: Import libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Step 2: Upload CSV file
from google.colab import files
uploaded = files.upload()

# Step 3: Load dataset
# The uploaded file will keep the same filename
df = pd.read_csv('Mall_Customers.csv')

# Step 4: Quick exploration
print("First 5 rows:")
print(df.head())

print("\nData summary:")
print(df.describe())

print("\nMissing values:")
print(df.isnull().sum())

# Step 5: Visualizations

plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
sns.histplot(df['Age'], kde=True, bins=20, color='skyblue')
plt.title('Age Distribution')

plt.subplot(1, 3, 2)
sns.histplot(df['Annual Income (k$)'], kde=True, bins=20, color='salmon')
plt.title('Annual Income Distribution')

plt.subplot(1, 3, 3)
sns.histplot(df['Spending Score (1-100)'], kde=True, bins=20, color='limegreen')
plt.title('Spending Score Distribution')

plt.tight_layout()
plt.show()

# Scatter plot: Income vs Spending Score
plt.figure(figsize=(6,5))
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Gender')
plt.title('Income vs Spending Score by Gender')
plt.show()

# Step 6: KMeans Clustering

# Use Annual Income & Spending Score only
X = df[['Annual Income (k$)', 'Spending Score (1-100)']]

# Find optimal number of clusters (Elbow method)
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(6,4))
plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Choose k=5 based on elbow plot
kmeans = KMeans(n_clusters=5, init='k-means++', random_state=42)
df['Cluster'] = kmeans.fit_predict(X)

# Step 7: Visualize clusters
plt.figure(figsize=(8,6))
sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'],
                hue=df['Cluster'], palette='tab10', s=100)
plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1],
            s=300, c='yellow', label='Centroids', marker='*')
plt.title('Customer Segments')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show()

# Step 8: Analyze clusters
print("\nAverage values by cluster:")
print(df.groupby('Cluster')[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']].mean())

# Step 9: Suggest target customers
print("\nSuggestion:")
print("Clusters with high spending scores and moderate to high income can be chosen as target customers.")
