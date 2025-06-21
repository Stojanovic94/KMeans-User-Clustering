import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans

# 1. Simulated guest data
data = {
    'user_id': range(1, 11),
    'average_rating': [4.5, 4.8, 1.2, 2.0, 4.7, 1.5, 2.2, 4.9, 1.3, 2.5],
    'review_count': [200, 180, 5, 10, 220, 6, 15, 210, 4, 18]
}
df = pd.DataFrame(data)

# 2. Normalize features using Min-Max scaling
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df[['average_rating', 'review_count']])

# 3. K-Means clustering (K=2)
kmeans = KMeans(n_clusters=2, random_state=42)
df['cluster'] = kmeans.fit_predict(scaled_data)

# 4. Visualize clusters
plt.figure(figsize=(8, 5))
sns.scatterplot(
    x='average_rating',
    y='review_count',
    hue='cluster',
    data=df,
    palette='Set2',
    s=100
)
plt.title('User Clustering Based on Rating Behavior')
plt.xlabel('Average Rating')
plt.ylabel('Number of Reviews')
plt.grid(True)
plt.legend(title="Cluster")
plt.tight_layout()
plt.show()

# 5. Output clustered DataFrame
print("\nClustered DataFrame:\n", df)