import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from collections import Counter

# Download required NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)


def preprocess_keyword(keyword):
    # Cache the lemmatizer and stop_words to avoid recreating them for each keyword
    if not hasattr(preprocess_keyword, 'lemmatizer'):
        preprocess_keyword.lemmatizer = WordNetLemmatizer()
        preprocess_keyword.stop_words = set(stopwords.words('english'))
    
    keyword = re.sub(r'[^a-zA-Z0-9\s]', '', str(keyword).lower())
    tokens = keyword.split()
    tokens = [preprocess_keyword.lemmatizer.lemmatize(token) 
             for token in tokens 
             if token not in preprocess_keyword.stop_words]
    return ' '.join(tokens)


def cluster_keywords(file_path, max_cluster_size=50, min_cluster_size=15):
    try:
        df = pd.read_csv(file_path)
        required_columns = ['keyword', 'clicks', 'impressions', 'conversions', 'ad_group']
        if not all(col in df.columns for col in required_columns):
            raise ValueError(f"CSV must contain columns: {', '.join(required_columns)}")

        df['clicks'] = pd.to_numeric(df['clicks'], errors='coerce').fillna(0).astype(int)
        df['impressions'] = pd.to_numeric(df['impressions'], errors='coerce').fillna(0).astype(int)
        df['conversions'] = pd.to_numeric(df['conversions'], errors='coerce').fillna(0).astype(int)

        df['processed_keyword'] = df['keyword'].apply(preprocess_keyword)

        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(df['processed_keyword'])

        # Normalize the TF-IDF matrix
        normalized_matrix = normalize(tfidf_matrix)

        # Function to perform hierarchical clustering
        def hierarchical_clustering(matrix, n_clusters):
            clusterer = AgglomerativeClustering(n_clusters=n_clusters)
            return clusterer.fit_predict(matrix.toarray())

        # Add early stopping for very small datasets
        n_samples = normalized_matrix.shape[0]
        if n_samples < min_cluster_size:
            print(f"Warning: Dataset too small ({n_samples} samples) for clustering with min_cluster_size={min_cluster_size}")
            n_clusters = 1
        else:
            n_clusters = min(max(n_samples // max_cluster_size, 1), n_samples // min_cluster_size)

        best_score = -1
        best_labels = None
        patience = 3  # Number of iterations without improvement before stopping
        no_improvement = 0

        while n_clusters > 1:
            labels = hierarchical_clustering(normalized_matrix, n_clusters)
            
            # Calculate silhouette score for cluster quality
            if n_clusters > 1:  # Silhouette score needs at least 2 clusters
                score = silhouette_score(normalized_matrix.toarray(), labels)
                if score > best_score:
                    best_score = score
                    best_labels = labels
                    no_improvement = 0
                else:
                    no_improvement += 1
                    
                if no_improvement >= patience:
                    labels = best_labels
                    break
            
            cluster_sizes = pd.Series(labels).value_counts()
            
            if cluster_sizes.max() <= max_cluster_size and cluster_sizes.min() >= min_cluster_size:
                break
                
            n_clusters -= 1

        df['cluster'] = labels

        # Calculate cluster metrics
        cluster_data = df.groupby('cluster').agg({
            'keyword': list,
            'clicks': 'sum',
            'impressions': 'sum',
            'conversions': 'sum',
            'ad_group': lambda x: x.mode().iloc[0] if not x.mode().empty else 'Multiple'
        }).reset_index()

        def get_top_keywords(cluster_keywords, top_n=5):
            words = ' '.join(cluster_keywords).split()
            return ', '.join(word for word, _ in Counter(words).most_common(top_n))

        cluster_data['top_keywords'] = cluster_data['keyword'].apply(get_top_keywords)
        cluster_data['keyword_count'] = cluster_data['keyword'].apply(len)
        cluster_data['ctr'] = (cluster_data['clicks'] / cluster_data['impressions'] * 100).round(2)
        cluster_data['conversion_rate'] = (cluster_data['conversions'] / cluster_data['clicks'] * 100).round(2)

        # Calculate performance score
        cluster_data['performance_score'] = (
                                                    cluster_data['ctr'].rank(pct=True) +
                                                    cluster_data['conversion_rate'].rank(pct=True)
                                            ) / 2

        column_order = ['cluster', 'top_keywords', 'keyword_count', 'clicks', 'impressions', 'conversions',
                        'ctr', 'conversion_rate', 'performance_score', 'ad_group', 'keyword']
        cluster_data = cluster_data[column_order]

        return cluster_data.sort_values('performance_score', ascending=False)

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None


def suggest_ad_groups(cluster_data):
    suggestions = []
    for _, row in cluster_data.iterrows():
        suggestion = {
            'cluster': row['cluster'],
            'suggested_ad_group_name': f"Cluster_{row['cluster']}_{row['top_keywords'].split(', ')[0]}",
            'top_keywords': row['top_keywords'],
            'performance_score': row['performance_score'],
            'ctr': row['ctr'],
            'conversion_rate': row['conversion_rate'],
            'keyword_count': row['keyword_count']
        }
        suggestions.append(suggestion)
    return suggestions


# Main execution
if __name__ == "__main__":
    file_path = input("Enter the path to your CSV file: ")

    cluster_data = cluster_keywords(file_path)

    if cluster_data is not None:
        print("\nCluster Summary:")
        for _, row in cluster_data.iterrows():
            print(f"\nCluster {row['cluster']}:")
            print(f"Top Keywords: {row['top_keywords']}")
            print(f"Number of Keywords: {row['keyword_count']}")
            print(f"Clicks: {row['clicks']}")
            print(f"Impressions: {row['impressions']}")
            print(f"Conversions: {row['conversions']}")
            print(f"CTR: {row['ctr']}%")
            print(f"Conversion Rate: {row['conversion_rate']}%")
            print(f"Performance Score: {row['performance_score']:.2f}")
            print(f"Main Ad Group: {row['ad_group']}")

        # Generate ad group suggestions
        ad_group_suggestions = suggest_ad_groups(cluster_data)

        print("\nSuggested Ad Groups for New Campaigns:")
        for suggestion in ad_group_suggestions:
            print(f"\nCluster {suggestion['cluster']}:")
            print(f"Suggested Ad Group Name: {suggestion['suggested_ad_group_name']}")
            print(f"Top Keywords: {suggestion['top_keywords']}")
            print(f"Performance Score: {suggestion['performance_score']:.2f}")
            print(f"CTR: {suggestion['ctr']}%")
            print(f"Conversion Rate: {suggestion['conversion_rate']}%")
            print(f"Number of Keywords: {suggestion['keyword_count']}")

        # Save results to CSV
        cluster_data.to_csv('keyword_cluster_results.csv', index=False)
        print("\nDetailed results saved to 'keyword_cluster_results.csv'")
    else:
        print("Failed to cluster keywords. Please check your input file and try again.")