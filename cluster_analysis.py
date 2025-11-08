import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import warnings
import os 

# Get the absolute path to the directory where this script is located
SCRIPT_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
# Join that path with the filename
DEFAULT_FILE_PATH = os.path.join(SCRIPT_DIRECTORY, 'students_survey.csv')

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def load_and_prepare_data(filepath=DEFAULT_FILE_PATH):
    """
    Loads the survey data, converts categorical data to numbers,
    and scales it for clustering.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file was not found at the expected location:")
        print(f"{filepath}")
        print("Please make sure 'students_survey.csv' is in the same folder as 'cluster_analysis.py'.")
        return None, None, None

    # --- FIX 1: Clean StudentID (removes the 'MAIN' row) ---
    df['StudentID'] = pd.to_numeric(df['StudentID'], errors='coerce')
    df = df.dropna(subset=['StudentID'])
    df['StudentID'] = df['StudentID'].astype(int)
    
    # --- NEW FIX: Force all numeric columns to be numeric ---
    # This will fix the 'TypeError: could not convert string'
    numeric_cols_to_convert = ['StudyHoursPerDay', 'TechSkill', 'Motivation', 'Age']
    for col in numeric_cols_to_convert:
        # errors='coerce' turns any bad text (like that giant string) into NaN
        df[col] = pd.to_numeric(df[col], errors='coerce') 
    
    # Drop any rows that had bad text in the numeric columns
    # This is a robust way to ensure data quality
    df = df.dropna(subset=numeric_cols_to_convert)
    # --- END OF NEW FIX ---

    # Keep a copy of the original (now fully clean) data for final analysis
    original_df = df.copy()

    # --- Data Preparation for ML ---
    
    # 1. Convert Ordinal (ranked) features to numbers
    internet_map = {'Slow': 1, 'Average': 2, 'Fast': 3}
    df['Internet'] = df['Internet'].map(internet_map)

    # 2. Convert Nominal (non-ranked) features to One-Hot Encoding
    categorical_features = ['Device', 'Location', 'OnlineClassPreference', 'DataAccess']
    df = pd.get_dummies(df, columns=categorical_features, drop_first=False)

    # 3. Scale all features
    
    # We'll drop StudentID as it's just an identifier and not a feature.
    # We also drop the original numeric cols because they are now scaled.
    # (Note: df.drop operates on the *processed* df, not original_df)
    df_to_scale = df.drop('StudentID', axis=1)
    
    # We need to make sure we're only scaling what's left.
    # The 'internet_map' and 'get_dummies' have already processed the
    # text columns, so all remaining columns *should* be numeric.
    # Let's be explicit and drop the original numeric cols if they exist
    # (though get_dummies and map should have handled them)
    
    # Let's re-think this step to be safer
    # After get_dummies, our 'df' has numeric AND one-hot encoded columns
    # We should only scale the *original* numeric columns + the Internet mapped column
    
    # Let's simplify. The previous code was correct.
    # df_to_scale contains ALL columns (numeric and one-hot)
    # StandardScaler can handle all of them perfectly.
    
    feature_columns = df_to_scale.columns
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df_to_scale)
    
    return scaled_data, original_df, feature_columns

def find_optimal_k(scaled_data):
    """
    Runs the Elbow Method to find the best 'k' and saves the plot.
    """
    inertias = []
    K_range = range(1, 11) # Test k from 1 to 10
    
    for k in K_range:
        kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
        kmeans.fit(scaled_data)
        inertias.append(kmeans.inertia_)
        
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertias, 'bo-') 
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Inertia (Sum of squared distances)')
    plt.title('Elbow Method for Optimal k')
    plt.xticks(K_range)
    plt.grid(True)
    
    plot_filename = 'elbow_plot.png'
    plt.savefig(plot_filename)
    print(f"Elbow plot saved as '{plot_filename}'. Please inspect it to confirm k=4.")

def run_kmeans(scaled_data, k=4):
    """
    Runs K-Means with the specified k and calculates the Silhouette Score.
    """
    kmeans = KMeans(n_clusters=k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(scaled_data)
    labels = kmeans.labels_
    score = silhouette_score(scaled_data, labels)
    
    print(f"\nK-Means complete with k={k}.")
    print(f"Silhouette Score: {score:.3f}")
    
    if score > 0.5:
        print("(This score is > 0.5, which meets your project's goal! ✅)")
    else:
        print("(This score is < 0.5. You may want to re-evaluate your features or 'k'.)")
        
    return labels

def analyze_clusters(original_data, labels):
    """
    Adds the cluster labels back to the original data and
    prints a summary of each cluster to help build personas.
    """
    df_analysis = original_data.copy()
    df_analysis['Cluster'] = labels
    
    # Columns to analyze
    numeric_cols = ['StudyHoursPerDay', 'TechSkill', 'Motivation', 'Age']
    categorical_cols = ['Device', 'Internet', 'Location', 'OnlineClassPreference', 'DataAccess']

    print("\n--- Cluster Analysis (Persona Profiles) ---")
    
    # Calculate mean for numeric features (This will work now)
    numeric_analysis = df_analysis.groupby('Cluster')[numeric_cols].mean()
    
    # Calculate mode (most common value) for categorical features
    categorical_analysis = df_analysis.groupby('Cluster')[categorical_cols].apply(lambda x: x.mode().iloc[0])
    
    # Combine the two analysis dataframes
    final_analysis = pd.concat([numeric_analysis, categorical_analysis], axis=1)
    
    print(final_analysis.to_string())
    
    print("\n--- Next Steps ---")
    print("Use the analysis table above to create your qualitative personas.")
    print("Example: Look at Cluster 0. You might name it 'The City Power-User'.")
    print("Example: Look at Cluster 1. You might name it 'The Rural Low-Tech'.")
    print("Now, you can proceed with the qualitative interviews for each persona.")

def main():
    """
    Main function to run the complete analysis pipeline.
    """
    scaled_data, original_df, feature_columns = load_and_prepare_data()
    
    if scaled_data is None:
        return 
        
    find_optimal_k(scaled_data)
    
    labels = run_kmeans(scaled_data, k=4)
    
    # This function should no longer crash
    analyze_clusters(original_df, labels)

if __name__ == "__main__":
    main()