import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

def load_and_explore_data(file_paths):
    """Load datasets and explore their structure"""
    datasets = {}
    
    for i, path in enumerate(file_paths, 1):
        try:
            df = pd.read_csv(path)
            datasets[f'dataset_{i}'] = df
            print(f"Dataset {i} loaded successfully:")
            print(f"  Shape: {df.shape}")
            print(f"  Columns: {list(df.columns)}")
            print(f"  Missing values: {df.isnull().sum().sum()}")
            print("-" * 50)
        except Exception as e:
            print(f"Error loading dataset {i}: {e}")
    
    return datasets

def merge_datasets(datasets):
    """Merge datasets based on common columns"""
    dataset_list = list(datasets.values())
    
    if len(dataset_list) < 2:
        return dataset_list[0] if dataset_list else pd.DataFrame()
    
    # Start with the first dataset
    merged_df = dataset_list[0].copy()
    
    # Merge with subsequent datasets
    for i, df in enumerate(dataset_list[1:], 2):
        # Find common columns for merging
        common_cols = list(set(merged_df.columns) & set(df.columns))
        
        if common_cols:
            print(f"Merging with dataset {i} on columns: {common_cols}")
            # Use outer join to preserve all data
            merged_df = pd.merge(merged_df, df, on=common_cols, how='outer', suffixes=('', f'_ds{i}'))
        else:
            # If no common columns, concatenate horizontally
            print(f"No common columns found with dataset {i}. Concatenating horizontally.")
            merged_df = pd.concat([merged_df, df], axis=1, sort=False)
    
    print(f"Final merged dataset shape: {merged_df.shape}")
    return merged_df

def clean_data(df):
    """Clean the merged dataset"""
    print("Starting data cleaning...")
    
    # Remove duplicate rows
    initial_rows = len(df)
    df = df.drop_duplicates()
    print(f"Removed {initial_rows - len(df)} duplicate rows")
    
    # Remove columns with all missing values
    df = df.dropna(axis=1, how='all')
    
    # Remove rows with all missing values
    df = df.dropna(axis=0, how='all')
    
    # Handle duplicate columns (from merging)
    duplicate_cols = []
    for col in df.columns:
        if col.endswith('_ds2') or col.endswith('_ds3'):
            base_col = col.split('_ds')[0]
            if base_col in df.columns:
                # Fill missing values in base column with values from duplicate column
                df[base_col] = df[base_col].fillna(df[col])
                duplicate_cols.append(col)
    
    # Drop duplicate columns
    df = df.drop(columns=duplicate_cols)
    
    print(f"Dataset shape after cleaning: {df.shape}")
    return df

def encode_categorical_features(df):
    """Encode categorical features using Label Encoding"""
    print("Encoding categorical features...")
    
    label_encoders = {}
    encoded_df = df.copy()
    
    # Identify categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    for col in categorical_cols:
        if col in encoded_df.columns:
            # Handle missing values in categorical columns
            encoded_df[col] = encoded_df[col].fillna('Unknown')
            
            # Apply label encoding
            le = LabelEncoder()
            encoded_df[col] = le.fit_transform(encoded_df[col].astype(str))
            label_encoders[col] = le
            
            print(f"  Encoded column: {col} ({len(le.classes_)} unique values)")
    
    return encoded_df, label_encoders

def apply_knn_imputation(df, n_neighbors=5):
    """Apply KNN Imputation to handle missing values"""
    print(f"Applying KNN Imputation with {n_neighbors} neighbors...")
    
    # Separate numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    if len(numeric_cols) == 0:
        print("No numeric columns found for KNN imputation")
        return df
    
    # Apply KNN imputation only to numeric columns
    numeric_data = df[numeric_cols]
    
    # Check if there are any missing values
    missing_count = numeric_data.isnull().sum().sum()
    print(f"Missing values in numeric columns: {missing_count}")
    
    if missing_count > 0:
        # Apply KNN imputation
        imputer = KNNImputer(n_neighbors=n_neighbors)
        imputed_data = imputer.fit_transform(numeric_data)
        
        # Create DataFrame with imputed data
        imputed_df = pd.DataFrame(imputed_data, columns=numeric_cols, index=df.index)
        
        # Combine with non-numeric columns
        non_numeric_cols = df.select_dtypes(exclude=[np.number]).columns
        if len(non_numeric_cols) > 0:
            final_df = pd.concat([imputed_df, df[non_numeric_cols]], axis=1)
        else:
            final_df = imputed_df
        
        print("KNN Imputation completed successfully")
    else:
        final_df = df
        print("No missing values found in numeric columns")
    
    return final_df

def preprocess_data(file_paths, output_path, n_neighbors=5):
    """Main preprocessing function"""
    print("=" * 60)
    print("STARTING DATA PREPROCESSING PIPELINE")
    print("=" * 60)
    
    # Step 1: Load datasets
    print("\n1. LOADING DATASETS")
    datasets = load_and_explore_data(file_paths)
    
    if not datasets:
        print("No datasets loaded successfully. Exiting...")
        return
    
    # Step 2: Merge datasets
    print("\n2. MERGING DATASETS")
    merged_df = merge_datasets(datasets)
    
    # Step 3: Clean data
    print("\n3. CLEANING DATA")
    cleaned_df = clean_data(merged_df)
    
    # Step 4: Encode categorical features
    print("\n4. ENCODING CATEGORICAL FEATURES")
    encoded_df, label_encoders = encode_categorical_features(cleaned_df)
    
    # Step 5: Apply KNN Imputation
    print("\n5. APPLYING KNN IMPUTATION")
    final_df = apply_knn_imputation(encoded_df, n_neighbors)
    
    # Step 6: Final data summary
    print("\n6. FINAL DATA SUMMARY")
    print(f"Final dataset shape: {final_df.shape}")
    print(f"Missing values: {final_df.isnull().sum().sum()}")
    print(f"Data types:\n{final_df.dtypes.value_counts()}")
    
    # Step 7: Save processed data
    print("\n7. SAVING PROCESSED DATA")
    try:
        final_df.to_csv(output_path, index=False)
        print(f"Processed data saved successfully to: {output_path}")
    except Exception as e:
        print(f"Error saving file: {e}")
    
    print("\n" + "=" * 60)
    print("DATA PREPROCESSING COMPLETED")
    print("=" * 60)
    
    return final_df, label_encoders

# File paths
file_path1 = r'C:\Users\HP\Documents\Mini Project 5th sem\Final Model\Final Dataset\cleaned_fertilizer_data.csv'
file_path2 = r'C:\Users\HP\Documents\Mini Project 5th sem\Final Model\Final Dataset\cleanedFertility.csv'
file_path3 = r'C:\Users\HP\Documents\Mini Project 5th sem\Final Model\Final Dataset\cleaned_crop_data.csv'

file_paths = [file_path1, file_path2, file_path3]
output_path = r'C:\Users\HP\Documents\Mini Project 5th sem\Final Model\Final Dataset\Merge_data.csv'

# Execute preprocessing pipeline
if __name__ == "__main__":
    processed_data, encoders = preprocess_data(file_paths, output_path, n_neighbors=5)
    
    # Optional: Display first few rows of processed data
    if processed_data is not None:
        print("\nFirst 5 rows of processed data:")
        print(processed_data.head())
        
        print("\nColumn names in final dataset:")
        print(list(processed_data.columns))
        
        # Save encoding information
        if encoders:
            print(f"\nLabel encoders created for {len(encoders)} categorical columns:")
            for col, encoder in encoders.items():
                print(f"  {col}: {len(encoder.classes_)} categories")