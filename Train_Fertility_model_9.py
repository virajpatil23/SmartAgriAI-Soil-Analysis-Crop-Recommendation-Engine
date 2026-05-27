import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model, callbacks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.metrics import classification_report, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
import pickle
import os
warnings.filterwarnings('ignore')

class AgriculturePredictionModel:
    def __init__(self, data_path):
        """
        Initialize the comprehensive agriculture prediction model
        """
        self.data_path = data_path
        self.data = None
        self.scaler = StandardScaler()
        self.fertility_scaler = MinMaxScaler()
        self.label_encoders = {}
        self.model = None
        self.feature_columns = None  # Store feature column names
        
        # Crop sequencing knowledge base
        self.crop_sequences = {
            'high_fertility': ['Rice', 'Wheat', 'Maize', 'Cotton', 'Sugarcane'],
            'medium_fertility': ['Barley', 'Millet', 'Sorghum', 'Groundnut', 'Soybean'],
            'low_fertility': ['Pulses', 'Legumes', 'Fodder', 'Green Manure', 'Cover Crops']
        }
        
        # Fertilizer recommendation database
        self.fertilizer_recommendations = {
            'high_fertility': {
                'primary': ['NPK 20-20-20', 'Urea', 'DAP'],
                'secondary': ['Potash', 'Zinc Sulfate'],
                'organic': ['Compost', 'Farmyard Manure'],
                'timeline': {'pre_sowing': 15, 'post_sowing': 30, 'flowering': 60}
            },
            'medium_fertility': {
                'primary': ['NPK 19-19-19', 'Single Super Phosphate', 'Muriate of Potash'],
                'secondary': ['Boron', 'Magnesium Sulfate'],
                'organic': ['Vermicompost', 'Biofertilizer'],
                'timeline': {'pre_sowing': 20, 'post_sowing': 35, 'flowering': 65}
            },
            'low_fertility': {
                'primary': ['NPK 15-15-15', 'Rock Phosphate', 'Gypsum'],
                'secondary': ['Lime', 'Sulfur'],
                'organic': ['Green Manure', 'Organic Compost', 'Mycorrhizal Inoculant'],
                'timeline': {'pre_sowing': 25, 'post_sowing': 40, 'flowering': 70}
            }
        }
    
    def load_and_prepare_data(self):
        """Load and prepare the merged dataset"""
        print("Loading and preparing data...")
        
        try:
            self.data = pd.read_csv(self.data_path)
            print(f"Data loaded successfully. Shape: {self.data.shape}")
            
            # Display basic information
            print(f"Columns: {list(self.data.columns)}")
            print(f"Missing values: {self.data.isnull().sum().sum()}")
            
            # Handle any infinite values
            self.data = self.data.replace([np.inf, -np.inf], np.nan)
            
            # Fill any remaining NaN values with median for numeric columns
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if self.data[col].isnull().sum() > 0:
                    self.data[col].fillna(self.data[col].median(), inplace=True)
            
            return self.data
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return None
    
    def create_target_variables(self):
        """Create target variables for multi-task learning"""
        print("Creating target variables...")
        
        # Filter out target-related columns to get feature columns
        target_keywords = ['fertility', 'fertile', 'crop type', 'fertilizer name', 'output', 'yield']
        feature_cols = []
        
        for col in self.data.columns:
            col_lower = col.lower()
            is_target = any(keyword in col_lower for keyword in target_keywords)
            if not is_target:
                feature_cols.append(col)
        
        print(f"Feature columns identified: {len(feature_cols)} columns")
        print(f"Feature columns: {feature_cols}")
        
        # Check if we already have a fertility column
        fertility_col = None
        for col in self.data.columns:
            if 'fertility' in col.lower() and col.lower() not in ['fertility_class', 'fertility_rate']:
                fertility_col = col
                break
        
        # Create fertility classification
        if 'fertility_class' not in self.data.columns:
            if fertility_col is not None:
                # Use existing fertility column
                print(f"Using existing fertility column: {fertility_col}")
                
                # Check unique values in fertility column
                unique_values = self.data[fertility_col].unique()
                print(f"Unique values in {fertility_col}: {unique_values}")
                
                # Convert to numeric if it's categorical
                if self.data[fertility_col].dtype == 'object' or self.data[fertility_col].dtype.name == 'category':
                    # Try to map text values to numeric
                    fertility_text_map = {
                        'low': 0, 'medium': 1, 'high': 2,
                        'poor': 0, 'moderate': 1, 'good': 2, 'excellent': 2,
                        'infertile': 0, 'fertile': 2, 'semi-fertile': 1,
                        '0': 0, '1': 1, '2': 2
                    }
                    
                    fertility_numeric = self.data[fertility_col].astype(str).str.lower().map(fertility_text_map)
                    
                    if fertility_numeric.isna().sum() > len(self.data) * 0.5:  # If more than 50% are NaN
                        # If mapping failed, use label encoding
                        le_temp = LabelEncoder()
                        fertility_numeric = le_temp.fit_transform(self.data[fertility_col].astype(str))
                        print("Used label encoding for fertility column")
                    else:
                        print("Used text mapping for fertility column")
                else:
                    fertility_numeric = pd.to_numeric(self.data[fertility_col], errors='coerce')
                    if fertility_numeric.isna().sum() > 0:
                        fertility_numeric.fillna(fertility_numeric.median(), inplace=True)
                
                # Convert to fertility classes using quantiles for better distribution
                try:
                    self.data['fertility_class'] = pd.qcut(fertility_numeric, 
                                                         q=3, 
                                                         labels=['low_fertility', 'medium_fertility', 'high_fertility'],
                                                         duplicates='drop')
                except ValueError:
                    # If qcut fails due to duplicate bin edges, use cut
                    self.data['fertility_class'] = pd.cut(fertility_numeric, 
                                                        bins=3, 
                                                        labels=['low_fertility', 'medium_fertility', 'high_fertility'])
            else:
                # Create based on key soil parameters
                soil_params = []
                # Look for common soil parameters
                param_mapping = {
                    'n': ['n', 'nitrogen'], 'p': ['p', 'phosphorus'], 
                    'k': ['k', 'potassium'], 'ph': ['ph']
                }
                
                for param_key, param_variations in param_mapping.items():
                    for variation in param_variations:
                        matching_cols = [col for col in self.data.columns if variation == col.lower()]
                        if matching_cols:
                            soil_params.append(matching_cols[0])
                            break
                
                if len(soil_params) >= 3:
                    print(f"Creating fertility classification using soil parameters: {soil_params[:4]}")
                    # Normalize each parameter before averaging
                    normalized_params = self.data[soil_params[:4]].copy()
                    for col in normalized_params.columns:
                        normalized_params[col] = (normalized_params[col] - normalized_params[col].min()) / (normalized_params[col].max() - normalized_params[col].min())
                    
                    fertility_score = normalized_params.mean(axis=1)
                    self.data['fertility_class'] = pd.qcut(fertility_score, 
                                                         q=3, 
                                                         labels=['low_fertility', 'medium_fertility', 'high_fertility'],
                                                         duplicates='drop')
                else:
                    # Create based on overall data distribution
                    print("Creating fertility classes based on overall data distribution")
                    # Use the first few numeric columns as a proxy
                    numeric_cols = self.data[feature_cols].select_dtypes(include=[np.number]).columns[:5]
                    if len(numeric_cols) > 0:
                        normalized_data = self.data[numeric_cols].copy()
                        for col in normalized_data.columns:
                            normalized_data[col] = (normalized_data[col] - normalized_data[col].min()) / (normalized_data[col].max() - normalized_data[col].min())
                        
                        fertility_score = normalized_data.mean(axis=1)
                        self.data['fertility_class'] = pd.qcut(fertility_score, 
                                                             q=3, 
                                                             labels=['low_fertility', 'medium_fertility', 'high_fertility'],
                                                             duplicates='drop')
                    else:
                        # Last resort: random assignment with seed for reproducibility
                        print("Creating random fertility classes for demonstration")
                        np.random.seed(42)
                        self.data['fertility_class'] = np.random.choice(['low_fertility', 'medium_fertility', 'high_fertility'], 
                                                                      size=len(self.data))
        
        # Ensure fertility_class is string type, not categorical
        if self.data['fertility_class'].dtype.name == 'category':
            self.data['fertility_class'] = self.data['fertility_class'].astype(str)
        
        # Handle any NaN values in fertility_class
        if self.data['fertility_class'].isnull().sum() > 0:
            self.data['fertility_class'].fillna('medium_fertility', inplace=True)
        
        # Create fertility rate (if not exists)
        if 'fertility_rate' not in self.data.columns:
            fertility_map = {'low_fertility': 0.3, 'medium_fertility': 0.6, 'high_fertility': 0.9}
            self.data['fertility_rate'] = self.data['fertility_class'].map(fertility_map).astype(float)
            
            # Add some realistic noise
            np.random.seed(42)
            noise = np.random.normal(0, 0.05, len(self.data))  # Reduced noise
            self.data['fertility_rate'] = self.data['fertility_rate'] + noise
            self.data['fertility_rate'] = np.clip(self.data['fertility_rate'], 0.1, 1.0)
        
        # Create is_fertile binary target
        self.data['is_fertile'] = (self.data['fertility_rate'] > 0.5).astype(int)
        
        print("Target variables created successfully")
        print(f"Fertility class distribution:\n{self.data['fertility_class'].value_counts()}")
        print(f"Fertility rate range: {self.data['fertility_rate'].min():.3f} - {self.data['fertility_rate'].max():.3f}")
        print(f"Is fertile distribution:\n{self.data['is_fertile'].value_counts()}")
        
        return feature_cols
    
    def preprocess_features(self, feature_cols):
        """Preprocess features for model training"""
        print("Preprocessing features...")
        
        # Store feature columns for later use
        self.feature_columns = feature_cols.copy()
        
        # Handle categorical variables
        categorical_cols = self.data[feature_cols].select_dtypes(include=['object', 'category']).columns
        print(f"Categorical columns found: {list(categorical_cols)}")
        
        for col in categorical_cols:
            if col not in self.label_encoders:
                self.label_encoders[col] = LabelEncoder()
                # Handle missing values in categorical columns
                self.data[col] = self.data[col].fillna('Unknown')
                self.data[col] = self.label_encoders[col].fit_transform(self.data[col].astype(str))
        
        # Ensure all feature columns are numeric
        for col in feature_cols:
            if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
                if self.data[col].isnull().sum() > 0:
                    self.data[col].fillna(self.data[col].median(), inplace=True)
        
        # Prepare feature matrix
        X = self.data[feature_cols].values.astype(np.float32)
        
        # Prepare targets
        y_fertile = self.data['is_fertile'].values.astype(np.float32)
        y_fertility_rate = self.data['fertility_rate'].values.astype(np.float32)
        
        # Encode fertility class
        if 'fertility_class_encoder' not in self.label_encoders:
            self.label_encoders['fertility_class_encoder'] = LabelEncoder()
        y_fertility_class = self.label_encoders['fertility_class_encoder'].fit_transform(self.data['fertility_class'])
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"Features preprocessed. Shape: {X_scaled.shape}")
        print(f"Feature columns stored: {len(self.feature_columns)}")
        print(f"Target distributions:")
        print(f"  - Fertile binary: {np.bincount(y_fertile.astype(int))}")
        print(f"  - Fertility classes: {np.bincount(y_fertility_class)}")
        
        return X_scaled, y_fertile, y_fertility_rate, y_fertility_class
    
    def build_model(self, input_dim, num_classes=3):
        """Build multi-task deep learning model"""
        print("Building multi-task neural network...")
        
        # Input layer
        input_layer = layers.Input(shape=(input_dim,), name='soil_features')
        
        # Shared layers with batch normalization and dropout
        shared = layers.Dense(256, activation='relu', name='shared_1')(input_layer)
        shared = layers.BatchNormalization()(shared)
        shared = layers.Dropout(0.3)(shared)
        
        shared = layers.Dense(128, activation='relu', name='shared_2')(shared)
        shared = layers.BatchNormalization()(shared)
        shared = layers.Dropout(0.2)(shared)
        
        shared = layers.Dense(64, activation='relu', name='shared_3')(shared)
        shared = layers.BatchNormalization()(shared)
        shared = layers.Dropout(0.2)(shared)
        
        # Task-specific branches
        
        # 1. Fertility Binary Classification
        fertility_branch = layers.Dense(32, activation='relu', name='fertility_branch_1')(shared)
        fertility_branch = layers.Dropout(0.1)(fertility_branch)
        fertility_output = layers.Dense(1, activation='sigmoid', name='is_fertile')(fertility_branch)
        
        # 2. Fertility Rate Regression
        fertility_rate_branch = layers.Dense(32, activation='relu', name='fertility_rate_branch_1')(shared)
        fertility_rate_branch = layers.Dropout(0.1)(fertility_rate_branch)
        fertility_rate_output = layers.Dense(1, activation='sigmoid', name='fertility_rate')(fertility_rate_branch)
        
        # 3. Fertility Class Classification
        fertility_class_branch = layers.Dense(32, activation='relu', name='fertility_class_branch_1')(shared)
        fertility_class_branch = layers.Dropout(0.1)(fertility_class_branch)
        fertility_class_output = layers.Dense(num_classes, activation='softmax', name='fertility_class')(fertility_class_branch)
        
        # Create model
        model = Model(
            inputs=input_layer,
            outputs=[fertility_output, fertility_rate_output, fertility_class_output],
            name='Agriculture_Prediction_Model'
        )
        
        # Compile model with appropriate optimizers and loss functions
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss={
                'is_fertile': 'binary_crossentropy',
                'fertility_rate': 'huber',  # More robust for regression
                'fertility_class': 'sparse_categorical_crossentropy'
            },
            loss_weights={
                'is_fertile': 1.0,
                'fertility_rate': 0.8,
                'fertility_class': 1.2
            },
            metrics={
                'is_fertile': ['accuracy', 'precision', 'recall'],
                'fertility_rate': ['mae', 'mse'],
                'fertility_class': ['accuracy']
            }
        )
        
        print("Model built successfully")
        print(model.summary())
        return model
    
    def train_model(self, X, y_fertile, y_fertility_rate, y_fertility_class, 
                   test_size=0.2, validation_size=0.15, epochs=50, batch_size=64):
        """Train the multi-task model"""
        print("Training the model...")
        
        # Split data
        X_train, X_temp, y_fertile_train, y_fertile_temp, y_rate_train, y_rate_temp, y_class_train, y_class_temp = train_test_split(
            X, y_fertile, y_fertility_rate, y_fertility_class, 
            test_size=test_size + validation_size, random_state=42, stratify=y_fertile
        )
        
        X_val, X_test, y_fertile_val, y_fertile_test, y_rate_val, y_rate_test, y_class_val, y_class_test = train_test_split(
            X_temp, y_fertile_temp, y_rate_temp, y_class_temp,
            test_size=test_size/(test_size + validation_size), random_state=42, stratify=y_fertile_temp
        )
        
        # Prepare training data
        train_targets = {
            'is_fertile': y_fertile_train,
            'fertility_rate': y_rate_train,
            'fertility_class': y_class_train
        }
        
        val_targets = {
            'is_fertile': y_fertile_val,
            'fertility_rate': y_rate_val,
            'fertility_class': y_class_val
        }
        
        # Callbacks
        early_stopping = callbacks.EarlyStopping(
            monitor='val_loss', patience=10, restore_best_weights=True, verbose=1
        )
        
        reduce_lr = callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1
        )
        
        # Train model
        history = self.model.fit(
            X_train, train_targets,
            validation_data=(X_val, val_targets),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )
        
        # Evaluate on test set
        test_targets = {
            'is_fertile': y_fertile_test,
            'fertility_rate': y_rate_test,
            'fertility_class': y_class_test
        }
        
        test_loss = self.model.evaluate(X_test, test_targets, verbose=0)
        print(f"Test Loss: {test_loss}")
        
        # Make predictions
        predictions = self.model.predict(X_test, verbose=0)
        
        # Evaluate each task
        self.evaluate_predictions(predictions, y_fertile_test, y_rate_test, y_class_test)
        
        return history, (X_test, test_targets)
    
    def evaluate_predictions(self, predictions, y_fertile_test, y_rate_test, y_class_test):
        """Evaluate model predictions"""
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        
        pred_fertile, pred_rate, pred_class = predictions
        
        # Binary classification evaluation
        pred_fertile_binary = (pred_fertile > 0.5).astype(int).flatten()
        print("\n1. FERTILITY BINARY CLASSIFICATION:")
        print(classification_report(y_fertile_test, pred_fertile_binary))
        
        # Regression evaluation
        print("\n2. FERTILITY RATE REGRESSION:")
        mse = mean_squared_error(y_rate_test, pred_rate.flatten())
        mae = mean_absolute_error(y_rate_test, pred_rate.flatten())
        print(f"MSE: {mse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"RMSE: {np.sqrt(mse):.4f}")
        
        # Multi-class classification evaluation
        pred_class_labels = np.argmax(pred_class, axis=1)
        print("\n3. FERTILITY CLASS CLASSIFICATION:")
        print(classification_report(y_class_test, pred_class_labels))
    
    def predict_comprehensive(self, soil_data):
        """Make comprehensive predictions for new soil data"""
        
        if self.feature_columns is None:
            raise ValueError("Model not trained yet. Please train the model first.")
        
        if isinstance(soil_data, dict):
            # Convert single prediction to array format using stored feature columns
            soil_array = np.array([[soil_data.get(col, 0) for col in self.feature_columns]])
        else:
            # Ensure the input array has the correct number of features
            if len(soil_data.shape) == 1:
                soil_array = soil_data.reshape(1, -1)
            else:
                soil_array = soil_data
            
            # Check if the number of features matches
            if soil_array.shape[1] != len(self.feature_columns):
                print(f"Warning: Input has {soil_array.shape[1]} features, but model expects {len(self.feature_columns)}")
                # Truncate or pad as needed
                if soil_array.shape[1] > len(self.feature_columns):
                    soil_array = soil_array[:, :len(self.feature_columns)]
                else:
                    # Pad with zeros
                    padding = np.zeros((soil_array.shape[0], len(self.feature_columns) - soil_array.shape[1]))
                    soil_array = np.hstack([soil_array, padding])
        
        # Convert to float32 and handle any inf/nan values
        soil_array = soil_array.astype(np.float32)
        soil_array = np.nan_to_num(soil_array, nan=0.0, posinf=1e6, neginf=-1e6)
        
        # Scale features
        soil_scaled = self.scaler.transform(soil_array)
        
        # Make predictions
        predictions = self.model.predict(soil_scaled, verbose=0)
        pred_fertile, pred_rate, pred_class = predictions
        
        # Process predictions
        is_fertile = pred_fertile[0][0] > 0.5
        fertility_rate = float(np.clip(pred_rate[0][0], 0, 1))
        fertility_class_idx = np.argmax(pred_class[0])
        fertility_class = self.label_encoders['fertility_class_encoder'].inverse_transform([fertility_class_idx])[0]
        
        return {
            'is_fertile': bool(is_fertile),
            'fertility_rate': fertility_rate,
            'fertility_class': fertility_class,
            'confidence_scores': {
                'fertile_probability': float(pred_fertile[0][0]),
                'class_probabilities': pred_class[0].tolist()
            }
        }
    
    def recommend_crop_sequence(self, fertility_class, num_seasons=4):
        """Recommend crop sequence based on fertility class"""
        crops = self.crop_sequences.get(fertility_class, self.crop_sequences['medium_fertility'])
        
        # Create seasonal sequence
        sequence = []
        for season in range(num_seasons):
            season_names = ['Spring', 'Summer', 'Monsoon', 'Winter']
            crop_index = season % len(crops)
            sequence.append({
                'season': season_names[season % 4],
                'crop': crops[crop_index],
                'priority': 'High' if season < 2 else 'Medium'
            })
        
        return sequence
    
    def recommend_fertilizer(self, fertility_class, crop_type=None):
        """Recommend fertilizer based on fertility class and crop type"""
        fertilizer_info = self.fertilizer_recommendations.get(fertility_class, 
                                                            self.fertilizer_recommendations['medium_fertility'])
        
        # Calculate application dates
        today = datetime.now()
        application_schedule = {}
        
        for stage, days_offset in fertilizer_info['timeline'].items():
            application_date = today + timedelta(days=days_offset)
            application_schedule[stage] = application_date.strftime('%Y-%m-%d')
        
        recommendation = {
            'fertility_level': fertility_class,
            'primary_fertilizers': fertilizer_info['primary'],
            'secondary_fertilizers': fertilizer_info['secondary'],
            'organic_options': fertilizer_info['organic'],
            'application_schedule': application_schedule,
            'dosage_recommendation': {
                'primary': '2-3 kg per acre' if fertility_class == 'low_fertility' else '1.5-2 kg per acre',
                'secondary': '0.5-1 kg per acre',
                'organic': '500-1000 kg per acre'
            }
        }
        
        return recommendation
    
    def generate_complete_recommendation(self, soil_data):
        """Generate complete agricultural recommendation"""
        print("\n" + "="*60)
        print("GENERATING COMPREHENSIVE AGRICULTURAL RECOMMENDATION")
        print("="*60)
        
        try:
            # Make predictions
            prediction = self.predict_comprehensive(soil_data)
            
            # Get crop sequence
            crop_sequence = self.recommend_crop_sequence(prediction['fertility_class'])
            
            # Get fertilizer recommendation
            fertilizer_rec = self.recommend_fertilizer(prediction['fertility_class'])
            
            # Compile complete recommendation
            complete_recommendation = {
                'soil_analysis': prediction,
                'crop_sequence': crop_sequence,
                'fertilizer_recommendation': fertilizer_rec,
                'additional_recommendations': {
                    'soil_improvement': self.get_soil_improvement_tips(prediction['fertility_class']),
                    'monitoring_schedule': self.get_monitoring_schedule(),
                    'seasonal_considerations': self.get_seasonal_considerations()
                }
            }
            
            # Display recommendation
            self.display_recommendation(complete_recommendation)
            
            return complete_recommendation
            
        except Exception as e:
            print(f"Error generating recommendation: {e}")
            return None
    
    def get_soil_improvement_tips(self, fertility_class):
        """Get soil improvement tips based on fertility class"""
        tips = {
            'high_fertility': [
                'Maintain current fertility levels with balanced fertilization',
                'Focus on micronutrient management',
                'Regular soil testing every 6 months'
            ],
            'medium_fertility': [
                'Gradual increase in organic matter',
                'Balanced NPK application',
                'Consider crop rotation with legumes'
            ],
            'low_fertility': [
                'Heavy organic matter addition required',
                'Soil conditioning with lime if pH is low',
                'Extended fallow periods with green manure crops'
            ]
        }
        return tips.get(fertility_class, tips['medium_fertility'])
    
    def get_monitoring_schedule(self):
        """Get soil monitoring schedule"""
        return {
            'soil_testing': 'Every 3-6 months',
            'pH_monitoring': 'Monthly during growing season',
            'nutrient_assessment': 'Before each crop cycle',
            'organic_matter_check': 'Annually'
        }
    
    def get_seasonal_considerations(self):
        """Get seasonal farming considerations"""
        return {
            'spring': 'Focus on nitrogen-rich fertilizers for new growth',
            'summer': 'Ensure adequate potassium for stress resistance',
            'monsoon': 'Monitor drainage and prevent nutrient leaching',
            'winter': 'Prepare soil with organic amendments'
        }
    
    def display_recommendation(self, recommendation):
        """Display comprehensive recommendation in formatted way"""
        print("\n📊 SOIL ANALYSIS RESULTS:")
        soil_analysis = recommendation['soil_analysis']
        print(f"   Fertile: {'Yes' if soil_analysis['is_fertile'] else 'No'}")
        print(f"   Fertility Rate: {soil_analysis['fertility_rate']:.2%}")
        print(f"   Fertility Class: {soil_analysis['fertility_class'].replace('_', ' ').title()}")
        
        print("\n🌱 RECOMMENDED CROP SEQUENCE:")
        for i, crop_info in enumerate(recommendation['crop_sequence'], 1):
            print(f"   {i}. {crop_info['season']}: {crop_info['crop']} (Priority: {crop_info['priority']})")
        
        print("\n🧪 FERTILIZER RECOMMENDATIONS:")
        fert_rec = recommendation['fertilizer_recommendation']
        print(f"   Primary: {', '.join(fert_rec['primary_fertilizers'])}")
        print(f"   Secondary: {', '.join(fert_rec['secondary_fertilizers'])}")
        print(f"   Organic: {', '.join(fert_rec['organic_options'])}")
        
        print("\n📅 APPLICATION SCHEDULE:")
        for stage, date in fert_rec['application_schedule'].items():
            print(f"   {stage.replace('_', ' ').title()}: {date}")
        
        print("\n💡 SOIL IMPROVEMENT TIPS:")
        for tip in recommendation['additional_recommendations']['soil_improvement']:
            print(f"   • {tip}")
    
    def save_model_complete(self, model_path, metadata_path=None):
        """Save the trained model with metadata"""
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            
            # Save the keras model
            self.model.save(model_path)
            
            # Save metadata (scalers, encoders, feature columns)
            if metadata_path is None:
                metadata_path = model_path.replace('.h5', '_metadata.pkl')
            
            metadata = {
                'feature_columns': self.feature_columns,
                'scaler': self.scaler,
                'label_encoders': self.label_encoders,
                'fertility_scaler': self.fertility_scaler
            }
            
            with open(metadata_path, 'wb') as f:
                pickle.dump(metadata, f)
            
            print(f"Model saved to: {model_path}")
            print(f"Metadata saved to: {metadata_path}")
            
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model_complete(self, model_path, metadata_path=None):
        """Load a pre-trained model with metadata"""
        try:
            # Load the keras model
            self.model = keras.models.load_model(model_path)
            
            # Load metadata
            if metadata_path is None:
                metadata_path = model_path.replace('.h5', '_metadata.pkl')
            
            with open(metadata_path, 'rb') as f:
                metadata = pickle.load(f)
            
            self.feature_columns = metadata['feature_columns']
            self.scaler = metadata['scaler']
            self.label_encoders = metadata['label_encoders']
            self.fertility_scaler = metadata['fertility_scaler']
            
            print(f"Model loaded from: {model_path}")
            print(f"Metadata loaded from: {metadata_path}")
            print(f"Feature columns: {len(self.feature_columns)}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
    
    def save_model(self, model_path):
        """Save the trained model (legacy method - calls complete save)"""
        self.save_model_complete(model_path)
    
    def load_model(self, model_path):
        """Load a pre-trained model"""
        try:
            self.model = keras.models.load_model(model_path)
            print(f"Model loaded from: {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}")

# Main execution function
def main():
    """Main function to execute the complete pipeline"""
    
    try:
        # Initialize model
        data_path = r'C:\Users\HP\Documents\Mini Project 5th sem\Final Model DNN\Final Dataset\Merge_data.csv'
        model = AgriculturePredictionModel(data_path)
        
        # Load and prepare data
        print("Step 1: Loading data...")
        data = model.load_and_prepare_data()
        if data is None:
            print("Failed to load data. Exiting...")
            return None, None, None
        
        # Create target variables
        print("\nStep 2: Creating target variables...")
        feature_cols = model.create_target_variables()
        
        if len(feature_cols) == 0:
            print("No feature columns found. Exiting...")
            return None, None, None
        
        # Preprocess features
        print("\nStep 3: Preprocessing features...")
        X, y_fertile, y_fertility_rate, y_fertility_class = model.preprocess_features(feature_cols)
        
        # Build model
        print("\nStep 4: Building model...")
        model.model = model.build_model(X.shape[1])
        
        # Train model
        print("\nStep 5: Training model...")
        history, test_data = model.train_model(X, y_fertile, y_fertility_rate, y_fertility_class, 
                                             epochs=30, batch_size=64)
        
        # Save model
        print("\nStep 6: Saving model...")
        model_save_path = r'C:\Users\HP\Documents\Mini Project 5th sem\Final Model DNN\agriculture_model.h5'
        model.save_model(model_save_path)
        
        # Example prediction
        print("\n" + "="*60)
        print("STEP 7: EXAMPLE PREDICTION")
        print("="*60)
        
        # Create sample soil data using the actual feature columns from training
        sample_soil_data = {}
        np.random.seed(42)  # For reproducible results
        
        for col in model.feature_columns:
            col_lower = col.lower()
            # Use realistic values based on column names
            if any(keyword in col_lower for keyword in ['temperature', 'temp']):
                sample_soil_data[col] = np.random.uniform(20, 35)
            elif any(keyword in col_lower for keyword in ['humidity', 'moisture', 'rh']):
                sample_soil_data[col] = np.random.uniform(40, 80)
            elif 'ph' in col_lower:
                sample_soil_data[col] = np.random.uniform(6.0, 7.5)
            elif col_lower in ['n', 'nitrogen']:
                sample_soil_data[col] = np.random.uniform(20, 80)
            elif col_lower in ['p', 'phosphorus']:
                sample_soil_data[col] = np.random.uniform(10, 50)
            elif col_lower in ['k', 'potassium']:
                sample_soil_data[col] = np.random.uniform(15, 60)
            elif any(keyword in col_lower for keyword in ['rainfall', 'rain']):
                sample_soil_data[col] = np.random.uniform(500, 2000)
            elif any(keyword in col_lower for keyword in ['light']):
                sample_soil_data[col] = np.random.uniform(8, 14)
            elif any(keyword in col_lower for keyword in ['ec', 'conductivity']):
                sample_soil_data[col] = np.random.uniform(0.5, 2.5)
            elif any(keyword in col_lower for keyword in ['oc', 'organic']):
                sample_soil_data[col] = np.random.uniform(0.5, 3.0)
            elif any(keyword in col_lower for keyword in ['sand', 'silt', 'clay']):
                sample_soil_data[col] = np.random.uniform(10, 60)
            elif any(keyword in col_lower for keyword in ['zn', 'fe', 'cu', 'mn', 'b', 's']):
                sample_soil_data[col] = np.random.uniform(1, 20)
            else:
                # Default random value for other parameters
                # Check the actual data range for this column
                if col in model.data.columns:
                    col_min = model.data[col].min()
                    col_max = model.data[col].max()
                    if pd.isna(col_min) or pd.isna(col_max) or col_min == col_max:
                        sample_soil_data[col] = np.random.uniform(0, 100)
                    else:
                        sample_soil_data[col] = np.random.uniform(col_min, col_max)
                else:
                    sample_soil_data[col] = np.random.uniform(0, 100)
        
        print(f"Sample soil data created with {len(sample_soil_data)} features")
        print("Sample values (first 10):")
        for i, (key, value) in enumerate(list(sample_soil_data.items())[:10]):
            print(f"  {key}: {value:.2f}")
        if len(sample_soil_data) > 10:
            print("  ...")
        
        # Generate complete recommendation
        print("\nStep 8: Generating recommendation...")
        recommendation = model.generate_complete_recommendation(sample_soil_data)
        
        if recommendation:
            print("\n" + "="*60)
            print("PIPELINE COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"Model trained on {len(model.data)} samples")
            print(f"Using {len(model.feature_columns)} features")
            print(f"Model saved to: {model_save_path}")
        
        return model, history, recommendation
        
    except Exception as e:
        print(f"Error in main pipeline: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None

def create_simple_prediction_interface(model):
    """Create a simple interface for making predictions with custom soil data"""
    
    def predict_soil_fertility(**soil_params):
        """
        Make prediction with custom soil parameters
        
        Example usage:
        predict_soil_fertility(
            temperature=25.0, humidity=65.0, ph=6.8, 
            n=45, p=23, k=38, rainfall=1200
        )
        """
        try:
            # Fill missing parameters with defaults
            complete_soil_data = {}
            for col in model.feature_columns:
                if col.lower() in [k.lower() for k in soil_params.keys()]:
                    # Find the matching key (case-insensitive)
                    matching_key = next(k for k in soil_params.keys() if k.lower() == col.lower())
                    complete_soil_data[col] = soil_params[matching_key]
                else:
                    # Use median value from training data or reasonable default
                    if col in model.data.columns:
                        complete_soil_data[col] = model.data[col].median()
                    else:
                        complete_soil_data[col] = 50.0  # Default value
            
            return model.generate_complete_recommendation(complete_soil_data)
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return None
    
    return predict_soil_fertility


# Execute the pipeline
if __name__ == "__main__":
    print("Starting Agricultural Prediction Model Training...")
    print("="*60)
    
    trained_model, training_history, sample_recommendation = main()
    
    if trained_model is not None:
        print("\n" + "="*60)
        print("CREATING PREDICTION INTERFACE")
        print("="*60)
        
        # Create prediction interface
        predict_function = create_simple_prediction_interface(trained_model)
        
        print("Prediction interface created successfully!")
        print("\nYou can now make predictions using:")
        print("predict_function(temperature=25, humidity=65, ph=6.8, n=45, p=23, k=38)")
        
        # Example of custom prediction
        print("\nExample custom prediction:")
        custom_prediction = predict_function(
            temperature=28.5,
            humidity=72.0, 
            ph=6.9,
            n=52,
            p=28,
            k=41,
            rainfall=1400
        )
        
    else:
        print("Model training failed. Please check the error messages above.")
        
        
        
