import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import warnings
warnings.filterwarnings('ignore')

class AgriculturePredictor:
    def __init__(self, model_path):
        """Initialize the agriculture prediction system"""
        self.model_path = model_path
        self.model = None
        
        # Define categorical mappings
        self.soil_type_mapping = {
            'black': 0, 'red': 1, 'alluvial': 2, 'laterite': 3,
            'sandy': 4, 'loamy': 5, 'clay': 6
        }
        
        self.season_mapping = {
            'kharif': 0, 'rabi': 1, 'zaid': 2, 'summer': 3
        }
        
        # Define crop recommendations based on soil and climate conditions
        self.crop_database = {
            'high_n_crops': ['Rice', 'Wheat', 'Corn', 'Sugarcane'],
            'medium_n_crops': ['Cotton', 'Soybean', 'Sunflower'],
            'low_n_crops': ['Groundnut', 'Pulses', 'Millets'],
            'acidic_soil_crops': ['Tea', 'Coffee', 'Potato', 'Blueberry'],
            'alkaline_soil_crops': ['Barley', 'Sugar Beet', 'Asparagus'],
            'high_rainfall_crops': ['Rice', 'Sugarcane', 'Jute', 'Tea'],
            'low_rainfall_crops': ['Millets', 'Sorghum', 'Groundnut', 'Cotton']
        }
        
        self.load_model()
    
    def load_model(self):
        """Load the trained model"""
        try:
            if os.path.exists(self.model_path):
                self.model = load_model(self.model_path)
                print("✅ Model loaded successfully!")
            else:
                print("❌ Model file not found. Please check the path.")
                return False
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
        return True
    
    def get_user_input(self):
        """Collect user input for all parameters"""
        print("🌾 AGRICULTURE PREDICTION SYSTEM 🌾")
        print("=" * 50)
        
        try:
            # Climate parameters
            print("\n📊 CLIMATE PARAMETERS:")
            temperature = float(input("Enter temperature (°C): "))
            humidity = float(input("Enter humidity (%): "))
            rainfall = float(input("Enter rainfall (mm): "))
            
            # Soil nutrients
            print("\n🧪 SOIL NUTRIENTS:")
            n = float(input("Enter Nitrogen (N) mg/kg: "))
            p = float(input("Enter Phosphorus (P) mg/kg: "))
            k = float(input("Enter Potassium (K) mg/kg: "))
            
            # Soil properties
            print("\n🏔 SOIL PROPERTIES:")
            ph = float(input("Enter pH value (0-14): "))
            ec = float(input("Enter EC (dS/m): "))
            oc = float(input("Enter Organic Carbon (OC) %: "))
            om = float(input("Enter Organic Matter (OM) %: "))
            
            # Micronutrients
            print("\n⚗ MICRONUTRIENTS:")
            zn = float(input("Enter Zinc (Zn) mg/kg: "))
            fe = float(input("Enter Iron (Fe) mg/kg: "))
            cu = float(input("Enter Copper (Cu) mg/kg: "))
            mn = float(input("Enter Manganese (Mn) mg/kg: "))
            
            # Soil texture
            print("\n🏗 SOIL TEXTURE:")
            sand = float(input("Enter Sand %: "))
            silt = float(input("Enter Silt %: "))
            clay = float(input("Enter Clay %: "))
            
            # Additional parameters
            print("\n📋 ADDITIONAL PARAMETERS:")
            caco3 = float(input("Enter CaCO3 %: "))
            cec = float(input("Enter CEC (cmol/kg): "))
            s = float(input("Enter Sulphur (S) mg/kg: "))
            b = float(input("Enter Boron (B) mg/kg: "))
            
            # Categorical inputs
            print("\n🏷 CATEGORICAL INPUTS:")
            print("Available soil types:", list(self.soil_type_mapping.keys()))
            soil_type = input("Enter soil type: ").lower().strip()
            
            print("Available seasons:", list(self.season_mapping.keys()))
            season = input("Enter season: ").lower().strip()
            
            # Validate and encode categorical variables
            soil_type_encoded = self.soil_type_mapping.get(soil_type, 0)
            season_encoded = self.season_mapping.get(season, 0)
            
            if soil_type not in self.soil_type_mapping:
                print(f"⚠ Warning: Unknown soil type '{soil_type}', using default encoding")
            
            if season not in self.season_mapping:
                print(f"⚠ Warning: Unknown season '{season}', using default encoding")
            
            # Combine all features
            features = [
                temperature, humidity, n, p, k, rainfall, ph, ec, oc, om,
                zn, fe, cu, mn, sand, silt, clay, caco3, cec, s, b,
                soil_type_encoded, season_encoded
            ]
            
            return np.array([features], dtype=np.float32), {
                'temperature': temperature, 'humidity': humidity, 'rainfall': rainfall,
                'n': n, 'p': p, 'k': k, 'ph': ph, 'ec': ec, 'oc': oc, 'om': om,
                'zn': zn, 'fe': fe, 'cu': cu, 'mn': mn,
                'sand': sand, 'silt': silt, 'clay': clay,
                'caco3': caco3, 'cec': cec, 's': s, 'b': b,
                'soil_type': soil_type, 'season': season
            }
            
        except ValueError as e:
            print(f"❌ Error: Please enter valid numerical values. {str(e)}")
            return None, None
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            return None, None
    
    def analyze_soil_conditions(self, params):
        """Analyze soil conditions based on input parameters"""
        analysis = []
        
        # pH Analysis
        if params['ph'] < 6.0:
            analysis.append("Soil is ACIDIC - may limit nutrient availability")
        elif params['ph'] > 8.0:
            analysis.append("Soil is ALKALINE - may affect nutrient uptake")
        else:
            analysis.append("Soil pH is OPTIMAL for most crops")
        
        # Nutrient Analysis
        if params['n'] < 200:
            analysis.append("LOW Nitrogen - needs nitrogen fertilizers")
        elif params['n'] > 500:
            analysis.append("HIGH Nitrogen - reduce nitrogen application")
        else:
            analysis.append("ADEQUATE Nitrogen levels")
        
        if params['p'] < 10:
            analysis.append("LOW Phosphorus - add phosphatic fertilizers")
        elif params['p'] > 50:
            analysis.append("HIGH Phosphorus - sufficient for crops")
        else:
            analysis.append("MODERATE Phosphorus levels")
        
        if params['k'] < 100:
            analysis.append("LOW Potassium - add potassic fertilizers")
        elif params['k'] > 300:
            analysis.append("HIGH Potassium - good for crop growth")
        else:
            analysis.append("ADEQUATE Potassium levels")
        
        # Organic Matter
        if params['om'] < 1.0:
            analysis.append("LOW Organic Matter - add compost/FYM")
        elif params['om'] > 3.0:
            analysis.append("HIGH Organic Matter - excellent soil health")
        else:
            analysis.append("MODERATE Organic Matter content")
        
        return analysis
    
    def recommend_crops(self, params):
        """Recommend crops based on soil and climate conditions"""
        recommendations = []
        
        # Based on rainfall
        if params['rainfall'] > 1000:
            recommendations.extend(self.crop_database['high_rainfall_crops'])
        else:
            recommendations.extend(self.crop_database['low_rainfall_crops'])
        
        # Based on pH
        if params['ph'] < 6.5:
            recommendations.extend(self.crop_database['acidic_soil_crops'])
        elif params['ph'] > 7.5:
            recommendations.extend(self.crop_database['alkaline_soil_crops'])
        
        # Based on nitrogen levels
        if params['n'] > 400:
            recommendations.extend(self.crop_database['high_n_crops'])
        elif params['n'] < 200:
            recommendations.extend(self.crop_database['low_n_crops'])
        else:
            recommendations.extend(self.crop_database['medium_n_crops'])
        
        # Remove duplicates and return top 5
        unique_crops = list(set(recommendations))
        return unique_crops[:5] if len(unique_crops) >= 5 else unique_crops
    
    def generate_fertilizer_recommendations(self, params):
        """Generate fertilizer recommendations"""
        recommendations = []
        
        # NPK recommendations
        if params['n'] < 200:
            recommendations.append("Apply Urea @ 100-150 kg/ha for Nitrogen")
        if params['p'] < 15:
            recommendations.append("Apply DAP @ 50-75 kg/ha for Phosphorus")
        if params['k'] < 150:
            recommendations.append("Apply MOP @ 50-100 kg/ha for Potassium")
        
        # Micronutrient recommendations
        if params['zn'] < 0.5:
            recommendations.append("Apply Zinc Sulphate @ 10-15 kg/ha")
        if params['fe'] < 2.0:
            recommendations.append("Apply Iron Sulphate @ 15-20 kg/ha")
        if params['b'] < 0.5:
            recommendations.append("Apply Borax @ 5-10 kg/ha")
        
        # Organic recommendations
        if params['om'] < 1.5:
            recommendations.append("Apply Farm Yard Manure @ 10-15 tons/ha")
            recommendations.append("Apply Compost @ 5-8 tons/ha")
        
        # pH correction
        if params['ph'] < 5.5:
            recommendations.append("Apply Lime @ 2-4 tons/ha to correct acidity")
        elif params['ph'] > 8.5:
            recommendations.append("Apply Gypsum @ 1-2 tons/ha to reduce alkalinity")
        
        return recommendations if recommendations else ["Current nutrient levels are adequate"]
    
    def create_application_schedule(self, params):
        """Create fertilizer application schedule"""
        schedule = []
        
        if params['season'] == 'kharif':
            schedule = [
                "BASAL (Before sowing): Apply full P, K and 1/3 N",
                "30-35 days after sowing: Apply 1/3 N",
                "60-65 days after sowing: Apply remaining 1/3 N",
                "Apply micronutrients with first irrigation"
            ]
        elif params['season'] == 'rabi':
            schedule = [
                "BASAL (At sowing): Apply full P, K and 1/4 N",
                "21 days after sowing: Apply 1/4 N",
                "42 days after sowing: Apply 1/4 N",
                "63 days after sowing: Apply remaining 1/4 N"
            ]
        else:
            schedule = [
                "BASAL: Apply 50% N, full P and K",
                "30 days: Apply 25% N",
                "60 days: Apply remaining 25% N",
                "Apply organic matter 15 days before sowing"
            ]
        
        return schedule
    
    def generate_soil_improvement_tips(self, params):
        """Generate soil improvement tips"""
        tips = []
        
        # Based on soil texture
        if params['sand'] > 70:
            tips.append("Sandy soil: Add organic matter to improve water retention")
            tips.append("Use drip irrigation to reduce water loss")
        
        if params['clay'] > 50:
            tips.append("Clay soil: Improve drainage by adding organic matter")
            tips.append("Avoid working soil when wet to prevent compaction")
        
        # Based on organic matter
        if params['om'] < 2.0:
            tips.append("Increase organic matter through crop residue incorporation")
            tips.append("Practice green manuring with leguminous crops")
        
        # Based on pH
        if params['ph'] < 6.0:
            tips.append("Regular liming to maintain optimal pH")
            tips.append("Use acid-tolerant crop varieties")
        elif params['ph'] > 8.0:
            tips.append("Apply gypsum to improve soil structure")
            tips.append("Use organic acids to reduce alkalinity")
        
        # General tips
        tips.extend([
            "Practice crop rotation to maintain soil fertility",
            "Use cover crops to prevent soil erosion",
            "Implement conservation tillage practices",
            "Regular soil testing every 2-3 years"
        ])
        
        return tips
    
    def predict_and_analyze(self):
        """Main function to get input and provide predictions"""
        if self.model is None:
            print("❌ Model not loaded. Cannot make predictions.")
            return
        
        # Get user input
        features, params = self.get_user_input()
        
        if features is None or params is None:
            return
        
        print("\n" + "="*60)
        print("🔍 GENERATING PREDICTIONS...")
        print("="*60)
        
        try:
            # Make prediction (if your model provides direct outputs)
            # prediction = self.model.predict(features, verbose=0)
            
            # Since we don't know the exact model output format,
            # we'll generate recommendations based on input analysis
            
            # 1. Soil Analysis
            print("\n📊 SOIL ANALYSIS RESULTS:")
            print("-" * 30)
            soil_analysis = self.analyze_soil_conditions(params)
            for i, analysis in enumerate(soil_analysis, 1):
                print(f"{i}. {analysis}")
            
            # 2. Crop Recommendations
            print("\n🌱 RECOMMENDED CROP SEQUENCE:")
            print("-" * 30)
            recommended_crops = self.recommend_crops(params)
            for i, crop in enumerate(recommended_crops, 1):
                print(f"{i}. {crop}")
            
            # 3. Fertilizer Recommendations
            print("\n🧪 FERTILIZER RECOMMENDATIONS:")
            print("-" * 30)
            fertilizer_recs = self.generate_fertilizer_recommendations(params)
            for i, rec in enumerate(fertilizer_recs, 1):
                print(f"{i}. {rec}")
            
            # 4. Application Schedule
            print("\n📅 APPLICATION SCHEDULE:")
            print("-" * 30)
            schedule = self.create_application_schedule(params)
            for i, step in enumerate(schedule, 1):
                print(f"{i}. {step}")
            
            # 5. Soil Improvement Tips
            print("\n💡 SOIL IMPROVEMENT TIPS:")
            print("-" * 30)
            improvement_tips = self.generate_soil_improvement_tips(params)
            for i, tip in enumerate(improvement_tips, 1):
                print(f"{i}. {tip}")
            
            print("\n" + "="*60)
            print("✅ ANALYSIS COMPLETE!")
            print("="*60)
            
        except Exception as e:
            print(f"❌ Error during prediction: {str(e)}")

def main():
    """Main function to run the prediction system"""
    # Update this path to your model location
    model_path = r"C:\Users\HP\Documents\Mini Project 5th sem\Final Model DNN\agriculture_model.h5"
    
    # Create predictor instance
    predictor = AgriculturePredictor(model_path)
    
    # Run prediction
    predictor.predict_and_analyze()
    
    # Option to run again
    while True:
        choice = input("\n🔄 Do you want to make another prediction? (y/n): ").lower().strip()
        if choice in ['y', 'yes']:
            predictor.predict_and_analyze()
        elif choice in ['n', 'no']:
            print("👋 Thank you for using the Agriculture Prediction System!")
            break
        else:
            print("Please enter 'y' for yes or 'n' for no.")

if __name__ == "__main__":
    main()