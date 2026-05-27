from flask import Flask, render_template, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)

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
                return True
            else:
                print("❌ Model file not found. Please check the path.")
                return False
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            return False
    
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
    
    def predict(self, form_data):
        """Process form data and make predictions"""
        try:
            # Convert form data to parameters
            params = {
                'temperature': float(form_data['temperature']),
                'humidity': float(form_data['humidity']),
                'rainfall': float(form_data['rainfall']),
                'n': float(form_data['n']),
                'p': float(form_data['p']),
                'k': float(form_data['k']),
                'ph': float(form_data['ph']),
                'ec': float(form_data['ec']),
                'oc': float(form_data['oc']),
                'om': float(form_data['om']),
                'zn': float(form_data['zn']),
                'fe': float(form_data['fe']),
                'cu': float(form_data['cu']),
                'mn': float(form_data['mn']),
                'sand': float(form_data['sand']),
                'silt': float(form_data['silt']),
                'clay': float(form_data['clay']),
                'caco3': float(form_data['caco3']),
                'cec': float(form_data['cec']),
                's': float(form_data['s']),
                'b': float(form_data['b']),
                'soil_type': form_data['soil_type'].lower(),
                'season': form_data['season'].lower()
            }
            
            # Generate predictions
            results = {
                'soil_analysis': self.analyze_soil_conditions(params),
                'recommended_crops': self.recommend_crops(params),
                'fertilizer_recommendations': self.generate_fertilizer_recommendations(params),
                'application_schedule': self.create_application_schedule(params),
                'soil_improvement_tips': self.generate_soil_improvement_tips(params)
            }
            
            return {'status': 'success', 'results': results}
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}

# Initialize predictor
MODEL_PATH = r"C:\Users\HP\Documents\Mini Project 5th sem\Final Model\agriculture_model.h5"
predictor = AgriculturePredictor(MODEL_PATH)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = request.form.to_dict()
        
        # Make prediction
        result = predictor.predict(form_data)
        
        if result['status'] == 'success':
            return render_template('results.html', results=result['results'], form_data=form_data)
        else:
            return render_template('index.html', error=result['message'])
            
    except Exception as e:
        return render_template('index.html', error=str(e))

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for AJAX requests"""
    try:
        data = request.get_json()
        result = predictor.predict(data)
        return jsonify(result)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)