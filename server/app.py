from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib
import os

app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://localhost:5173", "http://127.0.0.1:5173"],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type"]
    }
})

# Add a test endpoint
@app.route('/', methods=['GET'])
def home():
    return jsonify({"message": "Server is running"}), 200

# Add a route to check if the data file exists
@app.route('/check-data', methods=['GET'])
def check_data():
    file_path = r'C:\Users\prpra\New folder (2)\model\xAPI-Edu-Data (2).csv'
    if os.path.exists(file_path):
        return jsonify({"message": "Data file found"}), 200
    else:
        return jsonify({"error": f"Data file not found at {file_path}"}), 404

class StudentProgressTracker:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.performance_levels = {0: 'Low', 1: 'Medium', 2: 'High'}
        self.required_features = ['raisedhands', 'VisITedResources', 'Discussion']
        
    def load_and_prepare_data(self, filepath):
        try:
            if not os.path.exists(filepath):
                raise FileNotFoundError(f"Data file not found at {filepath}")
            
            df = pd.read_csv(filepath)
            print(f"Loaded data shape: {df.shape}")  # Debug print
            
            categorical_columns = df.select_dtypes(include=['object']).columns
            print(f"Categorical columns: {categorical_columns}")  # Debug print
            
            for column in categorical_columns:
                self.label_encoders[column] = LabelEncoder()
                df[column] = self.label_encoders[column].fit_transform(df[column])
            
            return df
        except Exception as e:
            print(f"Error in load_and_prepare_data: {str(e)}")  # Debug print
            raise

    def prepare_features(self, df):
        """Prepare features and target variable"""
        # Only use the features we're collecting from the frontend
        X = df[self.required_features]
        y = df['Class'] if 'Class' in df.columns else None
        return X, y
    
    def train_model(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
    def predict_progress(self, student_features):
        """Predict student progress based on input features"""
        # Ensure we only use the required features in the correct order
        features_to_use = student_features[self.required_features]
        student_features_scaled = self.scaler.transform(features_to_use)
        return self.model.predict(student_features_scaled)

tracker = StudentProgressTracker()

@app.route('/train', methods=['POST'])
def train():
    try:
        file_path = r'C:\Users\prpra\New folder (2)\model\xAPI-Edu-Data (2).csv'
        print(f"Training model with data from: {file_path}")
        
        data = tracker.load_and_prepare_data(file_path)
        X, y = tracker.prepare_features(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        tracker.train_model(X_train, y_train)
        
        # Save the trained model state
        tracker.is_trained = True
        return jsonify({"message": "Model trained successfully"})
    except Exception as e:
        print(f"Training error: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not hasattr(tracker, 'is_trained') or not tracker.is_trained:
            return jsonify({"error": "Please train the model first"}), 400
            
        if not request.json:
            return jsonify({"error": "No data provided"}), 400
            
        student_data = request.json
        print(f"Received student data: {student_data}")
        
        # Validate required fields
        required_fields = ['raisedhands', 'VisITedResources', 'Discussion']
        for field in required_fields:
            if field not in student_data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Create DataFrame with only the required features
        input_data = {
            'raisedhands': float(student_data['raisedhands']),
            'VisITedResources': float(student_data['VisITedResources']),
            'Discussion': float(student_data['Discussion'])
        }
        
        df = pd.DataFrame([input_data])
        prediction = tracker.predict_progress(df)
        performance_level = tracker.performance_levels[prediction[0]]
        
        # Calculate performance metrics
        metrics = {
            'participation_score': int(student_data['raisedhands']),
            'resource_usage': int(student_data['VisITedResources']),
            'discussion_engagement': int(student_data['Discussion']),
            'overall_engagement': int((int(student_data['raisedhands']) + 
                                    int(student_data['VisITedResources']) + 
                                    int(student_data['Discussion'])) / 3)
        }
        
        # Generate performance indicators
        indicators = {
            'participation': 'Good' if metrics['participation_score'] >= 70 else 'Needs Improvement',
            'resource_usage': 'Good' if metrics['resource_usage'] >= 70 else 'Needs Improvement',
            'discussion': 'Good' if metrics['discussion_engagement'] >= 70 else 'Needs Improvement'
        }
        
        return jsonify({
            "prediction": performance_level,
            "metrics": metrics,
            "indicators": indicators,
            "recommendations": generate_recommendations(df),
            "analysis_summary": f"Based on the analysis, the student shows {performance_level.lower()} performance. " +
                              f"Overall engagement is {metrics['overall_engagement']}%."
        })
    except Exception as e:
        print(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

def generate_recommendations(student_data):
    recommendations = []
    if student_data['raisedhands'].values[0] < 50:
        recommendations.append("Increase class participation by raising hands more frequently")
    if student_data['VisITedResources'].values[0] < 50:
        recommendations.append("Spend more time exploring educational resources")
    if student_data['Discussion'].values[0] < 50:
        recommendations.append("Participate more actively in class discussions")
    return recommendations

# Add this new endpoint after your existing routes
@app.route('/get-sample-data', methods=['GET'])
def get_sample_data():
    try:
        file_path = r'C:\Users\prpra\New folder (2)\model\xAPI-Edu-Data (2).csv'
        df = pd.read_csv(file_path)
        sample_data = df.head().to_dict('records')
        return jsonify({
            "data": sample_data,
            "columns": list(df.columns)
        })
    except Exception as e:
        print(f"Error getting sample data: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Add debug message at startup
    print("Starting Flask server...")
    print(f"Working directory: {os.getcwd()}")
    app.run(debug=True, port=5000) 