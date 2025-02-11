from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import joblib

app = Flask(__name__)
CORS(app)

class StudentProgressTracker:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.performance_levels = {0: 'Low', 1: 'Medium', 2: 'High'}
        
    def load_and_prepare_data(self, filepath):
        df = pd.read_csv(filepath)
        categorical_columns = df.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            self.label_encoders[column] = LabelEncoder()
            df[column] = self.label_encoders[column].fit_transform(df[column])
        return df
    
    def prepare_features(self, df):
        X = df.drop(['Class'], axis=1)
        y = df['Class']
        return X, y
    
    def train_model(self, X, y):
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
    def predict_progress(self, student_features):
        student_features_scaled = self.scaler.transform(student_features)
        return self.model.predict(student_features_scaled)

tracker = StudentProgressTracker()

@app.route('/train', methods=['POST'])
def train():
    try:
        data = tracker.load_and_prepare_data('model/xAPI-Edu-Data.csv')
        X, y = tracker.prepare_features(data)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        tracker.train_model(X_train, y_train)
        return jsonify({"message": "Model trained successfully"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        student_data = request.json
        df = pd.DataFrame([student_data])
        prediction = tracker.predict_progress(df)
        performance_level = tracker.performance_levels[prediction[0]]
        return jsonify({
            "prediction": performance_level,
            "recommendations": generate_recommendations(df)
        })
    except Exception as e:
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

if __name__ == '__main__':
    app.run(debug=True) 