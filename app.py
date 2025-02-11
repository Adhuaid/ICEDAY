import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

class StudentProgressTracker:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.performance_levels = {0: 'Low', 1: 'Medium', 2: 'High'}  # Add performance level mapping
        
    def load_and_prepare_data(self, filepath):
        """Load and prepare the xAPI-Edu-Data dataset"""
        # Load the data using the provided filepath
        df = pd.read_csv(r"C:\Users\prpra\New folder (2)\model\xAPI-Edu-Data (2).csv")
        
        # Convert categorical variables to numerical using Label Encoding
        categorical_columns = df.select_dtypes(include=['object']).columns
        for column in categorical_columns:
            self.label_encoders[column] = LabelEncoder()
            df[column] = self.label_encoders[column].fit_transform(df[column])
        
        return df
    
    def prepare_features(self, df):
        """Prepare features and target variable"""
        # Use 'Class' as target variable
        X = df.drop(['Class'], axis=1)
        y = df['Class']
        
        return X, y
    
    def train_model(self, X, y):
        """Train the model with student data"""
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        
    def predict_progress(self, student_features):
        """Predict student progress based on input features"""
        student_features_scaled = self.scaler.transform(student_features)
        return self.model.predict(student_features_scaled)
    
    def get_feature_importance(self, X):
        """Get the importance of each feature in predicting progress"""
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': self.model.feature_importances_
        })
        return feature_importance.sort_values('importance', ascending=False)
    
    def plot_feature_importance(self, feature_importance):
        """Plot feature importance"""
        plt.figure(figsize=(12, 6))
        sns.barplot(x='importance', y='feature', data=feature_importance)
        plt.title('Feature Importance for Student Progress')
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, y_test, y_pred):
        """Plot confusion matrix"""
        plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()
        
    def analyze_performance_factors(self, df):
        """Analyze key performance factors"""
        plt.figure(figsize=(15, 5))
        
        # Plot 1: Raised Hands vs Class
        plt.subplot(1, 3, 1)
        sns.boxplot(x='Class', y='raisedhands', data=df)
        plt.title('Raised Hands vs Class')
        
        # Plot 2: Resources Visited vs Class
        plt.subplot(1, 3, 2)
        sns.boxplot(x='Class', y='VisITedResources', data=df)
        plt.title('Visited Resources vs Class')
        
        # Plot 3: Discussion Participation vs Class
        plt.subplot(1, 3, 3)
        sns.boxplot(x='Class', y='Discussion', data=df)
        plt.title('Discussion Participation vs Class')
        
        plt.tight_layout()
        plt.show()
    
    def get_performance_trends(self, student_data, time_periods=5):
        """Analyze student performance trends over time"""
        trends = {
            'raisedhands': student_data['raisedhands'].values[0],
            'VisITedResources': student_data['VisITedResources'].values[0],
            'Discussion': student_data['Discussion'].values[0]
        }
        return trends
    
    def generate_recommendations(self, student_data, feature_importance):
        """Generate personalized recommendations based on student data"""
        recommendations = []
        
        # Analyze raised hands participation
        if student_data['raisedhands'].values[0] < 50:
            recommendations.append("Increase class participation by raising hands more frequently")
            
        # Analyze resource utilization
        if student_data['VisITedResources'].values[0] < 50:
            recommendations.append("Spend more time exploring educational resources")
            
        # Analyze discussion participation
        if student_data['Discussion'].values[0] < 50:
            recommendations.append("Participate more actively in class discussions")
            
        return recommendations

def generate_student_report(student_data, prediction, feature_importance, tracker):
    """Generate a detailed report for a student"""
    performance_level = tracker.performance_levels[prediction[0]]
    
    print("\nStudent Progress Report")
    print("=" * 50)
    print(f"Predicted Performance Level: {performance_level}")
    
    # Performance Metrics
    print("\nKey Performance Metrics:")
    print("-" * 30)
    print(f"Class Participation (Raised Hands): {student_data['raisedhands'].values[0]}/100")
    print(f"Resource Utilization: {student_data['VisITedResources'].values[0]}/100")
    print(f"Discussion Engagement: {student_data['Discussion'].values[0]}/100")
    
    # Get performance trends
    trends = tracker.get_performance_trends(student_data)
    print("\nPerformance Trends:")
    print("-" * 30)
    for metric, value in trends.items():
        print(f"{metric}: {'↑' if value > 50 else '↓'} ({value})")
    
    # Get top 5 areas for improvement based on feature importance
    print("\nKey Areas for Improvement:")
    print("-" * 30)
    top_features = feature_importance.head()
    for _, row in top_features.iterrows():
        print(f"- {row['feature']} (Impact Score: {row['importance']:.3f})")
    
    # Generate personalized recommendations
    recommendations = tracker.generate_recommendations(student_data, feature_importance)
    print("\nPersonalized Recommendations:")
    print("-" * 30)
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec}")

def main():
    # Initialize the tracker
    tracker = StudentProgressTracker()
    
    try:
        # Load and prepare data - Updated path to the dataset
        data = tracker.load_and_prepare_data(r"C:\Users\prpra\New folder (2)\model\xAPI-Edu-Data (2).csv")
        
        # Prepare features and target
        X, y = tracker.prepare_features(data)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train the model
        tracker.train_model(X_train, y_train)
        
        # Make predictions
        y_pred = tracker.predict_progress(X_test)
        
        # Print model evaluation metrics
        print("Model Performance Evaluation:")
        print("=" * 50)
        print(classification_report(y_test, y_pred))
        
        # Get and display feature importance
        feature_importance = tracker.get_feature_importance(X)
        
        # Visualizations
        tracker.plot_feature_importance(feature_importance)
        tracker.plot_confusion_matrix(y_test, y_pred)
        tracker.analyze_performance_factors(data)
        
        # Example: Predict progress for a new student
        new_student = pd.DataFrame([X_test.iloc[0]], columns=X.columns)
        predicted_class = tracker.predict_progress(new_student)
        
        # Generate comprehensive student report
        generate_student_report(new_student, predicted_class, feature_importance, tracker)
        
    except FileNotFoundError:
        print("Error: Dataset file 'xAPI-Edu-Data.csv' not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    main()