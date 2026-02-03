import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

def train_battery_model(telemetry_df):
    """
    Trains a classification model to predict Battery Risk Level.
    Target: 0=Low Risk, 1=Medium Risk, 2=High Risk
    Derived from battery_level and other mock features (like charge cycles if we had them).
    """
    df = telemetry_df.copy()
    
    # Mock Target Generation (since we don't have historical failure labels)
    # Logic: < 20% = High Risk, 20-50% = Medium Risk, > 50% = Low Risk
    def get_risk_label(level):
        if level < 20: return 'Critical'
        elif level < 50: return 'Medium'
        else: return 'Low'
        
    df['risk_level'] = df['battery_level'].apply(get_risk_label)
    
    # Features: battery_level, maybe hour of day (temperature proxy?)
    # For now, just using battery_level to demonstrate the pipeline
    X = df[['battery_level']]
    y = df['risk_level']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_train, y_train)
    
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    metrics = {'Accuracy': round(accuracy, 2)}
    
    return clf, metrics
