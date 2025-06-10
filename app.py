from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import warnings

warnings.filterwarnings('ignore')

app = Flask(__name__)

model = None
scaler = None
feature_selector = None
selected_features = None
feature_names = None


def train_and_save_model():
   
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    X = df.drop('target', axis=1)
    y = df['target']
    
    global scaler
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    global feature_selector, selected_features, feature_names
    feature_selector = SelectKBest(score_func=f_classif, k=15)
    X_selected = feature_selector.fit_transform(X_scaled, y)

    selected_features = X.columns[feature_selector.get_support()]
    feature_names = list(selected_features)

    print(f"Selected 15 features: {feature_names}")

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )

    nb_model = GaussianNB()
    nb_model.fit(X_train, y_train)
    nb_score = nb_model.score(X_test, y_test)
    print(f"Naive Bayes accuracy: {nb_score:.4f}")

    lr_param_grid = {
        'C': [0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear']
    }

    cv_folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    lr_grid_search = GridSearchCV(
        LogisticRegression(random_state=42, max_iter=1000),
        lr_param_grid,
        cv=cv_folds,
        scoring='accuracy',
        n_jobs=-1
    )
    lr_grid_search.fit(X_train, y_train)
    best_lr = lr_grid_search.best_estimator_
    lr_score = best_lr.score(X_test, y_test)
    print(f"Logistic Regression accuracy: {lr_score:.4f}")
    print(f"Best LR parameters: {lr_grid_search.best_params_}")

    global model
    model = VotingClassifier(
        estimators=[
            ('naive_bayes', nb_model),
            ('logistic_regression', best_lr)
        ],
        voting='soft' 
    )
    model.fit(X_train, y_train)
    ensemble_score = model.score(X_test, y_test)
    print(f"Ensemble accuracy: {ensemble_score:.4f}")

    # Save all components
    joblib.dump(model, 'breast_cancer_model.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    joblib.dump(feature_selector, 'feature_selector.pkl')
    joblib.dump(feature_names, 'feature_names.pkl')

    print("Model trained and saved successfully!")

    y_pred = model.predict(X_test)
    print("\nEnsemble Model Performance:")
    print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Malignant', 'Benign']))

    return ensemble_score


def load_model():
    """Load the trained model and preprocessing components"""
    global model, scaler, feature_selector, feature_names

    try:
        model = joblib.load('breast_cancer_model.pkl')
        scaler = joblib.load('scaler.pkl')
        feature_selector = joblib.load('feature_selector.pkl')
        feature_names = joblib.load('feature_names.pkl')
        print("Model loaded successfully!")
        print(f"Using {len(feature_names)} selected features: {feature_names}")
        return True
    except FileNotFoundError:
        print("Model files not found. Training new model...")
        train_and_save_model()
        load_model()
        return True

load_model()


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/selected_features')
def get_selected_features():
  
    return jsonify({'features': feature_names})


@app.route('/predict', methods=['POST'])
def predict():
    try:
      
        data = request.get_json()

        model_type = data.get('model_type', 'ensemble')

        features = []
        for feature_name in feature_names:
            features.append(float(data.get(feature_name, 0)))

        input_features = np.array(features).reshape(1, -1)

        full_features = np.zeros((1, 30))  

        original_data = load_breast_cancer()
        original_feature_names = list(original_data.feature_names)

        for i, selected_feature in enumerate(feature_names):
            original_idx = original_feature_names.index(selected_feature)
            full_features[0, original_idx] = features[i]

        scaled_features = scaler.transform(full_features)

        selected_features_data = feature_selector.transform(scaled_features)

        if model_type == 'naive_bayes':
   
            nb_model = model.named_estimators_['naive_bayes']
            prediction = nb_model.predict(selected_features_data)[0]
            probability = nb_model.predict_proba(selected_features_data)[0]
            model_name = "Naive Bayes"
        elif model_type == 'logistic':
  
            lr_model = model.named_estimators_['logistic_regression']
            prediction = lr_model.predict(selected_features_data)[0]
            probability = lr_model.predict_proba(selected_features_data)[0]
            model_name = "Logistic Regression"
        else:
        
            prediction = model.predict(selected_features_data)[0]
            probability = model.predict_proba(selected_features_data)[0]
            model_name = "Ensemble (NB + LR)"

        result = {
            'prediction': 'Benign' if prediction == 1 else 'Malignant',
            'probability_malignant': round(probability[0] * 100, 2),
            'probability_benign': round(probability[1] * 100, 2),
            'confidence': round(max(probability) * 100, 2),
            'model_used': model_name,
            'features_used': len(feature_names)
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({'error': str(e)}), 400


@app.route('/sample_data')
def sample_data():

    data = load_breast_cancer()
    sample_idx = np.random.randint(0, len(data.data))
    sample = data.data[sample_idx]
    sample_target = data.target[sample_idx]


    sample_dict = {}
    for feature_name in feature_names:
        feature_idx = list(data.feature_names).index(feature_name)
        sample_dict[feature_name] = round(sample[feature_idx], 4)

    return jsonify({
        'features': sample_dict,
        'actual_diagnosis': 'Benign' if sample_target == 1 else 'Malignant'
    })


@app.route('/model_info')
def model_info():
  
    try:
   
        data = load_breast_cancer()
        X = pd.DataFrame(data.data, columns=data.feature_names)
        y = data.target

        X_scaled = scaler.transform(X)
        X_selected = feature_selector.transform(X_scaled)

     
        _, X_test, _, y_test = train_test_split(
            X_selected, y, test_size=0.2, random_state=42, stratify=y
        )

        nb_score = model.named_estimators_['naive_bayes'].score(X_test, y_test)
        lr_score = model.named_estimators_['logistic_regression'].score(X_test, y_test)
        ensemble_score = model.score(X_test, y_test)

        return jsonify({
            'selected_features': feature_names,
            'num_features': len(feature_names),
            'model_scores': {
                'naive_bayes': round(nb_score, 4),
                'logistic_regression': round(lr_score, 4),
                'ensemble': round(ensemble_score, 4)
            }
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
