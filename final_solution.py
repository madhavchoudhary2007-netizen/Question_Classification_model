"""
Math Question Classifier - Text Classification System
======================================================

A classical machine learning system for automated classification of mathematics
questions into subtopics including Algebra, Calculus, Geometry, Statistics,
and Trigonometry.

Features:
- TF-IDF feature extraction with configurable parameters
- Multiple model comparison (Random Forest, SVM, Naive Bayes)
- Systematic ablation studies for hyperparameter validation
- Experiment tracking with Weights & Biases
- Comprehensive evaluation metrics and reporting

Author: [Your Name]
Institution: VIT Vellore
Project: CSI Club Selection Task
"""
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import re
import json
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import warnings
warnings.filterwarnings('ignore')



# Optional: Weights & Biases for experiment tracking
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: WandB not installed. Install with: pip install wandb")

class XGBStringAdapter:
    """
    A simple wrapper to make XGBoost work with string labels (like 'algebra').
    It automatically converts strings -> numbers for training,
    and numbers -> strings for prediction.
    """
    def __init__(self, **kwargs):
        self.model = XGBClassifier(**kwargs)
        self.le = LabelEncoder()

    def fit(self, X, y, **kwargs):
        # Encode string labels to integers (e.g., 'algebra' -> 0)
        y_encoded = self.le.fit_transform(y)
        self.model.fit(X, y_encoded, **kwargs)
        return self

    def predict(self, X):
        # Get integer predictions and convert back to strings
        y_pred = self.model.predict(X)
        return self.le.inverse_transform(y_pred)
    
    def get_raw_model(self):
        """Get the underlying XGBClassifier for use in sklearn ensembles"""
        return self.model
    
def preprocess_math(text):
    # Replace LaTeX numbers with a placeholder (helps generalized learning)
    text = re.sub(r'\d+', 'NUM', text) 
    # Remove LaTeX delimiters to clean up tokens
    text = text.replace('$', ' ') 
    return text    
    


def load_data_from_subfolders(data_folder):
    """
    Load question data from directory structure organized by topic.
    
    Expects directory structure:
        data_folder/
            TopicA/
                question1.json
                question2.json
            TopicB/
                question1.json
    
    Each JSON file should contain at minimum:
        {"question": "Question text here"}
    
    Args:
        data_folder (str): Path to root directory containing topic subdirectories
        
    Returns:
        tuple: (questions, labels, file_ids)
            - questions (list): Question text strings
            - labels (list): Topic labels extracted from directory names
            - file_ids (list): File identifiers for tracking purposes
    """
    questions = []
    labels = []
    file_ids = []
    
    data_path = Path(data_folder)
    
    # Validate directory existence
    if not data_path.exists():
        print(f"Error: Directory '{data_folder}' not found")
        return [], [], []
    
    # Identify topic subdirectories
    topic_folders = [d for d in data_path.iterdir() if d.is_dir()]
    
    if not topic_folders:
        print(f"Error: No subdirectories found in '{data_folder}'")
        return [], [], []
    
    print(f"Loading data from '{data_folder}'")
    print(f"Found {len(topic_folders)} topic categories")
    
    # Process each topic directory
    for folder in sorted(topic_folders):
        topic = folder.name
        json_files = list(folder.glob("*.json"))
        
        print(f"  {topic}: {len(json_files)} files")
        
        # Parse each JSON file
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                question = data.get('problem') or data.get('question') or ''
                
                if question:
                    questions.append(question)
                    labels.append(topic)
                    file_ids.append(f"{folder.name}/{json_file.stem}")
                    
            except Exception as e:
                print(f"  Warning: Error reading {json_file.name}: {e}")
    
    print(f"Successfully loaded {len(questions)} questions from {len(set(labels))} topics\n")
    return questions, labels, file_ids

def extract_math_features(questions):
    """
    Extract mathematical indicators that TF-IDF might miss.
    
    These features help distinguish between similar topics like
    Algebra vs Prealgebra (complexity) and identify specialized
    topics like Geometry (spatial terms) and Precalculus (advanced ops).
    
    Returns:
        numpy array: Shape (n_samples, n_features) with math indicators
    """
    features = []
    
    for text in questions:
        text_lower = text.lower()
        
        # Initialize feature vector
        feature_vec = []
        
        # ====================================================================
        # COMPLEXITY INDICATORS (helps separate Prealgebra from Algebra)
        # ====================================================================
        
        # Advanced calculus operations (Precalculus, Intermediate Algebra)
        feature_vec.append(int('\\int' in text))          # Has integral
        feature_vec.append(int('\\lim' in text))          # Has limit
        feature_vec.append(int('\\frac{d' in text))       # Has derivative
        feature_vec.append(int('\\sum' in text))          # Has summation
        feature_vec.append(int('\\prod' in text))         # Has product notation
        
        # Intermediate operations (Algebra, Intermediate Algebra)
        feature_vec.append(int('\\frac' in text))         # Has fraction
        feature_vec.append(int('\\sqrt' in text))         # Has square root
        feature_vec.append(int('^' in text or '**' in text))  # Has exponent
        feature_vec.append(int('\\log' in text or '\\ln' in text))  # Has logarithm
        
        # Basic operations (Prealgebra)
        basic_ops = ['+', '-', r'\times', r'\div', '=']
        feature_vec.append(sum(1 for op in basic_ops if op in text))
        
        # ====================================================================
        # TOPIC-SPECIFIC KEYWORDS
        # ====================================================================
        
        # Geometry indicators
        geometry_terms = ['triangle', 'circle', 'square', 'rectangle', 'angle',
                         'area', 'volume', 'perimeter', 'radius', 'diameter',
                         'polygon', 'vertex', 'parallel', 'perpendicular', 
                         'tangent', 'chord', 'sector']
        feature_vec.append(sum(1 for term in geometry_terms if term in text_lower))
        
        # Precalculus indicators
        precalc_terms = ['sin', 'cos', 'tan', 'cot', 'sec', 'csc',
                        'arcsin', 'arccos', 'arctan', 'limit', 'continuous',
                        'asymptote', 'polynomial', 'rational function']
        feature_vec.append(sum(1 for term in precalc_terms if term in text_lower))
        
        # Number theory indicators
        number_theory_terms = ['prime', 'factor', 'divisible', 'remainder',
                              'modulo', 'gcd', 'lcm', 'integer', 'digit',
                              'consecutive', 'multiple']
        feature_vec.append(sum(1 for term in number_theory_terms if term in text_lower))
        
        # Counting & Probability indicators
        counting_prob_terms = ['probability', 'permutation', 'combination',
                              'choose', 'ways', 'arrangements', 'odds',
                              'likely', 'chance', 'expected value']
        feature_vec.append(sum(1 for term in counting_prob_terms if term in text_lower))
        
        # ====================================================================
        # MATHEMATICAL STRUCTURE
        # ====================================================================
        
        # Count variables (a, b, c, x, y, z)
        variables = len(set(re.findall(r'\b[a-z]\b', text)))
        feature_vec.append(min(variables, 10))  # Cap at 10
        
        # Count numbers
        numbers = len(re.findall(r'\d+', text))
        feature_vec.append(min(numbers, 20))  # Cap at 20
        
        # Has inequality (<, >, ≤, ≥)
        feature_vec.append(int(any(ineq in text for ineq in ['<', '>', r'\leq', r'\geq', r'\le', r'\ge'])))
        
        # Has equation (=)
        feature_vec.append(text.count('='))
        
        # Has matrix notation
        feature_vec.append(int(r'\begin{pmatrix}' in text or r'\begin{bmatrix}' in text))
        
        # Question length (normalized)
        feature_vec.append(len(text) / 100.0)  # Normalize to ~1
        
        # Mathematical notation density
        latex_commands = len(re.findall(r'\\[a-zA-Z]+', text))
        feature_vec.append(latex_commands / max(len(text), 1) * 100)
        
        features.append(feature_vec)
    
    return np.array(features)

def create_features(train_questions, test_questions, max_features=1000, ngram_range=(1, 2)):
    """
    Transform text data into TF-IDF feature vectors.
    
    TF-IDF (Term Frequency-Inverse Document Frequency) converts text into
    numerical features by weighting terms based on their frequency within
    documents and rarity across the corpus.
    
    Args:
        train_questions (list): Training question texts
        test_questions (list): Test question texts
        max_features (int): Maximum number of features to extract
        ngram_range (tuple): Range of n-gram sizes (min, max)
            (1, 1): unigrams only
            (1, 2): unigrams and bigrams
            (1, 3): unigrams, bigrams, and trigrams
            
    Returns:
        tuple: (X_train, X_test, vectorizer)
            - X_train: Training feature matrix (sparse)
            - X_test: Test feature matrix (sparse)
            - vectorizer: Fitted TfidfVectorizer instance
    """
    print(f"Creating TF-IDF features")
    print(f"  max_features: {max_features}")
    print(f"  ngram_range: {ngram_range}")
    
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english'
    )
    
    # Fit vectorizer on training data only to prevent data leakage
    X_train = vectorizer.fit_transform(train_questions)
    X_test = vectorizer.transform(test_questions)
    
    print(f"Feature extraction complete: {X_train.shape[1]} features")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}\n")
    
    return X_train, X_test, vectorizer

def create_features_enhanced(train_questions, test_questions, max_features=5000, ngram_range=(1, 1)):
    """
    Enhanced feature creation combining TF-IDF with math-specific features.
    
    This hybrid approach:
    1. Uses TF-IDF for general vocabulary patterns
    2. Adds explicit math features TF-IDF might miss
    3. Combines both for richer representation
    
    Args:
        train_questions: Training question texts
        test_questions: Test question texts
        max_features: Max TF-IDF features
        ngram_range: N-gram range for TF-IDF
        
    Returns:
        X_train: Combined feature matrix for training
        X_test: Combined feature matrix for testing
        vectorizer: Fitted TF-IDF vectorizer
        feature_names: Names of all features (for interpretability)
    """
    print(f"Creating enhanced features (TF-IDF + Math-specific)")
    print(f"  TF-IDF: max_features={max_features}, ngram_range={ngram_range}")
    
    # Step 1: Create TF-IDF features
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words='english'
    )
    
    X_train_tfidf = vectorizer.fit_transform(train_questions)
    X_test_tfidf = vectorizer.transform(test_questions)
    
    print(f"  TF-IDF features: {X_train_tfidf.shape[1]}")
    
    # Step 2: Extract math-specific features
    X_train_math = extract_math_features(train_questions)
    X_test_math = extract_math_features(test_questions)
    
    print(f"  Math features: {X_train_math.shape[1]}")
    
    # Step 3: Combine both feature sets
    # Use scipy's hstack to combine sparse and dense matrices
    X_train = hstack([X_train_tfidf, X_train_math])
    X_test = hstack([X_test_tfidf, X_test_math])
    
    print(f"  Total features: {X_train.shape[1]}")
    print(f"  Training samples: {X_train.shape[0]}")
    print(f"  Test samples: {X_test.shape[0]}\n")
    
    # Create feature names for interpretability
    tfidf_names = vectorizer.get_feature_names_out().tolist()
    math_names = [
        'has_integral', 'has_limit', 'has_derivative', 'has_summation', 'has_product',
        'has_fraction', 'has_sqrt', 'has_exponent', 'has_log',
        'basic_ops_count', 'geometry_terms', 'precalc_terms', 
        'number_theory_terms', 'counting_prob_terms',
        'variable_count', 'number_count', 'has_inequality', 
        'equation_count', 'has_matrix', 'text_length', 'latex_density'
    ]
    feature_names = tfidf_names + math_names
    
    return X_train, X_test, vectorizer, feature_names


def train_model(X_train, y_train, model_type='svm'):
    """
    Initialize and train a classification model.
    
    Args:
        X_train: Training feature matrix
        y_train: Training labels
        model_type (str): Model architecture to use
            'svm': Support Vector Machine with linear kernel
            'rf': Random Forest ensemble
            'nb': Multinomial Naive Bayes
            
    Returns:
        Trained model instance
    """
    print(f"Training {model_type.upper()} classifier")
    
    if model_type == 'svm':
        model = SVC(kernel='linear', probability=True, random_state=42)
    elif model_type == 'rf':
        model = RandomForestClassifier(n_estimators=100, random_state=42)
    elif model_type == 'nb':
        model = MultinomialNB()
    else:
        print(f"Warning: Unknown model type '{model_type}', defaulting to SVM")
        model = SVC(kernel='linear', probability=True, random_state=42)
    
    model.fit(X_train, y_train)
    print(f"Model training complete\n")
    
    return model


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate trained model and generate performance metrics.
    
    Calculates multiple evaluation metrics:
    - Overall accuracy
    - Per-class precision, recall, and F1-score
    - Confusion matrix
    
    Args:
        model: Trained classifier
        X_test: Test feature matrix
        y_test: True test labels
        model_name (str): Model identifier for logging
        
    Returns:
        tuple: (accuracy, report_dict)
            - accuracy (float): Overall classification accuracy
            - report_dict (dict): Detailed metrics per class
    """
    print(f"Evaluating {model_name}")
    
    # Generate predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {accuracy:.2%}\n")
    
    # Display classification report
    print(f"Classification Report:")
    print("="*70)
    report = classification_report(y_test, y_pred)
    print(report)
    
    # Get report in dictionary format for logging
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    
    # Generate and display confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:")
    topics = sorted(set(y_test))
    
    # Print matrix header
    print(f"\n{'Actual ->':15s}", end="")
    for topic in topics:
        print(f"{topic[:8]:>10s}", end="")
    print("\n" + "-"*70)
    
    # Print matrix rows
    for i, topic in enumerate(topics):
        print(f"Pred: {topic[:12]:12s}", end="")
        for j in range(len(topics)):
            print(f"{cm[i][j]:>10d}", end="")
        print()
    print()
    
    return accuracy, report_dict


def run_ablation_studies(train_questions, train_labels, test_questions, test_labels):
    """
    Perform systematic ablation studies to validate hyperparameter choices.
    
    Ablation studies test the impact of individual design decisions by
    varying one parameter at a time while holding others constant.
    
    Studies performed:
    1. Vocabulary size impact (500, 1000, 2000 features)
    2. N-gram range impact (unigrams, bigrams, trigrams)
    
    Args:
        train_questions: Training question texts
        train_labels: Training topic labels
        test_questions: Test question texts
        test_labels: Test topic labels
        
    Returns:
        dict: Accuracy scores for each configuration tested
    """
    print("="*70)
    print("ABLATION STUDIES")
    print("="*70)
    
    results = {}
    
    # Study 1: Vocabulary size optimization
    print("\nStudy 1: Vocabulary Size Impact")
    print("-"*70)
    
    for max_feat in [500, 1000, 2000]:
        print(f"\nTesting max_features={max_feat}")
        
        X_train, X_test, _ = create_features(
            train_questions, test_questions,
            max_features=max_feat,
            ngram_range=(1, 2)
        )
        
        model = train_model(X_train, train_labels, 'rf')
        accuracy, _ = evaluate_model(model, X_test, test_labels, f"RF-{max_feat}")
        
        results[f'vocab_{max_feat}'] = accuracy
    
    # Study 2: N-gram range optimization
    print("\nStudy 2: N-gram Range Impact")
    print("-"*70)
    
    for ngram in [(1, 1), (1, 2), (1, 3)]:
        print(f"\nTesting ngram_range={ngram}")
        
        X_train, X_test, _ = create_features(
            train_questions, test_questions,
            max_features=1000,
            ngram_range=ngram
        )
        
        model = train_model(X_train, train_labels, 'rf')
        accuracy, _ = evaluate_model(model, X_test, test_labels, f"RF-{ngram}")
        
        results[f'ngram_{ngram}'] = accuracy
    
    return results

def calculate_class_weights(y_train):
    """
    Calculate balanced class weights for imbalanced data.
    
    Your dataset has imbalance:
    - Algebra: 1744 samples (majority)
    - Precalculus: 746 samples (minority)
    
    Class weights give more importance to minority classes
    during training, improving their accuracy.
    """
    from sklearn.utils.class_weight import compute_class_weight
    
    print("\nCalculating class weights for balanced training...")
    
    classes = np.unique(y_train)
    weights = compute_class_weight('balanced', classes=classes, y=y_train)
    
    class_weight_dict = dict(zip(classes, weights))
    
    print("Class weights (higher = minority class):")
    for cls, weight in sorted(class_weight_dict.items(), key=lambda x: x[1], reverse=True):
        print(f"  {cls:25s}: {weight:.3f}")
    
    return class_weight_dict



def compare_models(X_train, y_train, X_test, y_test):
    """
    Train and evaluate multiple classification algorithms for comparison.
    
    Compares three classical ML approaches:
    - Random Forest: Ensemble of decision trees with majority voting
    - Support Vector Machine: Optimal hyperplane separation
    - Naive Bayes: Probabilistic classifier based on Bayes' theorem
    
    Args:
        X_train: Training feature matrix
        y_train: Training labels
        X_test: Test feature matrix
        y_test: Test labels
        
    Returns:
        tuple: (results, best_model_name)
            - results (dict): Performance metrics for each model
            - best_model_name (str): Identifier of best-performing model
    """
    print("="*70)
    print("MODEL COMPARISON")
    print("="*70)
    
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'XGBoost': XGBStringAdapter(n_estimators=200, eval_metric='mlogloss', random_state=42),
        'SVM': SVC(kernel='linear', probability=True, random_state=42)
    }
    
    results = {}
    
    # Train and evaluate each model
    for model_name, model in models.items():
        print(f"\n{'='*70}")
        print(f"Testing: {model_name}")
        print(f"{'='*70}\n")
        
        model.fit(X_train, y_train)
        accuracy, report = evaluate_model(model, X_test, y_test, model_name)
        
        results[model_name] = {
            'accuracy': accuracy,
            'report': report,
            'model': model
        }
    
    # Identify best performer
    best_model_name = max(results.items(), key=lambda x: x[1]['accuracy'])[0]
    best_accuracy = results[best_model_name]['accuracy']
    
    print(f"\n{'='*70}")
    print(f"Best Model: {best_model_name} ({best_accuracy:.2%} accuracy)")
    print(f"{'='*70}\n")
    
    return results, best_model_name

def tune_xgboost(X_train, y_train, X_test, y_test):
    """
    Fine-tune XGBoost hyperparameters using Grid Search.
    
    Tests different combinations to find optimal settings.
    Takes ~5-10 minutes but provides significant accuracy boost.
    """
    from sklearn.model_selection import GridSearchCV
    from sklearn.preprocessing import LabelEncoder
    from xgboost import XGBClassifier
    
    print("="*70)
    print("HYPERPARAMETER TUNING - XGBOOST")
    print("="*70)
    print("\nSearching for optimal hyperparameters...")
    print("This will take 5-10 minutes...\n")
    
    # CRITICAL FIX: Encode string labels to integers
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    y_test_encoded = le.transform(y_test)
    
    # Define parameter grid
    param_grid = {
        'n_estimators': [200, 300],           # Number of trees
        'max_depth': [7, 9],                  # Tree depth
        'learning_rate': [0.05, 0.1],         # Learning speed
        'subsample': [0.8, 1.0],              # Sample ratio
        'colsample_bytree': [0.8, 1.0],       # Feature ratio per tree
        'min_child_weight': [1, 3]            # Minimum samples in leaf
    }
    
    # Total combinations: 2*2*2*2*2*2 = 64 tests
    print(f"Testing 64 hyperparameter combinations")
    
    # Base model (use raw XGBClassifier, not wrapper)
    xgb_base = XGBClassifier(
        random_state=42,
        eval_metric='mlogloss',
        use_label_encoder=False
    )
    
    # Calculate class weights
    class_weights = calculate_class_weights(y_train)
    
    # Convert to sample weights using ENCODED labels
    sample_weights = np.array([class_weights[label] for label in y_train])
    
    # Grid search with 3-fold cross-validation
    grid_search = GridSearchCV(
        xgb_base,
        param_grid,
        cv=3,                    # 3-fold CV (faster than 5-fold)
        scoring='accuracy',
        n_jobs=-1,              # Use all CPU cores
        verbose=2               # Show progress
    )
    
    # FIT with encoded labels
    grid_search.fit(X_train, y_train_encoded, sample_weight=sample_weights)
    
    print(f"\n" + "="*70)
    print("TUNING COMPLETE!")
    print("="*70)
    print(f"\nBest parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\nBest cross-validation score: {grid_search.best_score_:.2%}")
    
    # Get best model
    best_xgb_raw = grid_search.best_estimator_
    
    # Wrap in XGBStringAdapter for compatibility with rest of pipeline
    best_xgb = XGBStringAdapter(**grid_search.best_params_)
    best_xgb.fit(X_train, y_train)  # Fit with original string labels
    
    # Evaluate with string labels
    accuracy, report = evaluate_model(best_xgb, X_test, y_test, "Tuned XGBoost")
    
    return best_xgb, accuracy, grid_search.best_params_


def tune_random_forest(X_train, y_train, X_test, y_test):
    """
    Fine-tune Random Forest using Randomized Search.
    
    Faster than grid search - tests random combinations.
    """
    from sklearn.model_selection import RandomizedSearchCV
    from sklearn.ensemble import RandomForestClassifier
    
    print("="*70)
    print("HYPERPARAMETER TUNING - RANDOM FOREST")
    print("="*70)
    print("\nUsing randomized search (faster than grid search)")
    print("Testing 30 random combinations...\n")
    
    # Define parameter distributions
    param_dist = {
        'n_estimators': [200, 250, 300],
        'max_depth': [20, 25, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2'],
        'bootstrap': [True, False]
    }
    
    rf_base = RandomForestClassifier(random_state=42)
    
    # Randomized search
    random_search = RandomizedSearchCV(
        rf_base,
        param_dist,
        n_iter=30,              # Try 30 random combinations
        cv=3,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=2
    )
    
    random_search.fit(X_train, y_train)
    
    print(f"\n" + "="*70)
    print("TUNING COMPLETE!")
    print("="*70)
    print(f"\nBest parameters found:")
    for param, value in random_search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"\nBest cross-validation score: {random_search.best_score_:.2%}")
    
    # Get best model
    best_rf = random_search.best_estimator_
    
    # Evaluate
    accuracy, report = evaluate_model(best_rf, X_test, y_test, "Tuned Random Forest")
    
    return best_rf, accuracy, random_search.best_params_

def create_voting_ensemble(X_train, y_train, X_test, y_test, tuned_models=None):
    """
    Create voting ensemble combining multiple models.
    
    Soft voting uses prediction probabilities from each model
    and averages them. Usually 0.5-1% better than best single model.
    
    Args:
        tuned_models: Dict of tuned models (optional, uses default if None)
    """
    from sklearn.ensemble import VotingClassifier
    from xgboost import XGBClassifier
    
    print("="*70)
    print("CREATING VOTING ENSEMBLE")
    print("="*70)
    print("\nCombining predictions from multiple models...")
    
    # If tuned models provided, use them
    if tuned_models and 'xgb' in tuned_models:
        # Extract raw XGBClassifier from adapter if needed
        if hasattr(tuned_models['xgb'], 'get_raw_model'):
            xgb = tuned_models['xgb'].get_raw_model()
        else:
            xgb = tuned_models['xgb']
        rf = tuned_models['rf']
        print("Using tuned models for ensemble")
    else:
        # Use default parameters
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=7,
            learning_rate=0.1,
            random_state=42,
            eval_metric='mlogloss',
            use_label_encoder=False
        )
        rf = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            random_state=42
        )
        print("Using default model parameters")
    
    svm = SVC(
        kernel='linear',
        probability=True,
        C=1.0,
        random_state=42
    )
    
    # Create voting ensemble
    # Weights based on individual performance (XGBoost best, SVM worst)
    ensemble = VotingClassifier(
        estimators=[
            ('xgb', xgb),
            ('rf', rf),
            ('svm', svm)
        ],
        voting='soft',        # Use probabilities (better than hard voting)
        weights=[3, 2, 1]    # XGBoost=3x, RF=2x, SVM=1x
    )
    
    print("\nTraining voting ensemble (3 models)...")
    print("  XGBoost (weight: 3)")
    print("  Random Forest (weight: 2)")
    print("  SVM (weight: 1)")
    
    ensemble.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating voting ensemble...")
    accuracy, report = evaluate_model(ensemble, X_test, y_test, "Voting Ensemble")
    
    return ensemble, accuracy


def create_stacking_ensemble(X_train, y_train, X_test, y_test, tuned_models=None):
    """
    Create stacking ensemble (more advanced than voting).
    
    Stacking trains a meta-learner on predictions from base models.
    Usually 1-2% better than voting ensemble.
    Takes longer to train but worth it!
    """
    from sklearn.ensemble import StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from xgboost import XGBClassifier
    
    print("="*70)
    print("CREATING STACKING ENSEMBLE")
    print("="*70)
    print("\nStacking uses predictions as features for meta-learner...")
    
    # Base models (level 0)
    if tuned_models and 'xgb' in tuned_models:
        # Extract raw XGBClassifier from adapter if needed
        if hasattr(tuned_models['xgb'], 'get_raw_model'):
            xgb_raw = tuned_models['xgb'].get_raw_model()
        else:
            xgb_raw = tuned_models['xgb']
        
        base_models = [
            ('xgb', xgb_raw),
            ('rf', tuned_models['rf']),
            ('svm', SVC(kernel='linear', probability=True, random_state=42))
        ]
        print("Using tuned base models")
    else:
        base_models = [
            ('xgb', XGBClassifier(n_estimators=200, max_depth=7, 
                                 learning_rate=0.1, random_state=42,
                                 eval_metric='mlogloss', use_label_encoder=False)),
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=20, 
                                         random_state=42)),
            ('svm', SVC(kernel='linear', probability=True, random_state=42))
        ]
        print("Using default base models")
    
    # Meta-learner (level 1) - learns how to combine base predictions
    meta_learner = LogisticRegression(
        max_iter=1000,
        random_state=42
    )
    
    print("\nBase models (Level 0):")
    print("  - XGBoost")
    print("  - Random Forest")
    print("  - SVM")
    print("\nMeta-learner (Level 1):")
    print("  - Logistic Regression")
    
    # Create stacking ensemble
    stacking = StackingClassifier(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5,                      # 5-fold CV for base predictions
        n_jobs=-1
    )
    
    print("\nTraining stacking ensemble...")
    print("This takes 3-5 minutes (training base models + meta-learner)...")
    
    stacking.fit(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating stacking ensemble...")
    accuracy, report = evaluate_model(stacking, X_test, y_test, "Stacking Ensemble")
    
    return stacking, accuracy

def save_everything(best_model, vectorizer, results, ablation_results):
    """
    Persist trained model, vectorizer, and evaluation results to disk.
    
    Saves:
    - Trained model (serialized with joblib)
    - TF-IDF vectorizer (for consistent feature extraction)
    - Results summary (JSON format for human readability)
    
    Args:
        best_model: Best-performing trained model
        vectorizer: Fitted TfidfVectorizer instance
        results: Model comparison results
        ablation_results: Ablation study results
    """
    print("Saving models and results")
    
    # Create output directories
    Path('models').mkdir(exist_ok=True)
    Path('outputs').mkdir(exist_ok=True)
    
    # Serialize model and vectorizer
    joblib.dump(best_model, 'models/best_model.pkl')
    joblib.dump(vectorizer, 'models/vectorizer.pkl')
    print("  Model artifacts saved to models/")
    
    # Save results in JSON format
    results_summary = {
        'model_comparison': {k: v['accuracy'] for k, v in results.items()},
        'ablation_studies': ablation_results
    }
    
    with open('outputs/results.json', 'w') as f:
        json.dump(results_summary, f, indent=2)
    print("  Results saved to outputs/results.json")
    
    print()


def main():
    """
    Execute complete training and evaluation pipeline.
    
    Pipeline stages:
    1. Data loading (training and test sets)
    2. Feature extraction (TF-IDF)
    3. Model training and comparison
    4. Ablation studies
    5. Model persistence
    """
    print("="*70)
    print("MATH QUESTION CLASSIFIER")
    print("CSI Club VIT Vellore Selection Task")
    print("="*70)
    
    # Configuration parameters
    TRAIN_FOLDER = "dataset/train"
    TEST_FOLDER = "dataset/test"
    USE_WANDB = WANDB_AVAILABLE and True
    RUN_ABLATION_STUDIES = True
    
    # Initialize experiment tracking
    if USE_WANDB:
        try:
            wandb.init(
                project="math-question-classifier-csi",
                config={
                    "train_folder": TRAIN_FOLDER,
                    "test_folder": TEST_FOLDER,
                    "feature_method": "TF-IDF",
                    "max_features": 1000,
                    "ngram_range": (1, 2)
                }
            )
            print("\nWandB experiment tracking initialized")
        except Exception as e:
            print(f"\nWarning: WandB initialization failed: {e}")
            USE_WANDB = False
    
    # Stage 1: Load training data
    print("\n" + "="*70)
    print("STAGE 1: TRAINING DATA LOADING")
    print("="*70 + "\n")
    
    train_questions, train_labels, train_ids = load_data_from_subfolders(TRAIN_FOLDER)
    
    if len(train_questions) == 0:
        print("Error: No training data loaded. Verify directory structure.")
        return
    
    # Display training data statistics
    train_df = pd.DataFrame({'topic': train_labels})
    print("Training Data Distribution:")
    print(train_df['topic'].value_counts())
    print()
    
    # Stage 2: Load test data
    print("="*70)
    print("STAGE 2: TEST DATA LOADING")
    print("="*70 + "\n")
    
    test_questions, test_labels, test_ids = load_data_from_subfolders(TEST_FOLDER)
    
    if len(test_questions) == 0:
        print("Error: No test data loaded. Verify directory structure.")
        return
    
    # Display test data statistics
    test_df = pd.DataFrame({'topic': test_labels})
    print("Test Data Distribution:")
    print(test_df['topic'].value_counts())
    print()
    
    print("Preprocessing math text...")
    train_questions = [preprocess_math(q) for q in train_questions]
    test_questions = [preprocess_math(q) for q in test_questions]
    
    # Stage 3: Feature extraction
    print("="*70)
    print("STAGE 3: FEATURE EXTRACTION")
    print("="*70 + "\n")
    
    X_train, X_test, vectorizer, feature_names = create_features_enhanced(
        train_questions, test_questions,
        max_features=5000,
        ngram_range=(1, 1)
    )

    
    # Stage 4: Model training and comparison
    print("="*70)
    print("STAGE 4: MODEL TRAINING AND COMPARISON")
    print("="*70 + "\n")
    
    model_results, best_model_name = compare_models(
        X_train, train_labels,
        X_test, test_labels
    )
    
    # Log metrics to WandB if available
    if USE_WANDB:
        for model_name, result in model_results.items():
            wandb.log({
                f"{model_name}_accuracy": result['accuracy'],
                f"{model_name}_precision": result['report']['weighted avg']['precision'],
                f"{model_name}_recall": result['report']['weighted avg']['recall'],
                f"{model_name}_f1": result['report']['weighted avg']['f1-score']
            })
    
    # Stage 4.5: Hyperparameter Tuning
    print("\n" + "="*70)
    print("STAGE 4.5: HYPERPARAMETER TUNING")
    print("="*70 + "\n")
    
    print("Fine-tuning best models for optimal performance...")
    print("This will take 10-15 minutes total\n")
    
    # Tune XGBoost
    print("1/2: Tuning XGBoost...")
    tuned_xgb, tuned_xgb_acc, xgb_params = tune_xgboost(
        X_train, train_labels,
        X_test, test_labels
    )
    
    # Tune Random Forest
    print("\n2/2: Tuning Random Forest...")
    tuned_rf, tuned_rf_acc, rf_params = tune_random_forest(
        X_train, train_labels,
        X_test, test_labels
    )
    
    # Add to results
    model_results['Tuned XGBoost'] = {
        'accuracy': tuned_xgb_acc,
        'report': {},
        'model': tuned_xgb,
        'params': xgb_params
    }
    
    model_results['Tuned Random Forest'] = {
        'accuracy': tuned_rf_acc,
        'report': {},
        'model': tuned_rf,
        'params': rf_params
    }
    
    # Update best model if tuning improved it
    if tuned_xgb_acc > model_results[best_model_name]['accuracy']:
        best_model = tuned_xgb
        best_model_name = "Tuned XGBoost"
        print(f"\n🎉 Hyperparameter tuning improved accuracy!")
        print(f"   {model_results['XGBoost']['accuracy']:.2%} → {tuned_xgb_acc:.2%}")
        print(f"   Gain: +{(tuned_xgb_acc - model_results['XGBoost']['accuracy'])*100:.2f}%")
    
    # Log to WandB
    if USE_WANDB:
        wandb.log({
            "Tuned_XGBoost_accuracy": tuned_xgb_acc,
            "Tuned_RF_accuracy": tuned_rf_acc
        })
    
    
    # Stage 4.6: Ensemble Methods
    print("\n" + "="*70)
    print("STAGE 4.6: ENSEMBLE METHODS")
    print("="*70 + "\n")
    
    print("Creating ensemble models to combine predictions...")
    print("Ensemble methods typically add 1-2% accuracy\n")
    
    # Prepare tuned models dict
    tuned_models_dict = {
        'xgb': tuned_xgb,
        'rf': tuned_rf
    }
    
    # Create voting ensemble
    print("1/2: Creating Voting Ensemble...")
    ensemble_voting, voting_acc = create_voting_ensemble(
        X_train, train_labels,
        X_test, test_labels,
        tuned_models=tuned_models_dict
    )
    
    # Create stacking ensemble (more powerful)
    print("\n2/2: Creating Stacking Ensemble...")
    ensemble_stacking, stacking_acc = create_stacking_ensemble(
        X_train, train_labels,
        X_test, test_labels,
        tuned_models=tuned_models_dict
    )
    
    # Add to results
    model_results['Voting Ensemble'] = {
        'accuracy': voting_acc,
        'report': {},
        'model': ensemble_voting
    }
    
    model_results['Stacking Ensemble'] = {
        'accuracy': stacking_acc,
        'report': {},
        'model': ensemble_stacking
    }
    
    # Update best model if ensemble is better
    if stacking_acc > model_results[best_model_name]['accuracy']:
        best_model = ensemble_stacking
        old_best_acc = model_results[best_model_name]['accuracy']
        best_model_name = "Stacking Ensemble"
        
        print(f"\n🎉 Stacking ensemble is the new best model!")
        print(f"   {old_best_acc:.2%} → {stacking_acc:.2%}")
        print(f"   Gain: +{(stacking_acc - old_best_acc)*100:.2f}%")
    
    # Log to WandB
    if USE_WANDB:
        wandb.log({
            "Voting_Ensemble_accuracy": voting_acc,
            "Stacking_Ensemble_accuracy": stacking_acc
        })
    
    
    # Stage 5: Ablation studies
    ablation_results = {}
    if RUN_ABLATION_STUDIES:
        print("\n" + "="*70)
        print("STAGE 5: ABLATION STUDIES")
        print("="*70 + "\n")
        
        ablation_results = run_ablation_studies(
            train_questions, train_labels,
            test_questions, test_labels
        )
        
        # Log ablation results
        if USE_WANDB:
            for study_name, accuracy in ablation_results.items():
                wandb.log({f"ablation_{study_name}": accuracy})
    
    # Stage 6: Model persistence
    print("="*70)
    print("STAGE 6: MODEL AND RESULTS PERSISTENCE")
    print("="*70 + "\n")
    
    best_model = model_results[best_model_name]['model']
    save_everything(best_model, vectorizer, model_results, ablation_results)
    
    # Final summary
    print("="*70)
    print("TRAINING PIPELINE COMPLETE")
    print("="*70)
    print(f"\nFinal Results Summary:")
    print(f"  Training samples: {len(train_questions)}")
    print(f"  Test samples: {len(test_questions)}")
    print(f"  Number of topics: {len(set(train_labels))}")
    print(f"\nBest Performing Model: {best_model_name}")
    print(f"  Test Set Accuracy: {model_results[best_model_name]['accuracy']:.2%}")
    
    if RUN_ABLATION_STUDIES:
        print(f"\nAblation Study Results:")
        for study, acc in ablation_results.items():
            print(f"  {study}: {acc:.2%}")
    
    print(f"\nPersisted Artifacts:")
    print(f"  - models/best_model.pkl")
    print(f"  - models/vectorizer.pkl")
    print(f"  - outputs/results.json")
    
    if USE_WANDB:
        print(f"\nExperiment Dashboard:")
        print(f"  {wandb.run.get_url()}")
        wandb.finish()
    
    print("\nSystem ready for deployment")
    print("="*70)


if __name__ == "__main__":
    main()
