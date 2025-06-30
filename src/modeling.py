from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

def get_models():
    models = {
        'KNN': {
            'model': KNeighborsClassifier(),
            'params': {'n_neighbors': list(range(3, 12, 2))}
        },
        'SVM': {
            'model': SVC(probability=True, random_state=42),
            'params': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        },
        'MLP': {
            'model': MLPClassifier(max_iter=500, random_state=42),
            'params': {'hidden_layer_sizes': [(50,), (100,)], 
                       'activation': ['relu', 'tanh']}
        },
        'XGBoost': {
            'model': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
            'params': {'n_estimators': [50, 100], 
                       'max_depth': [3, 5], 
                       'learning_rate': [0.01, 0.1]}
        }
    }
    return models

def nested_cv(model, param_grid, X, y):
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    
    results = []
    
    for train_idx, test_idx in outer_cv.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        clf = GridSearchCV(model, param_grid, cv=inner_cv, scoring='accuracy', n_jobs=-1)
        clf.fit(X_train, y_train)
        
        best_model = clf.best_estimator_
        results.append({
            'best_params': clf.best_params_,
            'best_score': clf.best_score_,
            'test_score': best_model.score(X_test, y_test),
            'model': best_model,
            'test_set': (X_test, y_test)
        })
    
    return results