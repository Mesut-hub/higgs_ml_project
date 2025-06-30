from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif

def select_features(X, y, method='anova', k=15):
    if method == 'anova':
        selector = SelectKBest(score_func=f_classif, k=k)
    elif method == 'mutual_info':
        selector = SelectKBest(score_func=mutual_info_classif, k=k)
    else:
        raise ValueError("Invalid method. Choose 'anova' or 'mutual_info'")
    
    X_selected = selector.fit_transform(X, y)
    return X_selected, selector.get_support(indices=True)