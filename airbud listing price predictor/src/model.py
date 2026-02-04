from sklearn.ensemble import RandomForestRegressor

def build_model(n_estimators=100, max_depth=None, random_state=42):
    """
    Builds and returns a Random Forest Regressor model.
    """
    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1
    )
    return model
