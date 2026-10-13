from sklearn.metrics import mean_squared_error,mean_absolute_error

def evaluate_model(y_true, y_pred):
    return{
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
    }