import numpy as np
from sklearn import metrics

def describe_(df):
    print('shape', df.shape)
    for i in df:
        if df[i].dtypes == 'O':
            print("#############")
            print("Contagem de categorias:", i)
            print(df[i].value_counts())

def return_metrics(y_test, y_predicted):
    
    mse = np.sqrt(metrics.mean_squared_error(y_test, y_predicted))
    r2 = metrics.r2_score(y_test, y_predicted)
    mae = metrics.median_absolute_error(y_test, y_predicted)
    me = metrics.mean_absolute_error(y_test, y_predicted)
    mape = np.mean(np.abs((y_test - y_predicted) / y_test)) * 100
    
    return {'mse': mse,
            'r2': r2,
            'mae': mae,
            'me': me,
            'mape': mape}

# I made this function just to make the rounding process less verbose.
def r(x):
    return np.round(x, 2)