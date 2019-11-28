from normalize_data import getNormalizedDataset
from utils import target_encoding
from sklearn.ensemble import RandomForestRegressor


def pred_with_rf(df, df_test):
    
    #Obtener df normalizado y con feature engineering aplicado
    df_norm = getNormalizedDataset(df)
    df_test_norm = getNormalizedDataset(df_test, 'test')
    df_train, target_enc = target_encoding(df_norm,'train')
    
    #Data de train y test
    X_train = df_train
    y_train = df_norm['precio']
    X_test, target_enc = target_encoding(df_test_norm, 'test', target_enc)

    #Uso los hiperparametros que obtuve como resultado del random search
    rf = RandomForestRegressor(n_estimators=743, min_samples_split=5, min_samples_leaf=2,\
                           max_features='sqrt', max_depth=60, bootstrap= False)
    rf.fit(X_train,y_train)
    
    # Todos aquellos que sean de una categoria que yo no tenia en el set de datos, lo mando a la mas comun. #TODO
    X_test.ciudad.fillna(X_test.ciudad.value_counts().idxmax(), inplace=True)
    X_test.ciudad.fillna(X_test.provincia.value_counts().idxmax(), inplace=True)
    X_test.ciudad.fillna(X_test.tipodepropiedad.value_counts().idxmax(), inplace=True)

    pred = rf.predict(X_test)
    
    return pred
    