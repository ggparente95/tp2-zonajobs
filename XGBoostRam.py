# Métrica de evaluación
def RMSLE(actual, pred):
    actualLog = np.log(actual + 1)
    predLog = np.log(pred + 1)
    return (np.mean((actualLog - predLog) ** 2)) **.5

def normailize_df(refDf, train):
    df = refDf.copy()
    df['año'] = df['fecha'].dt.year
    df['antiguedad'] = df['antiguedad'] + (2016 - df['año']) # Se normaliza la antiguedad.
    df.antiguedad = df.antiguedad.fillna(0) #Asumo que si no tiene antiguedad entonces es nuevo
    df.drop(["direccion", 'descripcion', 'lat', 'lng', 'fecha', 'titulo'], axis=1, inplace=True)
    
    nulltotales = df[df['metrostotales'].isnull()]
    nullcubiertos = df[df['metroscubiertos'].isnull()]
    notnullapart = df[(~ df['metrostotales'].isnull()) & (df['metrostotales'] < df['metroscubiertos'])]
    notnullapart2 = df[(~ df['metrostotales'].isnull()) & (df['metrostotales'] > df['metroscubiertos'])]
    notnullapart3 = df[(~ df['metrostotales'].isnull()) & (df['metrostotales'] == df['metroscubiertos'])]
 
    df['habitable'] = False
    tipodepropiedades = df.tipodepropiedad.cat.categories.to_list()
    for tipodepropiedad in tipodepropiedades: 
              
        idsNullMetrosTotales = (df.tipodepropiedad == tipodepropiedad) & (df.metrostotales.isnull())
        idsNullMetrosCubiertos = (df.tipodepropiedad == tipodepropiedad) & (df.metroscubiertos.isnull())


        ## Verificamos si la cantidad de registros con metros cubiertos nulos es mayor a 2/5 de los totales. Si es asi 
        ## los consideramos propiedades no habitables. Y los tratamos de manera diferente
        if(len(nullcubiertos[nullcubiertos.tipodepropiedad == tipodepropiedad]) >= 2/5* len(df[df.tipodepropiedad == tipodepropiedad])):
            df.metrostotales.fillna(0, inplace=True)
            df.metroscubiertos.fillna(0, inplace=True)
        else:
            df[idsNullMetrosTotales]['metrostotales'] =  df[idsNullMetrosTotales]['metroscubiertos']
            df[idsNullMetrosCubiertos]['metroscubiertos'] =  df[idsNullMetrosCubiertos]['metrostotales']

        #Si la moda del tipo de propiedad de banos y habitaciones son ambas distintas de nan entonces la propiedad es habitable.
        banos = df[df.tipodepropiedad == tipodepropiedad].banos.mode(dropna=False);
        habitaciones = df[df.tipodepropiedad == tipodepropiedad].habitaciones.mode(dropna=False);
        df.loc[(df.tipodepropiedad == tipodepropiedad), 'habitable'] = not(np.isnan(banos[0]) and np.isnan(habitaciones[0]))
    
    df['metros'] = df['metrostotales'] + df['metroscubiertos']
    df.habitaciones = df.habitaciones.fillna(0)
    df.garages = df.garages.fillna(0)
    df.banos = df.banos.fillna(0)

    
    if(train):

        def is_outlier(group):
            Q1 = group.quantile(0.25)
            Q3 = group.quantile(0.75)
            IQR = Q3 - Q1
            precio_min = Q1 - 1.5 * IQR
            precio_max = Q3 + 1.5 * IQR
            return ~group.between(precio_min, precio_max)
        df['precio_mt2'] = df['precio'] / df['metros']
        print()
        
        df = df[~df.groupby('tipodepropiedad')['precio_mt2'].apply(is_outlier).fillna(False)]
        idDel = df[df.tipodepropiedad == 'Garage'].index
        df = df.drop(idDel)
        idDel = df[df.tipodepropiedad == 'Hospedaje'].index
        df = df.drop(idDel)
        print('Despues de filtrar: ', df.shape)
        cols = list(df.columns)
        cols =  cols[:15] + cols[16:] +[cols[15]]
        df = df[cols]

    return df

def getXGboostRam(pathTrain = '../train.csv', pathTest = '../train.csv', getTrain = False, getPred = True, predsTrain):
	import xgboost as xgb
	from sklearn.metrics import mean_squared_error
	import pandas as pd
	import numpy as np
	from sklearn.preprocessing import LabelEncoder
	import category_encoders as ce
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split

	#Archivos de train y test
	df_train = pd.read_csv(pathTrain,\
	        index_col='id',\
	        dtype={'gimnasio': int,\
	                'usosmultiples': int,\
	                'escuelascercanas': int,\
	                'piscina': int,\
	                'centroscomercialescercanos': int,\
	                'tipodepropiedad': 'category',\
	                'provincia': 'category',\
	                'ciudad': 'category'\
	            },\
	        parse_dates=['fecha'])
	df_test = pd.read_csv(pathTest,
	        index_col='id',\
	        dtype={'gimnasio': int,\
	                'usosmultiples': int,\
	                'escuelascercanas': int,\
	                'piscina': int,\
	                'centroscomercialescercanos': int,\
	                'tipodepropiedad': 'category',\
	                'provincia': 'category',\
	                'ciudad': 'category'\
	            },\
	        parse_dates=['fecha'])

	df_train = normailize_df(df_train, True)
	df_test = normailize_df(df_test, False)
	cat_features = ['tipodepropiedad', 'provincia', 'ciudad']
	target_enc = ce.TargetEncoder(cols=cat_features)



	# Fit the encoder using the categorical features and target
	target_enc.fit(df_train[cat_features], df_train['precio'])

	# Transform the features, rename the columns with _target suffix, and join to dataframe in Train
	df_train = df_train.join(target_enc.transform(df_train[cat_features]).add_suffix('_target'))
	df_train = df_train.drop(cat_features, axis=1)

	# Transform the features, rename the columns with _target suffix, and join to dataframe in Test
	df_test = df_test.join(target_enc.transform(df_test[cat_features]).add_suffix('_target'))
	df_test = df_test.drop(cat_features, axis=1)

	df_train.drop('precio_mt2', axis=1, inplace=True)
	X, y = df_train.drop('precio', axis = 1),df_train['precio']

	X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=123)

    xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', colsample_bytree = 0.8, learning_rate = 0.1, max_depth = 10, reg_alpha = 1.2, n_estimators = 500, reg_lambda = 1.4, subsample=0.8)

    xg_reg.fit(X_train,y_train)

    preds = xg_reg.predict(X_test)

    if getTrain:
    	Xt , yt = df_train.drop('precio', axis = 1),df_train['precio']
    	predsTrain = xg_reg.fit(Xt,yt)
    	res = pd.DataFrame(predsTrain, index=df_test.index, columns=['target'])
		res.to_csv("workshop-submission-XGBoostRa.csv", header=True)

    if getPred:
    	X_test = df_test
    	preds = xg_reg.predict(X_test)
    	res = pd.DataFrame(preds, index=df_test.index, columns=['target'])
		res.to_csv("workshop-submission-XGBoostRa.csv", header=True)

	return preds



