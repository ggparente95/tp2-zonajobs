import pandas as pd
import numpy as np

def is_outlier(group):
    ''' Para cada grupo devuelve por cada valor si es o no outlier. '''
    Q1 = group.quantile(0.25)
    Q3 = group.quantile(0.75)
    IQR = Q3 - Q1
    precio_min = Q1 - 1.5 * IQR
    precio_max = Q3 + 1.5 * IQR
    return ~group.between(precio_min, precio_max)


def delete_invalid_registers(df_orig):
    
    # Elimino propiedades que no tengan ni ciudad o provincia.
    df = df_orig.dropna(subset=['ciudad','provincia'])
    
    # Elimino aquellas propiedades sin tipo. (Son pocas).
    df = df_orig.dropna(subset=['tipodepropiedad'])
    
    return df


def fill_nans(df_orig):
    
    df = df_orig.copy()
    
    # Completo los nans de metros totales, con lo que tengan en la columna de metros cubiertos.
    df['metrostotales'].fillna(df['metroscubiertos'], inplace=True)

    # El resto de las filas con metros cubiertos nulo, son terrenos. Por ende los completo con 0.
    df['metroscubiertos'].fillna(0, inplace=True)

    # Agrego a los que son nans, los tipos, ciudad y provincia mas comunes.
    df['tipodepropiedad'].fillna(df['tipodepropiedad'].value_counts().idxmax(), inplace=True)
    df['ciudad'].fillna(df['ciudad'].value_counts().idxmax(), inplace=True)
    df['provincia'].fillna(df['provincia'].value_counts().idxmax(), inplace=True)
    
    # Asigno como baños y habitaciones nulas el promedio para el tipo de propiedad por ciudad en el que esta. 
    # No se puede asignarle "0" baños o "0" habitaciones porque no tendria sentido.
    df['banos'] = df.groupby(['tipodepropiedad','ciudad'])['banos'].transform(lambda x: x.fillna(x.mode()))
    df['habitaciones'] = df.groupby(['tipodepropiedad','ciudad'])['habitaciones'].transform(lambda x: x.fillna(x.mode()))
    df['banos'] = df.groupby(['tipodepropiedad'])['banos'].transform(lambda x: x.fillna(x.mode()))
    df['habitaciones'] = df.groupby(['tipodepropiedad'])['habitaciones'].transform(lambda x: x.fillna(x.mode()))

    # Aquellos que no tenian un valor de baño o habitacion en los grupos anteriores, los relleno con 0.
    df['banos'].fillna(0, inplace = True)
    df['habitaciones'].fillna(0, inplace = True)

    # Completo con la moda, despues si no se relleno ahi, completo con 0
    df['garages'] = df.groupby(['tipodepropiedad','ciudad'])['garages'].transform(lambda x: x.fillna(x.mode()))
    df['garages'].fillna(0, inplace=True)
    df['antiguedad'].fillna(0, inplace=True)    

    return df

    
def getNormalizedDataset(df_orig, mode='train'):
    ''' Devuelve el dataset de propiedades, completando valores nulos, arreglando errores y removiendo outliers. '''

    df = df_orig.copy()
    
    if mode=='train':    
        
        df = delete_invalid_registers(df)
        
        df = fill_nans(df)

        df.loc[:,'precio_m2'] = df['precio']/df['metrostotales']

        # Limpio los outliers
        df = df[~df.groupby('tipodepropiedad')['precio_m2'].apply(is_outlier)]

        df.drop('precio_m2', axis=1, inplace=True)
        
        # Hay 70000 filas donde los metros totales son menores a los cubiertos. Esto es invalido, pero son muchos datos para descartar
        # Se asigna para estos casos, los metros cubiertos como totales.
        df.loc[df['metrostotales']<df['metroscubiertos'], 'metrostotales'] = df['metroscubiertos']

        df.loc[:,'extras'] = df['garages']+df['piscina']+df['usosmultiples']+df['gimnasio']
    
    else:
        
        df = fill_nans(df)
        
        df.loc[df['metrostotales']<df['metroscubiertos'], 'metrostotales'] = df['metroscubiertos']

        df.loc[:,'extras'] = df['garages']+df['piscina']+df['usosmultiples']+df['gimnasio']
        
        #df = df.drop(['garages','piscina','usosmultiples','gimnasio']

 
    df.drop(['direccion','idzona','lat','lng','titulo','descripcion'], axis=1, inplace=True)
    df.loc[:,'garages'] = df['garages'].astype(int)
    df.loc[:,'antiguedad'] = df['antiguedad'].astype(int)
    df.loc[:,'banos'] = df['banos'].astype(int)
    df.loc[:,'habitaciones'] = df['habitaciones'].astype(int)
    df.loc[:,'metroscubiertos'] = df['metroscubiertos'].astype(int)
    df.loc[:,'metrostotales'] = df['metrostotales'].astype(int)
    df.loc[:,'dia'] = df.fecha.dt.day
    df.loc[:,'mes'] = df.fecha.dt.month
    df.loc[:,'anio'] = df.fecha.dt.year
    df.drop('fecha', axis=1, inplace=True)
    
    return df


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
        #df.loc[(df.tipodepropiedad == tipodepropiedad), 'habitable'] = not(np.isnan(banos[0]) and np.isnan(habitaciones[0]))
    
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


def transformar_antiguedad(x):
    if x<=20:
        return 1
    if x>20 and x<=40:
        return 2
    if x>40 and x<=60:
        return 3
    if x>60:
        return 4

    
def getNormalizedDataset_2(df_orig, mode='train'):
    ''' Devuelve el dataset de propiedades, completando valores nulos, arreglando errores y removiendo outliers. '''

    df = df_orig.copy()
    
    if mode=='train':    
        
        df = delete_invalid_registers(df)
        
        df = fill_nans(df)

        df.loc[:,'precio_m2'] = df['precio']/df['metrostotales']

        # Limpio los outliers
        df = df[~df.groupby('tipodepropiedad')['precio_m2'].apply(is_outlier)]

        df.drop('precio_m2', axis=1, inplace=True)
        
        # Hay 70000 filas donde los metros totales son menores a los cubiertos. Esto es invalido, pero son muchos datos para descartar
        # Se asigna para estos casos, los metros cubiertos como totales.
        df.loc[df['metrostotales']<df['metroscubiertos'], 'metrostotales'] = df['metroscubiertos']

        df.loc[:,'extras'] = df['garages']+df['piscina']+df['usosmultiples']+df['gimnasio']
    
    else:
        
        df = fill_nans(df)
        
        df.loc[df['metrostotales']<df['metroscubiertos'], 'metrostotales'] = df['metroscubiertos']

        df.loc[:,'extras'] = df['garages']+df['piscina']+df['usosmultiples']+df['gimnasio']
        
        
    df = df.drop(['garages','piscina','usosmultiples','gimnasio'], axis=1)
    df.loc[:,'antiguedad'] = df['antiguedad'].transform(transformar_antiguedad).astype(int)
    df.drop(['direccion','idzona','lat','lng','titulo','descripcion'], axis=1, inplace=True)
    df.loc[:,'banos'] = df['banos'].astype(int)
    df.loc[:,'habitaciones'] = df['habitaciones'].astype(int)
    df.loc[:,'metroscubiertos'] = df['metroscubiertos'].astype(int)
    df.loc[:,'metrostotales'] = df['metrostotales'].astype(int)
    df.loc[:,'anio'] = df.fecha.dt.year
    df.drop('fecha', axis=1, inplace=True)
    
    return df