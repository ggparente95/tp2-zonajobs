import pandas as pd


def is_outlier(group):
    ''' Para cada grupo devuelve por cada valor si es o no outlier. '''
    Q1 = group.quantile(0.25)
    Q3 = group.quantile(0.75)
    IQR = Q3 - Q1
    precio_min = Q1 - 1.5 * IQR
    precio_max = Q3 + 1.5 * IQR
    return ~group.between(precio_min, precio_max)


def delete_invalid_registers(df):
    
    # Elimino propiedades que no tengan ni ciudad o provincia.
    df.dropna(subset=['ciudad','provincia'], inplace=True)
    
    # Elimino aquellas propiedades sin tipo. (Son pocas).
    df.dropna(subset=['tipodepropiedad'],inplace=True)
    
    return df


def fill_nans(df):
    # Completo los nans de metros totales, con lo que tengan en la columna de metros cubiertos.
    df['metrostotales'].fillna(df['metroscubiertos'], inplace=True)

    # El resto de las filas con metros cubiertos nulo, son terrenos. Por ende los completo con 0.
    df['metroscubiertos'].fillna(0, inplace=True)

    # Agrego a los que son nans, los tipos, ciudad y provincia mas comunes.
    df['tipodepropiedad'].fillna(df['tipodepropiedad'].value_counts().index[0], inplace=True)
    df['ciudad'].fillna(df['ciudad'].value_counts().index[0], inplace=True)
    df['provincia'].fillna(df['provincia'].value_counts().index[0], inplace=True)
    
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

    
def getNormalizedDataset(df, mode='train'):
    ''' Devuelve el dataset de propiedades, completando valores nulos, arreglando errores y removiendo outliers. '''

    if mode=='train':    
        
        df = delete_invalid_registers(df)
        
        df = fill_nans(df)

        df['precio_m2'] = df['precio']/df['metrostotales']

        # Limpio los outliers
        df = df[~df.groupby('tipodepropiedad')['precio_m2'].apply(is_outlier)]

        df.drop('precio_m2', axis=1, inplace=True)
        
        # Hay 70000 filas donde los metros totales son menores a los cubiertos. Esto es invalido, pero son muchos datos para descartar
        # Se asigna para estos casos, los metros cubiertos como totales.
        df.loc[df['metrostotales']<df['metroscubiertos'], 'metrostotales'] = df['metroscubiertos']

        df['extras'] = df['garages']+df['piscina']+df['usosmultiples']+df['gimnasio']
    
    else:
        
        df = fill_nans(df)
        
        df.loc[df['metrostotales']<df['metroscubiertos'], 'metrostotales'] = df['metroscubiertos']

        df['extras'] = df['garages']+df['piscina']+df['usosmultiples']+df['gimnasio']

        
    df.drop(['direccion','idzona','lat','lng','titulo','descripcion'], axis=1, inplace=True)
    df['garages'] = df['garages'].astype(int)
    df['antiguedad'] = df['antiguedad'].astype(int)
    df['banos'] = df['banos'].astype(int)
    df['habitaciones'] = df['habitaciones'].astype(int)
    df['metroscubiertos'] = df['metroscubiertos'].astype(int)
    df['metrostotales'] = df['metrostotales'].astype(int)
    df['dia'] = df.fecha.dt.day
    df['mes'] = df.fecha.dt.month
    df['anio'] = df.fecha.dt.year
    df.drop('fecha', axis=1, inplace=True)
    
    return df

