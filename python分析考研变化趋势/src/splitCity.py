#-*- coding:utf-8 -*-
import pandas as pd
import numpy as np

if __name__ == '__main__':
    


    cityPath = '../dataSet/2015Cities-CHINA.csv'
    file2 = pd.read_csv( cityPath )
    dfFile2 = pd.DataFrame( file2 )
    # print( dfFile2['cities'].values )

    print(dfFile2)
    #cityFrame = pd.DataFrame(columns=['city', 'lon', 'lat'])

    #for city in cities:
        
    cities = "石家庄、太原、沈阳、哈尔滨、南京、南昌、济南、郑州、武汉、长沙、广州、南宁、成都、昆明、重庆、西安、兰州、呼和浩特"
    city3 = []
    cities = cities.split('、')

    dfsaveFile = pd.DataFrame( columns=['city', 'lon', 'lat'] )
    
    for city in cities:
        for i in range( len(dfFile2) ):
            if str(dfFile2.iloc[i].at['cities'][:-1]) == city:
                city3.append(city)
                dfsaveFile.loc[i] = dfFile2.iloc[i].at['cities'][:-1], dfFile2.iloc[i].at['lon'], dfFile2.iloc[i].at['lat']
                #print(city)
            #print(dfFile2.iloc[i].at['cities'][:-1])
            # print(city)
    print( len(city3) )
    print(dfsaveFile)
    dfsaveFile.to_csv( '../dataSet/cityLonLat.csv' )
    
