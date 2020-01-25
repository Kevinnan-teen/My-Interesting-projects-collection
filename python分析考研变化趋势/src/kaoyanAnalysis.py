import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import folium
import webbrowser
from folium.plugins import HeatMap

import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False


def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.text(rect.get_x()+rect.get_width()/2.- 0.2, 1.03*height, '%s' % int(height))


def totalkaoyanChange(year, totalnumber, admitratio, registergrowthrate, admitnumber):
    """
        全国考研总体趋势
    """
    #总人数变化图
    ##################################
    plt.figure(1,(12,10))
    plt.title( "全国考研总人数变化柱状图" )
    plt.xlabel( '年份（年）' )
    plt.ylabel( '人数（万人）' )
    # plt.xlim( 2000, 2018 )
    autolabel(plt.bar( range(len(year)), totalnumber, tick_label=year))
    plt.show()
    ###################################
    
    # 各因素综合走势图
    ###################################
    plt.figure(2,(12, 8))
    # plt.xlim( 2000, 2018 )
    # 总人数
    normalizedtotalnumber = ( totalnumber - min(totalnumber) ) / ( max(totalnumber) - min( totalnumber ) )
    plt.plot( year, normalizedtotalnumber, label='总人数', linestyle=':', marker='o' )
    # 报录比
    normalizedadmitratio = ( admitratio - min(admitratio) ) / ( max(admitratio)-min(admitratio) )
    plt.plot( year, normalizedadmitratio, label='报录比', linestyle=':', marker='*')
    # 报名增长率
    '''normalizedregistergrowthrate = ( registergrowthrate - min(registergrowthrate) ) / ( max(registergrowthrate) - min(registergrowthrate) )
    plt.plot( year, normalizedregistergrowthrate, label='报名增长率', linestyle=':', marker='>' )'''
    # 录取人数
    normalizedadmitnumber = ( admitnumber - min(admitnumber) ) / ( max(admitnumber) - min(admitnumber) )
    plt.plot( year,  normalizedadmitnumber, label='录取人数', linestyle=':', marker='p')
    plt.title( "全国考研总体趋势图" )
    plt.xlabel( "年份" )
    plt.legend(loc=1)
    plt.show()
    ###################################

    
    ###################################
    # plt.figure(3, (12,))
    ###################################


def provinceNumberHeadmap(year):
    """
        2016-2020部分省市考研人数热力图
    """
    path3 = '../dataSet/linkCityInfo.csv'
    file3 = pd.read_csv( path3 )
    dfFile3 = pd.DataFrame( file3 )
    print( dfFile3 )

    lat = np.array( dfFile3["lat"] )
    lon = np.array( dfFile3["lon"] )
    yearstr = str(year) + '年'
    singlenumber = np.array( dfFile3[yearstr] / 1000 )      # 防止数据过大
    data = [[lat[i],lon[i],singlenumber[i]] for i in range(len(lat))]#将数据制作成[lats,lons,weights]的形式
    map_osm = folium.Map(location=[35,120],zoom_start=5)    #绘制Map，开始缩放程度是5倍
    HeatMap(data).add_to(map_osm)                       # 将热力图添加到前面建立的map里

    file_path = '../results/' + str(year) + '-kaoyan.html'
    map_osm.save(file_path)     # 保存为html文件
    webbrowser.open(file_path)  # 默认浏览器打开


def comparePostgraduateAndGraduate():
    """
        比较历年以来毕业和考研的学生数量
    """
    path = '../dataSet/jiuyeAndkaoyan.csv'
    file3 = pd.read_csv( path )
    dfFile3 = pd.DataFrame( file3 )
    # print( dfFile3 )
    graduatetotalnumber = np.array(dfFile3.iloc[0])[1:].tolist()[::-1]
    postgraduatetotalnumber = np.array(dfFile3.iloc[1])[1:].tolist()[::-1]

    ####################################
    plt.figure(3, (12, 8))

    use1 = graduatetotalnumber[8:]          # 2008-2018年毕业和考研学生数据
    use2 = postgraduatetotalnumber[8:]      

    x = list( range(len(use1)) )
    total_width, n = 0.8, 2
    width = total_width / 2
    autolabel(plt.bar( x, use1, width=width, label='graduate', fc='#000080' ))      # 海军蓝
    for i in range(len(x)):
        x[i] += width
    
    y = 2008
    year = []
    while y <=2018:
        year.append(y)
        y += 1
    year = np.linspace(2008, 2018, 11).tolist()
    autolabel(plt.bar( x, use2, width=width, label='postgraduate',tick_label=year, fc='#9370DB' ))  # 浅紫色
    plt.title( "2008-2018年毕业和考研学生对比柱形图" )
    plt.xlabel( "年份（年）" )
    plt.ylabel( "人数（万人）" )
    plt.legend()
    plt.show()
    ####################################


    plt.figure(4, (12, 8))
    postgraduateAndgraduate_rate = np.array(use2) / np.array(use1)      # 考研和毕业比例
    print( postgraduateAndgraduate_rate )
    plt.plot( year, postgraduateAndgraduate_rate, label='考研和毕业比例', linestyle=':', marker='*')
    plt.ylim(0,1)
    plt.xlabel("年份")
    plt.ylabel("比例")
    plt.title("考研毕业比例曲线")
    plt.legend(loc=1)
    plt.show()
    
    


    
   




    


if __name__ == '__main__':
    path1 = '../dataSet/kaoyan01.csv'    #2000-2018年考研人数及变化趋势数据
    path2 = '../dataSet/kaoyan02.csv'    #2016-2020年部分省市考研人数 
    file1 = pd.read_csv( path1 )
    dfFile1 = pd.DataFrame( file1 )
    file2 = pd.read_csv( path2 )
    dfFile2 = pd.DataFrame( file2 )

   
    # 年份
    year1 = np.array( dfFile1['考硕年份'].values ).tolist()
    year1 = np.array( year1[::-1] )

    # 总人数
    totalnumber = np.array( dfFile1['报名人数(万人)'].values ).tolist()
    totalnumber = np.array( totalnumber[::-1] )

    # 报录比
    admitratio = np.array( dfFile1['考录比例'].values ).tolist()
    admitratio = np.array( admitratio[::-1] )

    # 报名增长率
    registergrowthrate = np.array( dfFile1['报名增长率'].values ).tolist()
    registergrowthrate = np.array( registergrowthrate[::-1] )
    print( registergrowthrate )

    # 录取人数
    admitnumber = np.array( dfFile1['录取人数（万）'].values ).tolist()
    admitnumber = np.array( admitnumber[::-1] )

    totalkaoyanChange( year1, totalnumber, admitratio, registergrowthrate, admitnumber)

    
    # provinceNumberHeadmap(2020)

    comparePostgraduateAndGraduate()
    





















    





















