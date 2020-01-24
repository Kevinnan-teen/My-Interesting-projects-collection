#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import shutil
import time
import pandas as pd


def move(numbers, origin, new):
    if not os.path.exists(origin):
        os.mkdir(origin)
    
    if not os.path.exists(new):
        os.mkdir(new)
    
    f = os.listdir( origin )
    print( len(f) )
    
    Class = pd.Series( [0,0,0,0,0,0,0,0,0,0]  )
    print(Class)
    
    for ff in f:
        for i in range(10):
            if ff.split( '_' )[0] == str(i):
                if Class[i] >= numbers:
                    break
                else:
                    Class[i] += 1
                    shutil.move( origin + ff, new + ff )
                    break


if __name__ == '__main__':
    path1 = './test5/'
    path2 = './train6/'

    path3 = './test6/'
    path4 = './test6/'
    move(200, path1, path2)
    move(100, path1, path4)
                
            
    
            
