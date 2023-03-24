import re

import pandas as pd

if __name__ == '__main__':

    dff = pd.read_csv('../../data/answer/train_set.csv')
    df = pd.read_csv('../../data/answer/test_set.csv')

    result1 = df["thread"].str.contains("elmlang")
    result2 = df["thread"].str.contains("pythondev")
    result3 = df["thread"].str.contains("irc_angularjs")
    result4 = df["thread"].str.contains("irc_opengl")
    result5 = df["thread"].str.contains("irc_c\+\+-general")
    result6 = dff["thread"].str.contains("elmlang")
    result7 = dff["thread"].str.contains("clojurians")
    result8 = dff["thread"].str.contains("pythondev")
    result6 = dff[result6]
    result7 = dff[result7]
    result8 = dff[result8]
   #print(result6)
    #print(result7)
    #print(result8)
    #result6.to_csv("F:/ChatEO/Chhh/data/answer/elmlangB_2017/train.csv")
    #result7.to_csv("F:/ChatEO/Chhh/data/answer/clojurians/train.csv")
    #result8.to_csv("F:/ChatEO/Chhh/data/answer/pythondev/train.csv")
    result = df["thread"].str.contains("clojurians")
    result = df[result]
    result1 = df[result1]
    result2 = df[result2]
    result3 = df[result3]
    result4 = df[result4]
    result5 = df[result5]
    #result.to_csv("F:/ChatEO/Chhh/data/answer/clojurians/test.csv")
   # result1.to_csv("F:/ChatEO/Chhh/data/answer/elmlangB_2017/test.csv")
   # result2.to_csv("F:/ChatEO/Chhh/data/answer/pythondev/test.csv")
   # result3.to_csv("F:/ChatEO/Chhh/data/answer/irc_angularjs/test.csv")
   # result4.to_csv("F:/ChatEO/Chhh/data/answer/irc_opengl/test.csv")
   # result5.to_csv("F:/ChatEO/Chhh/data/answer/irc_c++-general/test.csv")
    print(result)
    print(result1)
    print(result2)
    print(result3)
    print(result4)
    print(result5)
