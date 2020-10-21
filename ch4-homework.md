# 数据科学导论第二次作业

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as matplot
import seaborn as sns
import scipy.stats as st
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
%matplotlib inline

df = pd.read_csv('house_train.csv', index_col=0)#读入数据集，并将第一列设为目录

df.isnull().any()#检查缺失值
```


    distirct      False
    built_date    False
    green_rate     True
    area          False
    floor         False
    oriented      False
    traffic       False
    shockproof    False
    school        False
    crime_rate     True
    pm25          False
    price         False
    dtype: bool

第一块代码，调用了pandas、numpy、sklearn等库，并使用pandas的read_csv函数读入数据集，并检测了缺失值。


```python
for i in df.columns:
    if df[i].isnull().any():
        ave=df[i].mean()#计算该行均值
        df[i].fillna(ave,inplace=True)
        #用均值填充缺失值
df.isnull().any()#检查缺失值
```


    distirct      False
    built_date    False
    green_rate    False
    area          False
    floor         False
    oriented      False
    traffic       False
    shockproof    False
    school        False
    crime_rate    False
    pm25          False
    price         False
    dtype: bool

对数据集中的缺失值采用了均值填充的策略，填充后重新检测，发现缺失值已经没有了。


```python
need=['green_rate','area','traffic','shockproof','school','crime_rate','pm25']#需要进行异常值检测的列目录
df_c=df
for i in need:
    mmean=df[i].mean()
    sstd =df[i].std()
    print('%s属性中均值为：%.3f，标准差为：%.3f' % (i,mmean,sstd))
    error = df[i][np.abs(df[i] - mmean) > 3*sstd]
    print(error)
    df_c[i] = df[i][np.abs(df[i] - mmean) <= 3*sstd]#df_c为剔除异常值之后的数据
    print('异常值共%i条' % len(error))
```

    green_rate属性中均值为：52.383，标准差为：7.057
    id
    419     75.0
    764     29.0
    875     31.0
    943     31.0
    1316    28.0
    1664    75.0
    1838    30.0
    1951    83.0
    Name: green_rate, dtype: float64
    异常值共8条
    area属性中均值为：81.536，标准差为：6.238
    id
    236     104
    505     110
    1053    130
    1365    140
    1790    132
    Name: area, dtype: int64
    异常值共5条
    traffic属性中均值为：64.779，标准差为：13.902
    Series([], Name: traffic, dtype: int64)
    异常值共0条
    shockproof属性中均值为：57.931，标准差为：15.169
    id
    600     12
    1917    12
    Name: shockproof, dtype: int64
    异常值共2条
    school属性中均值为：2.356，标准差为：0.753
    Series([], Name: school, dtype: int64)
    异常值共0条
    crime_rate属性中均值为：5.788，标准差为：1.034
    id
    452     2.5
    995     2.6
    1055    2.6
    Name: crime_rate, dtype: float64
    异常值共3条
    pm25属性中均值为：63.024，标准差为：10.447
    id
    90      28
    323     30
    452     31
    729     29
    823     28
    995     29
    1030    30
    1055    31
    1223    30
    1288    31
    1615    30
    1893    29
    1909    30
    Name: pm25, dtype: int64
    异常值共13条

使用3σ原则进行异常值检测，检测后输出异常值，并删除。

```python
print(df_c.corr())
```

                distirct  green_rate      area  oriented   traffic  shockproof  \
    distirct    1.000000    0.069884 -0.034832 -0.008649 -0.019516   -0.045890   
    green_rate  0.069884    1.000000 -0.018609  0.047626  0.016044   -0.253599   
    area       -0.034832   -0.018609  1.000000  0.003501  0.307529    0.274374   
    oriented   -0.008649    0.047626  0.003501  1.000000  0.033021   -0.073630   
    traffic    -0.019516    0.016044  0.307529  0.033021  1.000000    0.094220   
    shockproof -0.045890   -0.253599  0.274374 -0.073630  0.094220    1.000000   
    school     -0.035775   -0.403446  0.197487 -0.087207  0.270105    0.458060   
    crime_rate -0.062386   -0.110959  0.408875 -0.064259  0.639726    0.345804   
    pm25       -0.019249   -0.200332  0.460155 -0.023011  0.766824    0.229675   
    price      -0.047704    0.053873  0.524945 -0.033219  0.339307    0.284843   
    
                  school  crime_rate      pm25     price  
    distirct   -0.035775   -0.062386 -0.019249 -0.047704  
    green_rate -0.403446   -0.110959 -0.200332  0.053873  
    area        0.197487    0.408875  0.460155  0.524945  
    oriented   -0.087207   -0.064259 -0.023011 -0.033219  
    traffic     0.270105    0.639726  0.766824  0.339307  
    shockproof  0.458060    0.345804  0.229675  0.284843  
    school      1.000000    0.225368  0.355978  0.225293  
    crime_rate  0.225368    1.000000  0.784214  0.406902  
    pm25        0.355978    0.784214  1.000000  0.402569  
    price       0.225293    0.406902  0.402569  1.000000  


由结果可以看出，与价格相关度最高的三个特征分别为房屋面积、犯罪率和pm2.5。房屋面积相关度高很好理解，房屋使用面积大，自然价格高。但是第二位和第三位的犯罪率和pm2.5需要进行一些合理的推理。
观察这两个特征向量，发现他们都与交通便利程度有很高的相关性。故可推测，交通便利的地方，人流大，人员组成成分复杂，犯罪率高也不是不能理解；与此同时，交通便利的地方交通工具多，产生的pm2.5自然也就多。所以这两个数据与其说是原因，不如说是交通便利、房价高导致的结果。


```python
pmean=df_c['price'].mean()
pstd=df_c['price'].std()
df_c['price']=(df_c['price']-pmean)/pstd #z-score标准化
print(df_c['price'])
```

    id
    1       0.552625
    2      -0.799097
    3      -1.124512
    4       1.553901
    5      -0.310975
              ...   
    1996   -0.010592
    1997    3.656580
    1998   -0.949288
    1999   -0.273427
    2000   -1.149543
    Name: price, Length: 2000, dtype: float64

使用z-score标准化，将price的数据减去平均值再除以标准差，得到的数据作为标准化后的新price数据。

```python
data = df_c.price
data_re = data.values.reshape((data.index.size, 1))
k = 6 #设置离散之后的数据段为10
k_model = KMeans(n_clusters = k, n_jobs = 4)
result = k_model.fit_predict(data_re)
df_c['price_discretized3'] = result
df_c.groupby('price_discretized3').price_discretized3.count()
```

    price_discretized3
    0    563
    1    279
    2     80
    3    149
    4    543
    5    386
    Name: price_discretized3, dtype: int64

使用K-Means算法分类，对标准化后的price数据进行离散化，将其分为了6类。

