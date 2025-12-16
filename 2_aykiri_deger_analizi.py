#############################
# cok değişikenli aykırı değer analizi : local outlier factor
#############################

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
from datetime import date
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler


#TEMEL BİLEŞEN ANALİZİ YONTEMİ İNCELE

# tek basına aykırı olmayacak degerler birlikte ele alındıgında aykırılık yaratıyorsa olur
# yas ve evlilik için 17 ve 3 defa evlenmek normaldir ancak 17 yasında 3 defa evlenmek outlier dır

# lof yontemi (local outlier factor)
# cok değişkenli bir aykırı deger belirleme yontemidir
# gozlemleri bulunduları noktada yogunluk tabanlı skorlayarak buna gore aykırı deger tanımı yapabilmemizi saglar.
# ilgili noktanın etrafındaki komsuluklar - bu komsuluklara gore uzaklık skoru hesaplama imkanı saglar lof kavramı
# komsularının cevresindeki yogunluktan daha dusuk   olan noktayı saptar
# inlier


def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range

    return low_limit, up_limit


def check_outlier(dataframe , col_name):#aykırı değer var mı yok mu
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False


def grab_col_names(dataframe, cat_th = 10, car_th = 20 ):#cat treshld ve kardinal threshold belirlenmiş

    #cat_cols , cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    #ilgili değişkendekii eşsiz değer sayısına bak eger bu
    # belirlediğin tresholddan kucukse ve değişkenin tipi de obje değilse numerik ama kategorik

    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    #eşsiz sınıf sayısı 20 den butyukse ve tipi de kategorikse kategorik ama kardinal
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat#at cols listesi guncellendi
    cat_cols = [col for col in cat_cols if col not in cat_but_car]#cat cols içindeki kardinalleri cıkar

    #num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"] #tipi objectten farklı olanları getirdik int ya da float olanlar geldi
    num_cols = [col for col in num_cols if col not in num_but_cat] #numerik ama kategorik olanları da içinden cıkarıyoruz içinde sadece numerik kolonlar kalıyor

    print(f"observation : {dataframe.shape[0]}")
    print(f"variables : {dataframe.shape[1]}")
    print(f"cat_cols : {len(cat_cols)}")
    print(f"num_cols : {len(num_cols)}")
    print(f"cat_but_car : {len(cat_but_car)}")
    print(f"num_but_cat : {len(num_but_cat)}")
    return cat_cols, num_cols, cat_but_car #gercek numerik gercek kategorik ve gercek kardinaller kaldı


def grab_outliers(dataframe, col_name, index=False):#aykırı degerlere bakıyım bunlar kimmiş
    low, up = outlier_thresholds(dataframe, col_name)
    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        #alt limitten kucuk veya ust limitten buyuk olanlar varsa bunlar getir.
        # eger bu getirdiklerinin sayısı 10dan buyukse bastakileri getir inceleyelim
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())
    else:
        #değilse hepsini getir bakalım
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index
    #index true ise indexleri return et


def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers


def replace_with_threshold(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit) , variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit) , variable] = up_limit



df = sns.load_dataset('diamonds')
df = df.select_dtypes(include=['float64', 'int64'])
df = df.dropna()
df.head()

for col in df.columns:
    print(col, check_outlier(df, col))

low , up = outlier_thresholds(df, "carat")

df[((df["carat"] < low) | (df["carat"] > up))].shape

low , up = outlier_thresholds(df, "depth")

df[((df["depth"] < low) | (df["depth"] > up))].shape

# aykırı degerler cok fazla
# outlier threshold degeri literaturde 25 76 idi ama boye yaparsak veri setini buyuk olcude kaybederiz
# eger onları outliera esitlersek(dodursaydık) veri setinde kirlilik gurultu yaratırız
# agac yontemleri kullanıyorsak bu sebeple dokunmamayı tercih etmeliyiz
# ya da ucundan en aykırı olanları alırız



clf = LocalOutlierFactor(n_neighbors=20)#komsuluk sayısı
clf.fit_predict(df)# lof skorlarını getirecek

df_scores = clf.negative_outlier_factor_ #skorları tutuyoruz
df_scores[0:5]
# df_scores = -df_scores
#degerlerin 1 e yakın olması inlier olma degerini gosteriyor
# kurdugumuz duzene gore -1 e yakın olması inlier oldugunu gosterecek
# ornegin -10 a gittikçe degerlerin daha aykırı olma durumu artacak

np.sort(df_scores)[0:5]

#threshold belirlemede elbow dirsek yontemi
#grafik çizdirip ordaki eğimin en cok oldugu kısımdan threshold belirleyebiliriz

scores = pd.DataFrame(np.sort(df_scores))
scores.plot(stacked = True, xlim = [0,50], style='.-')
plt.show()

th = np.sort(df_scores)[3]

df[df_scores < th]#daha kucuk değerler aykırı deger oluyor -6 gibi
# tek değişkenle incelendiğinde binlerce aykırı deger varken cok değişkenle incelendiğinde 3 taneye düştü
#nedenn
# değişkenlerin kombinasyonları burada etkili olmuştur
# mesela bir 5 karat elmasın 20 k olması verinin ortalama değer kriterlerinde olsun
# bu durumda 5 karat bir elmasın 80k fiyata satılması aykırı bir değerdir
# (burdaki ornekler dogrusal değildri birden fazla değişken goz onunde bulundurulmalı onların kombinasyonları esas alınmalı)

df.describe([0.01, 0.05, 0.75, 0.90, 0.99]).T

df[df_scores < th].index#ændex bilgileri yakalandı
df[df_scores < th].drop(axis=0, labels=df[df_scores < th].index)#outlierslar silindi

#baskılama nasıl yapabiliriz
#gozlem sayısı fazlaysa buraya veriler eklemek problemlere yol acabilir
#agac yontemleriyse dokunmamak en iyisi ya da 5 e 95 threshold
#dogrusal yontemlerde sıkıntı devam ediyor az sayıdaysa silinebilir
#doldurmak yerine baskılamak tercih edilebilir







