#BOXPLOT YONTEMİ

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

pd.set_option('display.max_columns', None)#butun sutunları goster
pd.set_option('display.max_rows', None)#tum satırları goster
pd.set_option('display.float_format', lambda x: '%.3f' % x)#Ondalık sayıları 3 basamakla göster
pd.set_option('display.width', 500)#Bir satırda yazdırılacak maksimum karakter genişliğini 500 olarak ayarla.


def load_application_train():#veri okuma işlemi için fonk
    data = pd.read_csv('3_aykiri_deger_yakalama/datasets/application_train.csv')
    return data

df = load_application_train()
df.head()


def load():
    data = pd.read_csv('3_aykiri_deger_yakalama/datasets/titanic.csv')
    return data

df = load()
df.head()

#############################
# outliers aykırı degerler
#############################

#############################
# aykırı degerleri yakalama
#############################

sns.boxplot(x=df["Age"])
plt.show()

#############################
# aykırı degerler nasıl yakalanır
# çeyrek degerleri hesaplayıp onun uzerinden ıqr hesabı yaparız
#############################

q1 = df["Age"].quantile(0.25)#bana yuzde 25 lik ceyregi getir
q3 = df["Age"].quantile(0.75)#3. ceyrek değer

iqr = q3 - q1

up = q3 + 1.5 * iqr
low = q1 - 1.5 * iqr

#alt sınırdan kucuk ya da ust sınırdan buyuk olanları getirelim
df[(df["Age"] < low) | (df["Age"] > up)]

df[(df["Age"] < low) | (df["Age"] > up)].index


#############################
# aykırı deger var mı yok mu
#############################
df[(df["Age"] < low) | (df["Age"] > up)].any(axis=None)

df[(df["Age"] < low)].any(axis=None)

#############################
# işlemleri fonksiyonlaştırma
#############################

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range

    return low_limit, up_limit

outlier_thresholds(df, "Age")
outlier_thresholds(df, "Fare")

low , up = outlier_thresholds(df, "Age")


df[(df["Fare"] < low) | (df["Fare"] > up)].head()
df[(df["Fare"] < low) | (df["Fare"] > up)].index()

def check_outlier(dataframe , col_name):#aykırı değer var mı yok mu
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False

check_outlier(df, "Age")
check_outlier(df, "Fare")


#############################
# grab_col_names veri setindeki sayısal kategorik degişken ayrımı
#kategorik gozuken ama kategorik olmayan
#saysıal gorunen ama sayısal olmayan değiişkenlerin tespiti
#############################

dff = load_application_train()
dff.head()
"""
veri setindeki kategorik , numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
not : kategorik değişkenlerin içerisine numerik gorunumlu kategorik değişkenler de dahildir

parametreler
-----------
dataframe : dataframe
    degişken isimleri alınmak istenen dataframe
cat_th : int , optional 
    numerik fakat kategorik olan değişkenler için sınıf eşik değeri
car_th : int , optional 
    kategorik fakat kardinal değişkenler için sınıf eşik değeri

returns 
---------
cat_cols : list
    kategorik değişken listesi
num_cols : list
    numerik değişken listesi

"""

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


cat_cols, num_cols, cat_but_car = grab_col_names(df)

num_cols = [col for col in num_cols if col not in "PassengerId"]

for col in num_cols:
    print(col, check_outlier(df, col))


cat_cols, num_cols, cat_but_car = grab_col_names(dff)

num_cols = [col for col in num_cols if col not in "SK_ID_CURR"]

for col in num_cols:
    print(col, check_outlier(dff, col))


#############################
# aykırı değerlerin kendiklerine erişmek
#############################

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


grab_outliers(df, "Age")
grab_outliers(df, "Age", True)
#birçok agac yontemi aykırı degerlere duyarsızdır eksik degerlere duyarsızdır

#############################
# aykırı değer problemini cozmek
#############################

#############################
# silme
#############################

low, up = outlier_thresholds(df, "Fare")
df.shape#veri setinde kac gozlem var

df[~((df["Fare"] < low) | (df["Fare"] > up))].shape #aykırı olmayanları getir

def remove_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]
    return df_without_outliers

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

for col in num_cols:
    new_df = remove_outlier(df, col)

df.shape[0] - new_df.shape[0] #kaç tane değişiklilk olduguna bakıyoruz

#############################
# baskılama yontemiyle (re-assignment wit thresholds)
#eşik değerin dısındaki degerler esik degerler ile değiştirilir bu sekilde veri kaybı onlenir
#############################

low, up = outlier_thresholds(df, "Fare")
df[((df["Fare"] < low) | (df["Fare"] > up))]["Fare"]

df.loc[((df["Fare"] < low) | (df["Fare"] > up), "Fare")]

df.loc[(df["Fare"] < up), "Fare"] = up #üst sınıra gore aykırı olan değerler up olarak değiştirildi

df.loc[(df["Fare"] < low), "Fare"] = low

def replace_with_threshold(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit) , variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit) , variable] = up_limit

df = load()
cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

df.shape

for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_threshold(df, col)

for col in num_cols:
    print(col, check_outlier(df, col))

#############################
# RECAP
#############################

df = load()#veri setini okuduk
outlier_thresholds(df, "Age")#aykırı değeri saptama işlemi yaptık
check_outlier(df, "Age")#outlier var mı yok mu diye sorduk
grab_outliers(df, "Age", index=True)#outlierları bize getir dedik

remove_outlier(df, "Age")#outlier ları silerek tedavi ettik
replace_with_threshold(df, "Age")#baskılama yontemiyle thresholdları değiştirdik
check_outlier(df, "Age")#tekrar gozlemledik ve outlierlardan kurtulduk











