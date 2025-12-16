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

#############################
# eksik değerlerin yakalanması
#############################

pd.set_option('display.max_columns', None)#butun sutunları goster
pd.set_option('display.max_rows', None)#tum satırları goster
pd.set_option('display.float_format', lambda x: '%.3f' % x)#Ondalık sayıları 3 basamakla göster
pd.set_option('display.width', 500)#Bir satırda yazdırılacak maksimum karakter genişliğini 500 olarak ayarla.


def load_application_train():#veri okuma işlemi için fonk
    data = pd.read_csv('3_aykiri_deger_yakalama/datasets/application_train.csv')
    return data

def load():
    data = pd.read_csv('3_aykiri_deger_yakalama/datasets/titanic.csv')
    return data


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

df = load()
df.head()

#butun veride eksik gozlem var mı yok mu sorgusu
df.isnull().values.any()#butun hucreleri gez eksiklik var mı diye bak. true false doner(isnul)
#bu true degerini values ta tutarız. herhangi bir eksikli varsa bunu getir deriz any( ) ile


#hangi değişkende kac tane eksiklik var
df.isnull().sum()

#değişkenlerdeki tam deger sayısı
df.notnull().sum()

#verideki butun eksik degerlerin toplamı
df.isnull().sum().sum() #kendisinde en az bir tane eksik hucre olan satır sayısı

#en az bir tane eksik degere sahip olan gozlem birimleri
df[df.isnull().any(axis=1)]

#tam olan gozlem birimleri
df[df.notnull().all(axis=1)]

#azalan sekilde sıralamak
df.isnull().sum().sort_values(ascending=False)

#veri setindeki oran
#her bir değişkendeki eksikliği ifade eeden bu frekansları veri setinın toplam gozlem sayısına bolecez ve sonra yuzde ile carpacaz
(df.isnull().sum() / df.shape[0] * 100).sort_values(ascending=False)

#sadece eksik değere sahip değişkenleri seçmek
#df nın sutunlarında gez eger o gezdiğin sutundaki topla eksik deger sayısı sıfırdan buyukse bunu sec na ya gonder
na_cols = [col for col in df.columns if df[col].isnull().sum() > 0]

#dataframe ve bir değişken alıyor bu değişken eksik degerlerin barındığı değişkenlerin isimlerini ver ya da verme
def missing_values_table(dataframe, na_name=False):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)#eksik deger sayısı
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)#eksik deger oranı
    missing_df = pd.concat([n_miss, np.round(ratio,2)], axis=1,keys=['n_miss', 'ratio'])#missing_df adında bir dataframede birleştirdik
    print(missing_df, end='\n')
    if na_name:
        return na_columns


missing_values_table(df, na_name=True)


#############################
# eksik değer problemini çözme
#############################

#ağac yontemleri kullanılıyorsa eksik değerler aykırı değerler gibi etkisi goz ardı edilebilir durumlardır
#agac yontemleri daha esnek ve dallara ayrılabilir oldukları için etkileri yoka yakındır goz ardı edilebilir
#iilgilenilen problem regresyon problemiyse ve bağımlı değişken de sayısal bir değişkense aykırılık olması durumunda sonuca gitme suresi uzayabilir
#dogrusal yontemlerde ve gradient descent temelli yontemlerde bu teknikler cok daha hassas iken agaca dayalı yontemlerde bunların etkisi cokkk daha dusuktur

missing_values_table(df)

#############################
# cozum 1 : hızlıca silmek
#############################

#bir satırda en az bir tane bile eksik değer varsa o satırı siliyoruz
df.dropna().shape #ama bu durumda gozlem sayısı cok azalacaktır


#############################
# cozum 2 : basit atama yontemleriyle doldurmak mod medyan ortalama ya da herhangi sabit değer
#############################

df["Age"].fillna(df["Age"].mean()).isnull().sum()
df["Age"].fillna(df["Age"].median()).isnull().sum()
df["Age"].fillna(0).isnull().sum()

df.apply(lambda x: x.fillna(x.mean()), axis=0)#axis=0 satırları 1 sutunları ifade eder

df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0).head()

#butun sayısal eksik değişkenleri ortalamasıyla dolduruyor
dff = df.apply(lambda x: x.fillna(x.mean()) if x.dtype != "O" else x, axis=0)#object tipinde olup olmadıgını kontrol ediyor ona gore islem yapıyor

dff.isnull().sum().sort_values(ascending=False)

df["Embarked"].fillna(df["Embarked"].mode()[0]).isnull().sum()#embarked değişkeninin modunu alıp string değişkenine erişmek

df["Embarked"].fillna("missing")#string bir ifadeyle doldurma

# eger bir kategorik değişken ve eşsiz deger sayısı 10 dan kucuk bir kategorik değişken ise bunun modunu bu değişkene ata değilse bırak
df.apply(lambda x: x.fillna(x.mode()[0]) if (x.dtype == "O" and len(x.unique()) <= 10) else x, axis=0).isnull().sum()


#############################
# kategorik değişken kırılımında değer atama
#############################

#veri seitnde var olan bazı kategorik değişkenleri kırılım olarak ele almak ve
#bu kırılımlar sonucunaki değerleri ilgili değerlere atamak

df.groupby("Sex")["Age"].mean()["female"]#cinsiyete gore ayır yas ortalamaarını al

df["Age"].mean()#direkt bir ortalama atamak yerine bunu cinsiyete ozel ortalama atamak olarak değiştirebiliriz

#cinsiyete gore veri setini groupby a al daha sonra yaş değişkenini seç sonra bu yaş değişkeninin ortalmasını alıp ilgili yerlere yaz
df["Age"].fillna(df.groupby("Sex")["Age"].transform("mean")).isnull().sum()

df.loc[(df["Age"].isnull()) & (df["Sex"] == "female"), "Age"] = df.groupby("Sex")["Age"].mean()["female"]
df.loc[(df["Age"].isnull()) & (df["Sex"] == "male"), "Age"] = df.groupby("Sex")["Age"].mean()["male"]

df.isnull().sum()

#############################
# çözüm 3 : tahmine dayalı atama le doldurma
#############################

#eksikliğe sahip olan değişkeni bagımlı değişken diğer değişkenleri bagımsız değişken olarak ele alıp modelleme işlemi gerçekleştirecez
#bu modelleme işlemine gore eksik olan deperleri tahmin etmeye calısacagız

#kategorik değişkenleri onehot encodera sokmamız lazım
#knn uzaklık temelli bir algoritma oldugundan dolayı değişkenleri standartlaştırmamız lazım

df = load()

cat_cols, num_cols, cat_but_car = grab_col_names(df)
num_cols = [col for col in num_cols if col not in "PassengerId"]

#label encodingi ve onehot encodingi aynı anda yapabilmek için getdummies kullanıyoruz
dff = pd.get_dummies(df[cat_cols + num_cols], drop_first=True)
#kategorik değişkenleri numerik bir sekilde ifade ettik

dff.head()

#değişkenlerin standartlaştırılması
scaler = MinMaxScaler()
dff = pd.DataFrame(scaler.fit_transform(dff) , columns=dff.columns)
dff.head()

#knn in uygulaması
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)#model nesnesini olusturduk komsu sayısını 5 yaptık
#noktanın cevresindeki en yakın 5 noktanın değerlerine bakarak bilinmeyen değeri tahmin eder
#bana arkadasını soyle sana kim oldugunu soyleyeyim der

dff = pd.DataFrame(imputer.fit_transform(dff), columns = dff.columns)
df.head()

dff = pd.DataFrame(scaler.inverse_transform(dff), columns=dff.columns)#yapılan standartlaştırma işlemi geri alınıt-yor ki atanan değerler asıl olarak gorunsun
#bosluklara değer atadım ama bunlar arasındaki kıyaslamayı nasıl yapıcam

df["age_imputed_knn"] = dff[["Age"]]
df.loc[df["Age"].isnull(), ["Age", "age_imputed_knn"]]#eksik değerlerin yerine atanmıs değerleri koyduk
df.loc[df["Age"].isnull()]
















