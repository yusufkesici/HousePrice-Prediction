import joblib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, VotingRegressor
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from helpers.eda import *
from helpers.data_prep import *
import missingno as msno
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.neighbors import LocalOutlierFactor, KNeighborsRegressor
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, f1_score, classification_report, roc_auc_score, recall_score, roc_curve,r2_score,mean_squared_error
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from xgboost import XGBClassifier
from catboost import CatBoostClassifier,CatBoostRegressor
from sklearn.linear_model import RidgeClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, train_test_split
from sklearn.metrics import confusion_matrix
import missingno as msno

# import warnings filter
from warnings import simplefilter
import warnings
# ignore all future warnings

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=Warning)


pd.set_option("display.max_columns", 500)
pd.set_option("display.max_rows", 200)

# Veriyi okuma
df = pd.read_csv("HousePricePrediction/train.csv")

# Genel bekış

check_df(df)

# kategorik ve numeric değişkenlerin getirilmesi

def HousePrice_dataPred(dff3):
    def grab_col_names2(dataframe, cat_th=10, car_th=20):
        """

        Veri setindeki kategorik, numerik ve kategorik fakat kardinal değişkenlerin isimlerini verir.
        Not: Kategorik değişkenlerin içerisine numerik görünümlü kategorik değişkenler de dahildir.

        Parameters
        ------
            dataframe: dataframe
                    Değişken isimleri alınmak istenilen dataframe
            cat_th: int, optional
                    numerik fakat kategorik olan değişkenler için sınıf eşik değeri
            car_th: int, optinal
                    kategorik fakat kardinal değişkenler için sınıf eşik değeri

        Returns
        ------
            cat_cols: list
                    Kategorik değişken listesi
            num_cols: list
                    Numerik değişken listesi
            cat_but_car: list
                    Kategorik görünümlü kardinal değişken listesi

        Examples
        ------
            import seaborn as sns
            df = sns.load_dataset("iris")
            print(grab_col_names(df))


        Notes
        ------
            cat_cols + num_cols + cat_but_car = toplam değişken sayısı
            num_but_cat cat_cols'un içerisinde.
            Return olan 3 liste toplamı toplam değişken sayısına eşittir: cat_cols + num_cols + cat_but_car = değişken sayısı

        """

        # cat_cols, cat_but_car
        cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
        num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                       dataframe[col].dtypes != "O"]
        cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                       dataframe[col].dtypes == "O"]
        cat_cols = cat_cols + num_but_cat
        cat_cols = [col for col in cat_cols if col not in cat_but_car]

        # num_cols
        num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
        num_cols = [col for col in num_cols if col not in num_but_cat]


        return cat_cols, num_cols, cat_but_car

    def missing_alot_col(dataframe):
        na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
        n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
        ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
        missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

        indeces = missing_df[missing_df["ratio"] > 40].index

        return indeces

    def missing_values_table2(dataframe, na_name=False):
        na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
        n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
        ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
        missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

        if na_name:
            return na_columns

    delete_cols = missing_alot_col(dff3)

    dff3.drop(delete_cols, axis=1, inplace=True)

    garage_na_indeces = dff3[dff3["GarageType"].isnull()].index


    dff3.drop(garage_na_indeces, inplace=True)

    Bsmt_na_indeces = dff3[dff3["BsmtExposure"].isnull()].index
    dff3.drop(Bsmt_na_indeces, inplace=True)
    Bsmt2_na_indeces = dff3[dff3["BsmtFinType2"].isnull()].index

    dff3.drop(Bsmt2_na_indeces, inplace=True)

    dff3["Electrical"].fillna(dff3["Electrical"].mode()[0], inplace=True)

    na_columns = missing_values_table2(dff3, na_name=True)

    for col in na_columns:
        if dff3[col].dtypes == "O":
            dff3[col].fillna(dff3[col].mode()[0], inplace=True)
        else:
            dff3[col].fillna(dff3[col].mean(), inplace=True)

    drop_cols = high_correlated_cols(dataframe=dff3, corr_th=0.80)

    dff3.drop(drop_cols, axis=1, inplace=True)

    # Rare encoding

    remove_cols = ["Street", "Utilities", "Condition2", "Heating", "PoolArea", "RoofMatl"]

    dff3.drop(remove_cols, axis=1, inplace=True)

    dff3 = rare_encoder(df, 0.05)

    cat_cols, num_cols, cat_but_car = grab_col_names2(dff3, car_th=26)

    dff3 = one_hot_encoder(dff3, cat_cols, drop_first=True)

    scaler = RobustScaler()

    num_cols = [col for col in num_cols if col not in ["Id", "SalePrice"]]
    dff3[num_cols] = scaler.fit_transform(dff3[num_cols])

    y = dff3["SalePrice"]
    X = dff3.drop(["Id", "SalePrice"], axis=1)

    return X, y

dff3 = pd.read_csv("datasets/Datas_HousePrice/train.csv")

X, y = HousePrice_dataPred(dff3)




cat_cols, num_cols, cat_but_car = grab_col_names(df, car_th=26)

num_cols = [col for col in num_cols if col not in ["Id", "SalePrice"]]

# Kategorik değişken analizi
for col in cat_cols:
    cat_summary(df, col, True)

# Sayısal değişken analizi
for col in num_cols:
    num_summary(df, col, True)

# Kategorik değişkenlerin target değişken ile analizi
for col in cat_cols:
    target_summary_with_cat(df, "SalePrice", col)

# Aykırı gözlem analizi

for col in num_cols:
    print(col, check_outlier(df, col))
    sns.boxplot(df[col])
    plt.show(block=True)

for col in num_cols:
    replace_with_thresholds(df, col)

# Eksik gözlem analizi

na_columns = missing_values_table(df, na_name=True)
missing_vs_target(df, "SalePrice", na_columns)

msno.bar(df)
msno.matrix(df)


# Eksik veri oranı %40 tan fazla ise o değişkenleri return eder.
def missing_alot_col(dataframe):
    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]
    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)
    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)
    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    indeces = missing_df[missing_df["ratio"] > 40].index

    return indeces


delete_cols = missing_alot_col(df)

df.drop(delete_cols, axis=1, inplace=True)

# garajı olmadığı için null yazılmış gözlemlerin silinerek çözülmesi.
garage = [col for col in df.columns if "Garage" in col]

for col in garage:
    print(col, df[col].dtypes)

garage_na_indeces = df[df["GarageType"].isnull()].index
df["GarageType"].unique()

df.drop(garage_na_indeces, inplace=True)

Bsmt_na_indeces = df[df["BsmtExposure"].isnull()].index
Bsmt2_na_indeces = df[df["BsmtFinType2"].isnull()].index
df["BsmtExposure"].unique()

df.drop(Bsmt_na_indeces, inplace=True)
df.drop(Bsmt2_na_indeces, inplace=True)

df["Electrical"].fillna(df["Electrical"].mode()[0], inplace=True)

for col in na_columns:
    print(col, df[col].dtypes)

for col in na_columns:
    if df[col].dtypes == "O":
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].mean(), inplace=True)


# Korelasyon analizi

corr = df.corr()
sns.heatmap(corr,annot=True)
plt.show()

drop_cols = high_correlated_cols(df,True,0.80)

df.drop(drop_cols, axis=1, inplace=True)




# Rare encoding
cat_cols, num_cols, cat_but_car = grab_col_names(df, car_th=26)



rare_analyser(df,"SalePrice",cat_cols)

remove_cols = ["Street","Utilities","Condition2","Heating","PoolArea","RoofMatl"]

df.drop(remove_cols,axis=1,inplace=True)

df = rare_encoder(df,0.05)

rare_analyser(df,"SalePrice",cat_cols)

dff = df.copy()



# Encoding & Scaling

cat_cols, num_cols, cat_but_car = grab_col_names(df, car_th=26)

df = one_hot_encoder(df, cat_cols, drop_first=True)


scaler = RobustScaler()

num_cols = [col for col in num_cols if col not in ["Id", "SalePrice"]]
df[num_cols] = scaler.fit_transform(df[num_cols])

df.head()

df.shape
# Base Model

y = df["SalePrice"]
X = df.drop(["Id", "SalePrice"], axis=1)

lgbm = LGBMRegressor()

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.25,random_state=2)

lgbm = LGBMRegressor().fit(X_train,y_train)

y_pred = lgbm.predict(X_test)

np.sqrt(mean_squared_error(y_test,y_pred))

y_test.mean()


def base_models(X, y):
    print("Base Models....")
    regressors = [('LR', LinearRegression()),
                   ('KNN', KNeighborsRegressor()),
                   ("CART", DecisionTreeRegressor()),
                   ("RF", RandomForestRegressor()),
                   ('Adaboost', AdaBoostRegressor()),
                   ('GBM', GradientBoostingRegressor()),
                   ('XGBoost', XGBRegressor()),
                   ('LightGBM', LGBMRegressor()),
                   # ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, regressor in regressors:
        cv_results = cross_validate(regressor, X, y, cv=3, scoring="neg_mean_squared_error")
        print(f"RMSE: {round(np.sqrt(-cv_results['test_score'].mean()), 4)} ({name}) ")


base_models(X, y)


######################################################
# 4. Automated Hyperparameter Optimization
######################################################

knn_params = {"n_neighbors": range(2, 50)}

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500]}


regresors = [('KNN', KNeighborsRegressor(), knn_params),
               ("CART", DecisionTreeRegressor(), cart_params),
               ("RF", RandomForestRegressor(), rf_params),
               ('XGBoost', XGBRegressor(), xgboost_params),
               ('LightGBM', LGBMRegressor(), lightgbm_params)]


def hyperparameter_optimization(X, y, cv=3,):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, regressor, params in regresors:
        print(f"########## {name} ##########")
        cv_results = cross_validate(regressor, X, y, cv=3, scoring="neg_mean_squared_error")
        print(f"RMSE: {round(np.sqrt(-cv_results['test_score'].mean()), 4)} ({name}) ")

        gs_best = GridSearchCV(regressor, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = regressor.set_params(**gs_best.best_params_)

        cv_results = cross_validate(regressor, X, y, cv=3, scoring="neg_mean_squared_error")
        print(f"RMSE: {round(np.sqrt(-cv_results['test_score'].mean()), 4)} ({name}) ")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models


best_models = hyperparameter_optimization(X,y)


######################################################
# 5. Stacking & Ensemble Learning
######################################################

def voting_classifier(best_models, X, y):
    print("Voting Regressor...")

    voting_clf = VotingRegressor(estimators=[('LightGBM', best_models["LightGBM"]),
                                              ('RF', best_models["RF"]),
                                              ('XGBoost', best_models["XGBoost"])]).fit(X, y)

    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring="neg_mean_squared_error")
    print(f"RMSE: {round(np.sqrt(-cv_results['test_score'].mean()), 4)}")
    return voting_clf

voting_clf = voting_classifier(best_models, X, y)


######################################################
# 6. Prediction for a New Observation
######################################################
import joblib
X.columns
random_user = X.sample(1, random_state=45)

voting_clf.predict(random_user)

joblib.dump(voting_clf, "voting_clf2.pkl")

new_model = joblib.load("voting_clf2.pkl")
new_model.predict(random_user)