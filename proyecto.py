import streamlit as st
import os
import tarfile
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml2/master/'
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data():
    os.makedirs(HOUSING_PATH, exist_ok=True)
    tgz_path = os.path.join(HOUSING_PATH, "housing.tgz")
    urllib.request.urlretrieve(HOUSING_URL, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=HOUSING_PATH)
    housing_tgz.close()

def load_housing_data():
    csv_path = os.path.join(HOUSING_PATH, "housing.csv")
    return pd.read_csv(csv_path)

fetch_housing_data()
housing = load_housing_data()

st.title("Análisis de Datos de Vivienda")

if st.checkbox("Mostrar datos de vivienda"):
    st.write(housing.head())

def split_data(data):
    train_set, test_set = train_test_split(data, test_size=0.2, random_state=42)
    return train_set, test_set

train_set, test_set = split_data(housing)

housing["income_cat"] = pd.cut(housing["median_income"],
                                bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                                labels=[1, 2, 3, 4, 5])

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

housing = strat_train_set.copy()

if st.checkbox("Mostrar gráfico de distribución geográfica"):
    fig, ax = plt.subplots()
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
                 s=housing["population"] / 100, label="población",
                 c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True, ax=ax)
    plt.legend()
    st.pyplot(fig)

housing["rooms_per_household"] = housing["total_rooms"] / housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"] / housing["total_rooms"]
housing["population_per_household"] = housing["population"] / housing["households"]

housing_labels = strat_train_set["median_house_value"].copy()

num_attribs = list(housing.select_dtypes(include=[np.number]).columns)
num_attribs.remove("median_house_value")
cat_attribs = ["ocean_proximity"]

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler()),
])

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs),
])

housing_prepared = full_pipeline.fit_transform(housing.drop("median_house_value", axis=1))

models = {
    "Regresión Lineal": LinearRegression(),
    "Árbol de Decisión": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor()
}

model_choice = st.selectbox("Selecciona un modelo", list(models.keys()))
model = models[model_choice]
model.fit(housing_prepared, housing_labels)

st.subheader("Evaluación del Modelo")
scores = cross_val_score(model, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
rmse_scores = np.sqrt(-scores)

st.write("Scores:", rmse_scores)
st.write("Media:", rmse_scores.mean())
st.write("Desviación estándar:", rmse_scores.std())

st.subheader("Predicción de Precio de Vivienda")
user_input = {}
for col in num_attribs:
    user_input[col] = st.number_input(f"{col}", value=float(housing[col].median()))

user_input["ocean_proximity"] = st.selectbox("Proximidad al océano", housing["ocean_proximity"].unique())

user_df = pd.DataFrame([user_input])
user_prepared = full_pipeline.transform(user_df)

if st.button("Predecir Precio"):
    prediction = model.predict(user_prepared)
    st.write(f"Precio estimado de la vivienda: ${prediction[0]:,.2f}")
