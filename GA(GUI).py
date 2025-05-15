import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import shap
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from deap import base, creator, tools, algorithms
import random
import warnings
import time

warnings.filterwarnings("ignore")

st.set_page_config(page_title="GA Feature Selection", layout="wide")
st.title("\U0001F9EA Genetic Algorithm for Feature Selection - Customer Churn")

# Sidebar - Settings
st.sidebar.header("Settings")
ngen = st.sidebar.slider("Generations", 5, 50, 10)
pop_size = st.sidebar.slider("Population Size", 10, 100, 20)
classifier_name = st.sidebar.selectbox("Classifier", ["Random Forest", "Logistic Regression", "SVM"])
runs = st.sidebar.slider("GA Trials", 1, 5, 1)

uploaded_file = st.file_uploader("\U0001F4C2 Upload dataset", type=["csv", "xlsx"])

# Load Data
@st.cache_data
def load_data(uploaded_file):
    if uploaded_file.name.endswith('csv'):
        return pd.read_csv(uploaded_file)
    else:
        return pd.read_excel(uploaded_file)

# Preprocess
def preprocess_data(df):
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']):
        df[col] = le.fit_transform(df[col])
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, list(X.columns)

# Classifier selector
def get_model(name):
    if name == "Random Forest":
        return RandomForestClassifier(random_state=42)
    elif name == "Logistic Regression":
        return LogisticRegression(max_iter=1000)
    else:
        return SVC(probability=True)

# Fitness Function
fitness_history = []
def evaluate(individual, X, y):
    selected = [i for i, bit in enumerate(individual) if bit == 1]
    if not selected:
        return (0,)
    X_sel = X[:, selected]
    X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.3, random_state=42)
    model = get_model(classifier_name)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    score = f1_score(y_test, pred)
    fitness_history.append(score)
    return (score,)

# GA Logic
best_combination = None
fitness_progression = []
all_selected_features = []
def run_ga(X, y, ngen, pop_size):
    global fitness_progression
    n_feats = X.shape[1]
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", random.randint, 0, 1)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=n_feats)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate, X=X, y=y)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=pop_size)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("max", np.max)

    result_pop, logbook = algorithms.eaSimple(pop, toolbox, cxpb=0.5, mutpb=0.2,
                                              ngen=ngen, stats=stats, halloffame=hof, verbose=False)

    fitness_progression = logbook.select("avg")
    return hof[0]

if uploaded_file:
    df = load_data(uploaded_file)
    st.subheader("\U0001F5C3 Raw Data Preview")
    st.dataframe(df.head())

    X, y, feature_names = preprocess_data(df)

    best_overall = None
    best_score = 0
    for i in range(runs):
        st.write(f"### Trial {i+1}")
        fitness_history.clear()
        best_ind = run_ga(X, y, ngen, pop_size)
        selected = [fname for bit, fname in zip(best_ind, feature_names) if bit == 1]
        score = max(fitness_history)
        all_selected_features.append(selected)
        if score > best_score:
            best_score = score
            best_overall = best_ind
        st.write("Selected Features:", selected)
        st.write("Best F1 Score:", f"{score:.4f}")

    selected_features = [name for bit, name in zip(best_overall, feature_names) if bit == 1]
    st.success(f"\U0001F389 Best Features Selected: {selected_features}")

    # Model Evaluation
    X_df = pd.DataFrame(X, columns=feature_names)
    X_selected = X_df[selected_features]
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.3, random_state=42)
    final_model = get_model(classifier_name)
    final_model.fit(X_train, y_train)
    y_pred = final_model.predict(X_test)

    st.subheader("\U0001F4CA Evaluation Metrics")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
    col2.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")
    col3.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
    col4.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")

    # Confusion Matrix
    st.subheader("\U0001F5FA Confusion Matrix")
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)

    # Feature Importance
    if hasattr(final_model, "feature_importances_"):
        st.subheader("\U0001F9ED Feature Importances")
        importances = final_model.feature_importances_
        fi_df = pd.DataFrame({"Feature": selected_features, "Importance": importances})
        fig_imp = px.bar(fi_df.sort_values(by="Importance"), x="Importance", y="Feature", orientation='h')
        st.plotly_chart(fig_imp)

    # SHAP Explanation
    st.subheader("\U0001F52E SHAP Explanation")
    explainer = shap.Explainer(final_model.predict, X_test)
    shap_values = explainer(X_test)
    fig_shap = plt.figure()
    shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
    st.pyplot(fig_shap)

    # Fitness Progression
    st.subheader("\U0001F4C8 Fitness Progression (Avg)")
    fig2 = px.line(x=range(len(fitness_progression)), y=fitness_progression,
                   labels={"x": "Generation", "y": "Avg Fitness"})
    st.plotly_chart(fig2)

    # Save best model
    if st.button("\U0001F4BE Save Best Individual"):
        with open("best_individual.pkl", "wb") as f:
            pickle.dump(best_overall, f)
        st.success("Saved best individual to 'best_individual.pkl'")