# Import necessary libraries
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import time

# Genetic Algorithm class for feature selection
class GeneticAlgorithmFeatureSelector:
    def __init__(self, population_size=50, generations=50, crossover_prob=0.9, mutation_prob=0.1):
        self.population_size = population_size
        self.generations = generations
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.best_individual = None
        self.best_fitness = -np.inf
        self.fitness_history = []

    def initialize_population(self, num_features):
        # Create initial random population
        population = []
        for _ in range(self.population_size):
            individual = np.random.randint(0, 2, num_features).astype(bool)
            if not np.any(individual):  # Ensure at least one feature is selected
                individual[np.random.randint(0, num_features)] = True
            population.append(individual)
        return population

    def fitness_function(self, individual, X_train, y_train, X_val, y_val):
        # Evaluate the fitness of an individual (feature subset)
        selected_features = X_train.columns[individual]
        if len(selected_features) == 0:
            return -np.inf
        clf = RandomForestClassifier(n_estimators=100, random_state=42)
        clf.fit(X_train[selected_features], y_train)
        y_pred = clf.predict(X_val[selected_features])
        score = f1_score(y_val, y_pred)
        penalty = 0.01 * np.sum(individual) / len(individual)  # Penalty for more features
        return score - penalty

    def selection(self, population, fitness_scores):
        # Tournament selection
        selected = []
        for _ in range(len(population)):
            candidates = np.random.choice(len(population), size=3, replace=False)
            best_candidate = candidates[np.argmax([fitness_scores[c] for c in candidates])]
            selected.append(population[best_candidate])
        return selected

    def crossover(self, parent1, parent2):
        # Single-point crossover
        if np.random.rand() > self.crossover_prob:
            return parent1.copy(), parent2.copy()
        point = np.random.randint(1, len(parent1)-1)
        child1 = np.concatenate([parent1[:point], parent2[point:]])
        child2 = np.concatenate([parent2[:point], parent1[point:]])
        return child1, child2

    def mutation(self, individual):
        # Bit-flip mutation
        for i in range(len(individual)):
            if np.random.rand() < self.mutation_prob:
                individual[i] = not individual[i]
        if not np.any(individual):  # Ensure at least one feature remains selected
            individual[np.random.randint(0, len(individual))] = True
        return individual

    def evolve(self, X_train, y_train, X_val, y_val):
        # Main loop of the genetic algorithm
        num_features = X_train.shape[1]
        population = self.initialize_population(num_features)
        for generation in range(self.generations):
            fitness_scores = [self.fitness_function(ind, X_train, y_train, X_val, y_val) for ind in population]
            current_best_idx = np.argmax(fitness_scores)
            current_best_fitness = fitness_scores[current_best_idx]
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_individual = population[current_best_idx].copy()
            self.fitness_history.append(self.best_fitness)
            selected = self.selection(population, fitness_scores)
            next_population = []
            for i in range(0, len(selected), 2):
                if i + 1 >= len(selected):
                    next_population.append(selected[i])
                    break
                child1, child2 = self.crossover(selected[i], selected[i+1])
                next_population.extend([child1, child2])
            population = [self.mutation(ind) for ind in next_population]
            population[0] = self.best_individual.copy()  # Keep the best individual
        return self.best_individual

# ===== Helper Functions =====

def preprocess_data(df):
    # Clean and preprocess the dataset
    if 'customerID' in df.columns:
        df = df.drop(['customerID'], axis=1)
    label_encoder = LabelEncoder()
    categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 
                        'InternetService', 'StreamingTV', 'Churn']
    for col in categorical_cols:
        if col in df.columns:
            df[col] = label_encoder.fit_transform(df[col])
    df = df.fillna(df.mean(numeric_only=True))
    return df

def split_and_scale(df):
    # Split dataset and scale numerical features
    X = df.drop('Churn', axis=1)
    y = df['Churn']
    X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    num_cols = [c for c in num_cols if c in X_train.columns]
    if num_cols:
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols] = scaler.transform(X_test[num_cols])
    return X_train, X_test, y_train, y_test

def evaluate_model(X_train, X_test, y_train, y_test, features):
    # Train model and calculate metrics
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train[features], y_train)
    y_pred = clf.predict(X_test[features])
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    return acc, f1, cm, y_test, y_pred

# ===== Streamlit App =====

# App title
st.title("🎯 Customer Churn Feature Selection using Genetic Algorithm")

# Sidebar for file upload
st.sidebar.title("Upload your Dataset 📄")
uploaded_file = st.sidebar.file_uploader("Choose an Excel file", type=["xlsx"])

if uploaded_file:
    try:
        # Read uploaded file
        df = pd.read_excel(uploaded_file)
        st.success("✅ File Uploaded Successfully!")

        # Preprocess the data
        df = preprocess_data(df)

        if 'Churn' not in df.columns:
            st.error("🚨 'Churn' column not found!")
        else:
            # Data splitting and scaling
            with st.spinner("Splitting and scaling the data..."):
                X_train, X_test, y_train, y_test = split_and_scale(df)

            # Display dataset preview
            st.subheader("🔍 Dataset Overview")
            st.dataframe(df.head())

            # Evaluate model with all features
            st.subheader("📊 Model Evaluation - All Features")
            all_acc, all_f1, all_cm, y_test_all, y_pred_all = evaluate_model(X_train, X_test, y_train, y_test, X_train.columns)
            st.write(f"**Accuracy:** `{all_acc:.4f}`")
            st.write(f"**F1 Score:** `{all_f1:.4f}`")

            # Start Genetic Algorithm
            st.subheader("🚀 Running Genetic Algorithm...")
            X_train_ga, X_val_ga, y_train_ga, y_val_ga = train_test_split(
                X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
            )

            ga = GeneticAlgorithmFeatureSelector(population_size=50, generations=50)

            # Fake progress bar for better UI experience
            progress_bar = st.progress(0)
            for i in range(ga.generations):
                time.sleep(0.05)
                progress_bar.progress((i+1)/ga.generations)

            # Run GA to select best features
            best_mask = ga.evolve(X_train_ga, y_train_ga, X_val_ga, y_val_ga)
            selected_features = X_train.columns[best_mask]

            # Evaluate model with selected features
            sel_acc, sel_f1, sel_cm, y_test_sel, y_pred_sel = evaluate_model(X_train, X_test, y_train, y_test, selected_features)

            st.success("🎉 Feature Selection Completed!")

            # Show performance comparison
            st.subheader("📈 Results Comparison")
            comparison_df = pd.DataFrame({
                "Model": ["All Features", "Selected Features"],
                "Accuracy": [all_acc, sel_acc],
                "F1 Score": [all_f1, sel_f1]
            })

            fig1, ax1 = plt.subplots()
            comparison_df.plot(x="Model", y=["Accuracy", "F1 Score"], kind="bar", ax=ax1)
            plt.title("Performance Comparison")
            plt.xticks(rotation=0)
            st.pyplot(fig1)

            # Show selected features
            st.subheader("🎯 Selected Features")
            st.write(selected_features.tolist())

            # Show fitness over generations
            st.subheader("📈 Fitness Progress Over Generations")
            fig2, ax2 = plt.subplots()
            ax2.plot(ga.fitness_history, color='green')
            ax2.set_xlabel("Generation")
            ax2.set_ylabel("Best F1 Score")
            ax2.set_title("Genetic Algorithm Fitness Progress")
            st.pyplot(fig2)

            # Pie chart for feature selection
            st.subheader("🎨 Selected vs Unselected Features")
            selected_count = len(selected_features)
            total_count = len(X_train.columns)
            pie_data = pd.Series([selected_count, total_count - selected_count], index=["Selected", "Not Selected"])
            fig3, ax3 = plt.subplots()
            pie_data.plot(kind="pie", autopct='%1.1f%%', startangle=90, colors=["skyblue", "lightcoral"])
            plt.title("Feature Selection Distribution")
            st.pyplot(fig3)

            # Confusion matrix for all features
            st.subheader("🧩 Confusion Matrix - All Features")
            fig4, ax4 = plt.subplots()
            sns.heatmap(all_cm, annot=True, fmt="d", cmap="Blues", ax=ax4)
            ax4.set_xlabel("Predicted")
            ax4.set_ylabel("Actual")
            ax4.set_title("Confusion Matrix (All Features)")
            st.pyplot(fig4)

            # Confusion matrix for selected features
            st.subheader("🧩 Confusion Matrix - Selected Features")
            fig5, ax5 = plt.subplots()
            sns.heatmap(sel_cm, annot=True, fmt="d", cmap="Greens", ax=ax5)
            ax5.set_xlabel("Predicted")
            ax5.set_ylabel("Actual")
            ax5.set_title("Confusion Matrix (Selected Features)")
            st.pyplot(fig5)

    except Exception as e:
        st.error(f"❌ Error: {e}")

