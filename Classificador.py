import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import make_scorer, f1_score, precision_score, recall_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTENC
import matplotlib.pyplot as plt
from sklearn.preprocessing import KBinsDiscretizer


# 1. Carregar dataset
df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

# 2. Remover coluna ID e mapear BMI ordinalmente
df = df.drop(columns=["ID"], errors="ignore")
bmi_map = {"Normal": 0, "Overweight": 1, "Obese": 2}
df["BMI Category"] = df["BMI Category"].map(bmi_map).astype(float)

# 3. Identificar colunas categóricas e alvo
categorical_cols = ["Gender", "Occupation"]
target_col = "Sleep Disorder"

# 4. Converter colunas categóricas para dtype 'category'
for col in categorical_cols:
    df[col] = df[col].astype("category")


# 5. Separar X (features) e y (rótulo)
X = df.drop(columns=[target_col])
y = df[target_col]

# Tentar ver como os histogramas seriam provavelmente gerados
num_cols = X.select_dtypes(include=[np.number]).columns
discretizer = KBinsDiscretizer(n_bins=255, encode='ordinal', strategy='quantile')
X_binned = discretizer.fit_transform(X[num_cols])

for i, col in enumerate(num_cols[:3]):
    plt.figure(figsize=(6, 4))
    plt.hist(X_binned[:, i], bins=50, color="steelblue", edgecolor="black", alpha=0.8)
    plt.title(f"Distribuição binned — {col}", fontsize=12)
    plt.xlabel("Bin index (0–254)")
    plt.ylabel("Frequência")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"Binned_{col}.png")
    plt.close()

# 6. Converter rótulos em numéricos e guardar labels
le = LabelEncoder()
y = le.fit_transform(y)
class_names = le.classes_

scoring = {
    "f1_weighted": make_scorer(f1_score, average="weighted"),
    "precision_weighted": make_scorer(precision_score, average="weighted"),
    "recall_weighted": make_scorer(recall_score, average="weighted"),
}


# 7. Remover dados muito correlacionados
print("\nAnalisando correlações entre colunas numéricas...")
num_cols = X.select_dtypes(include=[np.number]).columns

corr_matrix = X[num_cols].corr().abs()  # matriz de correlação (valor absoluto)
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Definir limite de correlação alta
limite_corr = 0.9

# Encontrar colunas a remover
to_drop = [column for column in upper.columns if any(upper[column] > limite_corr)]

if to_drop:
    print(f"Removendo colunas altamente correlacionadas (>{limite_corr}): {to_drop}")
    X = X.drop(columns=to_drop)
else:
    print("Nenhuma correlação acima do limite encontrada.")

# 8. Split (estratificado)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=0, stratify=y
)

# 9. Aplicar SMOTE apenas no treino
print("Aplicando SMOTE para balancear as classes...")
# SMOTE comum mas distancia entre k vizinhos não é euclidiana, usa o vaule distance metric
# nova categoria da coluna 'x' é o mais comum dentre os k vizinhos
smote = SMOTENC(categorical_features = categorical_cols, random_state=0, k_neighbors=5)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

print("Tamanho original:", np.bincount(y_train))
print("Tamanho após SMOTE:", np.bincount(y_train_res))

# 10. Modelo base
clf_hgb = HistGradientBoostingClassifier(random_state=0)

# 11. Grid Search
param_grid = {
    'max_depth': [None, 5, 10, 15],
    'learning_rate': [0.05, 0.1, 0.2, 0.25],
    'max_iter': [100, 200, 300]
}

best_hgb = GridSearchCV(
    clf_hgb,
    param_grid,
    scoring='f1_weighted',
    cv=3
)
best_hgb.fit(X_train_res, y_train_res)

print("\nMelhores parâmetros:", best_hgb.best_params_)
print("Melhor F1 (validação):", best_hgb.best_score_)

# 12. Avaliação final no conjunto de teste real (sem SMOTE)
best_model = best_hgb.best_estimator_
y_pred = best_model.predict(X_test)

# analisar árvores da última iteração, classe 0, 1 e 2
# não deu, a função é tão otimizada que removerão isso....

print("\n=== Relatório Final ===")
print(classification_report(y_test, y_pred, target_names=class_names))
print("F1 (weighted):", f1_score(y_test, y_pred, average='weighted'))

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
disp.plot(cmap='Blues', colorbar=False)
plt.title("Matriz de Confusão — Distúrbios do Sono")
plt.savefig("ConfusionMatrix.png")
plt.close()


# 13. Importância das features
print("\nCalculando importâncias das features...")
result = permutation_importance(best_model, X_test, y_test, n_repeats=10, random_state=0)
importances = pd.Series(result.importances_mean, index=X.columns).sort_values(ascending=False)

print("\n=== Importância das Features ===")
print(importances)

# 14. Plotar gráfico de importâncias
plt.figure(figsize=(8, 5))
importances.plot(kind="bar", color="teal")
plt.title("Importância das Features (Permutação)")
plt.ylabel("Impacto médio no F1-score")
plt.tight_layout()
plt.savefig("Importancia.png")
