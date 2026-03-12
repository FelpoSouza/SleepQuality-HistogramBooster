import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


# 1. Carregar dataset
df = pd.read_csv("Sleep_health_and_lifestyle_dataset.csv")

counts = df["Occupation"].value_counts()
disorders = df["Sleep Disorder"].value_counts()

#Cria o gráfico de pizza com as classes

plt.figure(figsize=(6,6))
plt.pie(
    disorders.values,              # tamanhos
    labels=disorders.index,        # rótulos (nomes das classes)
    autopct='%1.1f%%',          # porcentagem dentro das fatias
    startangle=90,              # rotaciona início
    counterclock=False
)
plt.title("Distúrbios do Sono")
plt.axis('equal')              # mantém círculo (aspect ratio)
plt.tight_layout()
plt.savefig("DisordersGraph.png")
plt.close()


# Cria o histograma (na verdade, um gráfico de barras categórico)
plt.figure(figsize=(10, 5))
counts.plot(kind="bar", color="skyblue", edgecolor="black")

plt.title("Distribuição de Profissões")
plt.xlabel("Profissão")
plt.ylabel("Número de Pessoas")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("HistProf.png")
plt.close()

counts = df["Age"].value_counts().sort_index()

# Cria o histograma 
plt.figure(figsize=(10, 5))
counts.plot(kind="bar", color="skyblue", edgecolor="black")

plt.title("Distribuição de Idades")
plt.xlabel("Idade")
plt.ylabel("Número de Pessoas")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("HistIdade.png")
plt.close()

# Cria o grafico de linhas atv física x qualidade sono 
plt.figure(figsize=(10, 5))
df["Quality of Sleep"].plot(kind="line", color="skyblue")

plt.title("Atividade Física x Qualidade do Sono")
plt.xlabel("Atividade Física")
plt.ylabel("Qualidade do Sono")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("Line_AtvQual.png")
plt.close()

# Cria o grafico de linhas duração do sono x qualidade sono
plt.figure(figsize=(10, 5))
df["Quality of Sleep"].plot(kind="line", color="skyblue")

plt.title("Duração do Sono x Qualidade do Sono")
plt.xlabel("Duração do Sono")
plt.ylabel("Qualidade do Sono")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("Line_DurQual.png")
plt.close()

# Scatter plot de analise do passos diários
fig, ax = plt.subplots()
sns.scatterplot(data=df, x="Daily Steps", y="Physical Activity Level", ax=ax)
plt.savefig("Scatter_StepsAL.png")
plt.close()
##########################################################################

# Box plot de analise do passos diários
fig, ax = plt.subplots()
sns.boxplot(y='Sleep Duration', data=df, showmeans=True, ax=ax)
plt.savefig("BoxPlotDuracao.png")
plt.close()
# Box plot de analise do passos diários
fig, ax = plt.subplots()
sns.boxplot(y='Quality of Sleep', data=df, showmeans=True, ax=ax)
plt.savefig("BoxPlotQualidade.png")
plt.close()

#BMI
fig, ax = plt.subplots()
sns.boxplot(x='BMI Category', y='Quality of Sleep', data=df, showmeans=True, ax=ax)
plt.savefig("BoxPlotBMIxQual.png")
plt.close()

fig, ax = plt.subplots()
sns.boxplot(x='BMI Category', y='Stress Level', data=df, showmeans=True, ax=ax)
plt.savefig("BoxPlotBMIxStress.png")
plt.close()

fig, ax = plt.subplots()
sns.boxplot(x='BMI Category', y='Systolic Pressure', data=df, showmeans=True, ax=ax)
plt.savefig("BoxPlotBMIxSistole.png")
plt.close()

fig, ax = plt.subplots()
sns.boxplot(x='BMI Category', y='Diastolic Pressure', data=df, showmeans=True, ax=ax)
plt.savefig("BoxPlotBMIxDiastole.png")
plt.close()

fig, ax = plt.subplots()
sns.boxplot(x='BMI Category', y='Heart Rate', data=df, showmeans=True, ax=ax)
plt.savefig("BoxPlotBMIxBatimento.png")
plt.close()

#Ocupação
sns.boxplot(x= 'Occupation', y='Stress Level', data=df, showmeans=True, ax=ax)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("BoxPlotOccupStress.png")
plt.close()

fig, ax = plt.subplots()
sns.boxplot(x= 'Occupation', y='Sleep Duration', data=df, showmeans=True, ax=ax)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("BoxPlotOccupSleepDur.png")
plt.close()

fig, ax = plt.subplots()
sns.boxplot(x= 'Occupation', y='Quality of Sleep', data=df, showmeans=True, ax=ax)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("BoxPlotOccupQual.png")
plt.close()

fig, ax = plt.subplots()
sns.boxplot(x= 'Occupation', y='Heart Rate', data=df, showmeans=True, ax=ax)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.savefig("BoxPlotOccupHR.png")
plt.close()

#IDADE x QUALIDADE DE SONO
stats = df.groupby("Age")["Quality of Sleep"].agg(["mean", "std", "median"])

plt.figure(figsize=(8,5))
plt.plot(stats.index, stats["mean"], label="Média", color="royalblue", marker="o")

# Área da variância (desvio padrão)
plt.fill_between(stats.index,
                 stats["mean"] - stats["std"],
                 stats["mean"] + stats["std"],
                 color="skyblue", alpha=0.3, label="±1 desvio padrão")

plt.plot(stats.index, stats["median"], label="Mediana", color="orange", linestyle="--")

plt.title("Qualidade do Sono por Idade (Média, Desvio e Mediana)")
plt.xlabel("Idade")
plt.ylabel("Qualidade do Sono")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("IdadexQual.png")
plt.close()
#IDADE x QUANTIDADE DE SONO
stats = df.groupby("Age")["Sleep Duration"].agg(["mean", "std", "median"])

plt.figure(figsize=(8,5))
plt.plot(stats.index, stats["mean"], label="Média", color="royalblue", marker="o")

# Área da variância (desvio padrão)
plt.fill_between(stats.index,
                 stats["mean"] - stats["std"],
                 stats["mean"] + stats["std"],
                 color="skyblue", alpha=0.3, label="±1 desvio padrão")

plt.plot(stats.index, stats["median"], label="Mediana", color="orange", linestyle="--")

plt.title("Duração do Sono por Idade (Média, Desvio e Mediana)")
plt.xlabel("Idade")
plt.ylabel("Duração do Sono")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("IdadexDura.png")
plt.close()
#IDADE x NIVEL ESTRESSE
stats = df.groupby("Age")["Stress Level"].agg(["mean", "std", "median"])

plt.figure(figsize=(8,5))
plt.plot(stats.index, stats["mean"], label="Média", color="royalblue", marker="o")

# Área da variância (desvio padrão)
plt.fill_between(stats.index,
                 stats["mean"] - stats["std"],
                 stats["mean"] + stats["std"],
                 color="skyblue", alpha=0.3, label="±1 desvio padrão")

plt.plot(stats.index, stats["median"], label="Mediana", color="orange", linestyle="--")

plt.title("Qualidade do Sono por Idade (Média, Desvio e Mediana)")
plt.xlabel("Idade")
plt.ylabel("Qualidade do Sono")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("IdadexStress.png")
plt.close()
#IDADE X NUMERO DE PASSOS
stats = df.groupby("Age")["Daily Steps"].agg(["mean", "std", "median"])

plt.figure(figsize=(8,5))
plt.plot(stats.index, stats["mean"], label="Média", color="royalblue", marker="o")

# Área da variância (desvio padrão)
plt.fill_between(stats.index,
                 stats["mean"] - stats["std"],
                 stats["mean"] + stats["std"],
                 color="skyblue", alpha=0.3, label="±1 desvio padrão")

plt.plot(stats.index, stats["median"], label="Mediana", color="orange", linestyle="--")

plt.title("Número de Passos Diário por Idade (Média, Desvio e Mediana)")
plt.xlabel("Idade")
plt.ylabel("Número de Passos")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("IdadexPasso.png")
plt.close()
#IDADE X ATIVIDADE FISICA
stats = df.groupby("Age")["Physical Activity Level"].agg(["mean", "std", "median"])

plt.figure(figsize=(8,5))
plt.plot(stats.index, stats["mean"], label="Média", color="royalblue", marker="o")

# Área da variância (desvio padrão)
plt.fill_between(stats.index,
                 stats["mean"] - stats["std"],
                 stats["mean"] + stats["std"],
                 color="skyblue", alpha=0.3, label="±1 desvio padrão")

plt.plot(stats.index, stats["median"], label="Mediana", color="orange", linestyle="--")

plt.title("Atividade Física em Minutos por Idade (Média, Desvio e Mediana)")
plt.xlabel("Idade")
plt.ylabel("Atividade Física (em minutos)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("IdadexAtvFis.png")
plt.close()

#
#APLICAR MINMAX NOS DADOS NUMERICOS PARA DEPOIS USAR HEATMAP E SUBPLOTS TALVEZ


# Selecionar colunas numéricas 
cols_to_scale = [
    "Age",
    "Sleep Duration",
    "Quality of Sleep",
    "Physical Activity Level",
    "Stress Level",
    "Systolic Pressure",
    "Diastolic Pressure",
    "Heart Rate",
    "Daily Steps"
]

# Garante que todas as colunas estão no dataset
df_pre_scaled = df[cols_to_scale].copy()

# Aplicar MinMaxScaler 
scaler = MinMaxScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_pre_scaled), columns=cols_to_scale)

# AQUI USAR SUBPLOTS

combinacoes = []
for i in range(len(cols_to_scale)):
    for j in range(i + 1, len(cols_to_scale)):
        combinacoes.append((cols_to_scale[i], cols_to_scale[j]))

# gera tupla de 2, combinando os elementos normalizados de 2 a 2

graficos_por_figura = 9     # 3x3
num_figuras = (len(combinacoes) + graficos_por_figura - 1) // graficos_por_figura
#  28 (combinação) + 9 (graficos por figura) - 1) // 9 -> 4 imagens

for fig_idx in range(num_figuras):
    # Seleciona as combinações dessa figura
    # qual o conjunto de tuplas que serão iterados sobre para gerar subplots 
    inicio = fig_idx * graficos_por_figura
    fim = inicio + graficos_por_figura
    # fim - inicio = graficos por figura (justamente como planejado)
    subset = combinacoes[inicio:fim]
    # a tupla a ser percorrida

    # Cria figura com 3x3 subplots
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    axes = axes.flatten()

    # Loop sobre cada subplot
    for ax, (x_col, y_col) in zip(axes, subset):
        # usa dos axes do plt anterior e as tuplas indicadas pelo subset plot
        ax.scatter(df_scaled[x_col], df_scaled[y_col], color="red", alpha=0.6)
        ax.set_xlabel(x_col, fontsize=8)
        ax.set_ylabel(y_col, fontsize=8)
        ax.set_title(f"{x_col} vs {y_col}", fontsize=9)

    # Desativa subplots vazios se não houver 9 gráficos
    for k in range(len(subset), len(axes)):
        axes[k].axis("off")

    plt.tight_layout()
    plt.suptitle(f"Relações 2 a 2 – Página {fig_idx+1}", fontsize=14, y=1.02)
    plt.savefig(f"CombinacaoNumerico {fig_idx+1}.png")

plt.close()

corr = df_scaled.corr(method="pearson")  # Pearson é o padrão e mais usado
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr,
    annot=True,           
    cmap="coolwarm",      
    center=0,             
    fmt=".2f",            
    square=True,
    linewidths=0.5
)
plt.title("Matriz de Correlação entre Variáveis Escalonadas", fontsize=14)
plt.tight_layout()
plt.savefig("HeatmapCorrelatoPearson.png")
plt.close()

corr = df_scaled.corr(method="spearman")  
plt.figure(figsize=(10, 8))
sns.heatmap(
    corr,
    annot=True,           
    cmap="coolwarm",      
    center=0,             
    fmt=".2f",            
    square=True,
    linewidths=0.5
)
plt.title("Matriz de Correlação entre Variáveis Escalonadas", fontsize=14)
plt.tight_layout()
plt.savefig("HeatmapCorrelatoSpearman.png")
plt.close()