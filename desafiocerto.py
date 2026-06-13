"""
GeoenergiaLab - Processo Seletivo de Iniciação Científica
Candidato: Eduardo Cavalheiro Pentiado

Análise Exploratória dos Dados
"""


# IMPORTAÇÃO DAS BIBLIOTECAS

import numpy as np
import matplotlib.pyplot as plt


# CONFIGURAÇÕES VISUAIS


plt.style.use('ggplot')

plt.rcParams['figure.dpi'] = 120
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['legend.fontsize'] = 10


# DADOS

casos = list(range(1, 31))

weber = [
    850, 920, 980, 1050, 1150, 1250, 1320, 1450, 1520, 1650,
    1750, 1850, 1950, 2050, 2150, 2250, 2350, 2450, 2550, 2650,
    2750, 2850, 2950, 3050, 3150, 3250, 3350, 3450, 3550, 3650
]

temperatura = [
    18, 19, 20, 22, 21, 23, 20, 19, 21, 22,
    24, 25, 19, 20, 22, 23, 21, 20, 22, 24,
    25, 26, 27, 21, 20, 22, 23, 24, 19, 65
]

vazao = [
    12, 14, 15, 15, 16, 18, 18, 20, 20, 22,
    22, 24, 25, 25, 27, 28, 28, 30, 30, 32,
    32, 34, 35, 35, 36, 38, 38, 40, 40, 42
]

d50 = [
    610, 585, 570, 550, 530, 505, 495, 470, 455, 440,
    425, 410, 395, 385, 370, 360, 350, 340, 330, 320,
    315, 305, 295, 290, 280, 275, 270, 265, None, 850
]


# TRATAMENTO DOS DADOS


weber_limpo = []
vazao_limpa = []
d50_limpo = []

for i in range(len(casos)):
    if d50[i] is not None and i != 29:  # remove Caso 30
        weber_limpo.append(weber[i])
        vazao_limpa.append(vazao[i])
        d50_limpo.append(d50[i])

weber_limpo = np.array(weber_limpo)
vazao_limpa = np.array(vazao_limpa)
d50_limpo = np.array(d50_limpo)


# GRÁFICO 1 - WEBER x D50


plt.figure(figsize=(8, 5))

plt.scatter(
    weber_limpo,
    d50_limpo,
    s=70,
    color='#1f77b4',
    edgecolors='black',
    label='Dados experimentais'
)

coef1 = np.polyfit(weber_limpo, d50_limpo, 2)
modelo1 = np.poly1d(coef1)

x1 = np.linspace(min(weber_limpo), max(weber_limpo), 300)

plt.plot(
    x1,
    modelo1(x1),
    color='darkgreen',
    linewidth=2.5,
    label='Curva de tendência'
)

# R²

y_pred1 = modelo1(weber_limpo)

ss_res1 = np.sum((d50_limpo - y_pred1) ** 2)
ss_tot1 = np.sum((d50_limpo - np.mean(d50_limpo)) ** 2)

r2_1 = 1 - ss_res1 / ss_tot1

plt.text(
    2700,
    520,
    f'R² = {r2_1:.4f}',
    bbox=dict(facecolor='white', edgecolor='black')
)

plt.title('Influência do Número de Weber no Tamanho das Gotas')
plt.xlabel('Número de Weber (We)')
plt.ylabel('Diâmetro Mediano da Gota d50 (µm)')
plt.legend()
plt.tight_layout()

plt.savefig('grafico1_weber_d50.png', dpi=300)
plt.show()


# GRÁFICO 2 - VAZÃO x D50


plt.figure(figsize=(8, 5))

plt.scatter(
    vazao_limpa,
    d50_limpo,
    s=70,
    color='#9467bd',
    edgecolors='black',
    label='Dados experimentais'
)

coef2 = np.polyfit(vazao_limpa, d50_limpo, 1)
modelo2 = np.poly1d(coef2)

x2 = np.linspace(min(vazao_limpa), max(vazao_limpa), 300)

plt.plot(
    x2,
    modelo2(x2),
    '--',
    color='black',
    linewidth=2,
    label='Regressão linear'
)

# R²

y_pred2 = modelo2(vazao_limpa)

ss_res2 = np.sum((d50_limpo - y_pred2) ** 2)
ss_tot2 = np.sum((d50_limpo - np.mean(d50_limpo)) ** 2)

r2_2 = 1 - ss_res2 / ss_tot2

plt.text(
    33,
    540,
    f'R² = {r2_2:.4f}',
    bbox=dict(facecolor='white', edgecolor='black')
)

plt.title('Influência da Vazão no Tamanho das Gotas')
plt.xlabel('Vazão Volumétrica (L/min)')
plt.ylabel('Diâmetro Mediano da Gota d50 (µm)')
plt.legend()
plt.tight_layout()

plt.savefig('grafico2_vazao_d50.png', dpi=300)
plt.show()


# GRÁFICO 3 - TEMPERATURA x CASO

plt.figure(figsize=(8, 5))

plt.plot(
    casos[:29],
    temperatura[:29],
    color='#2ca02c',
    marker='o',
    linewidth=2,
    markersize=5,
    label='Temperatura normal'
)

plt.scatter(
    30,
    temperatura[29],
    color='red',
    edgecolors='black',
    marker='X',
    s=180,
    zorder=5,
    label='Outlier térmico'
)

plt.annotate(
    'Caso 30\nT = 65°C',
    xy=(30, 65),
    xytext=(23, 55),
    fontsize=10,
    fontweight='bold',
    color='red',
    arrowprops=dict(
        arrowstyle='->',
        lw=1.5,
        color='red'
    )
)

plt.title('Monitoramento Térmico dos Ensaios')
plt.xlabel('Número do Caso')
plt.ylabel('Temperatura (°C)')

plt.xlim(1, 31)
plt.ylim(15, 70)

plt.legend()
plt.tight_layout()

plt.savefig('grafico3_temperatura.png', dpi=300)
plt.show()

# RESULTADOS

print("\n" + "="*50)
print("RESUMO DA ANÁLISE")
print("="*50)
print(f"Total de casos: {len(casos)}")
print(f"Casos válidos utilizados: {len(d50_limpo)}")
print("Caso removido: 30 (outlier térmico)")
print(f"R² Gráfico 1 (Weber x d50): {r2_1:.4f}")
print(f"R² Gráfico 2 (Vazão x d50): {r2_2:.4f}")
print("="*50)
print("Gráficos salvos com sucesso.")
print("="*50)