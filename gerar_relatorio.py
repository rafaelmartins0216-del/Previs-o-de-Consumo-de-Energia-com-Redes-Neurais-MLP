"""Gera um relatório Markdown a partir dos resultados reais do experimento."""

# Importa leitura de JSON.
import json
# Importa caminhos portáveis.
from pathlib import Path

# Localiza a pasta do projeto.
BASE = Path(__file__).resolve().parent
# Localiza o resumo produzido pelo treinamento.
RESUMO = BASE / "resultados" / "resumo.json"
# Define o relatório final.
SAIDA = BASE / "RELATORIO.md"

# Interrompe com uma orientação clara caso o modelo ainda não tenha sido executado.
if not RESUMO.exists():
    raise FileNotFoundError("Execute primeiro: python analise_mlp_energia.py")

# Carrega apenas resultados efetivamente medidos.
r = json.loads(RESUMO.read_text(encoding="utf-8"))
# Cria atalhos para as métricas.
m = r["metricas_teste_mlp"]
# Cria atalhos para a baseline.
b = r["metricas_teste_persistencia"]
# Cria um texto acadêmico enxuto e rastreável.
texto = f"""# Relatório breve — Predição de energia com MLP

## Objetivo

Prever o consumo de eletrodomésticos **10 minutos à frente** na residência de
baixo consumo energético estudada por Candanedo, Feldheim e Deramaix (2017).

## Protocolo

- Série ordenada no tempo e transformada em problema supervisionado.
- Defasagens de consumo de 10 min, 1 h, 2 h e 24 h, médias móveis e variáveis cíclicas.
- Divisão cronológica: {r['divisao']['treino']} treino, {r['divisao']['validacao']} validação
  e {r['divisao']['teste']} teste.
- Padronização aprendida exclusivamente no treino.
- Três arquiteturas, três sementes por arquitetura e early stopping pela validação.
- Teste consultado uma única vez depois da seleção.

## Resultados

| Modelo | MAE (Wh) | RMSE (Wh) | R² |
|---|---:|---:|---:|
| MLP selecionada | {m['MAE']:.2f} | {m['RMSE']:.2f} | {m['R2']:.3f} |
| Persistência | {b['MAE']:.2f} | {b['RMSE']:.2f} | {b['R2']:.3f} |

A arquitetura escolhida na validação foi **{r['melhor_modelo']['configuracao']}**,
com semente {r['melhor_modelo']['semente']} e melhor época
{r['melhor_modelo']['melhor_epoca']}. O tempo total foi
{r['tempo_segundos']:.1f} segundos.

No teste, a persistência foi superior à MLP nas três métricas. Isso é um resultado
negativo, porém cientificamente relevante: em horizonte tão curto, o consumo atual
é uma referência extremamente forte, e a maior complexidade da rede não garantiu
melhor generalização para o trecho cronológico mais recente.

## Análise crítica e paralelo com o artigo

O artigo original avaliou modelos orientados a dados — regressão linear, SVR,
random forest e gradient boosting — e enfatizou seleção/importância de variáveis.
Ele não apresentou uma MLP como modelo central. Assim, este trabalho **reproduz o
problema, os dados, o alvo e as métricas**, mas adapta a técnica para a MLP exigida
pelo Grupo B.

A principal diferença metodológica é temporal: esta versão prevê o próximo
intervalo e preserva a ordem cronológica. Isso evita que observações futuras
influenciem o treinamento, mas também torna os números incompatíveis com uma
comparação direta com resultados obtidos por partições aleatórias. As defasagens
dão memória explícita a uma rede feedforward, que, ao contrário de uma LSTM, não
mantém estado temporal interno.

## Limitações

Os dados representam uma única residência e cerca de 4,5 meses. Logo, não se pode
afirmar generalização para outras casas, estações ou perfis familiares. Picos de
consumo são raros e elevam o RMSE. Novos estudos deveriam usar validação walk-forward,
intervalos de confiança e dados de outras residências.

## Figuras

![Curva de aprendizado](resultados/curva_aprendizado.png)

![Configurações](resultados/comparacao_configuracoes.png)

![Real versus previsto](resultados/real_vs_previsto.png)
"""
# Grava o relatório em UTF-8.
SAIDA.write_text(texto, encoding="utf-8")
# Confirma o caminho gerado.
print(f"Relatório salvo em: {SAIDA}")
