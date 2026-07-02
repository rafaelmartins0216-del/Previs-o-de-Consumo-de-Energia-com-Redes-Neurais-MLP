"""
Reprodução adaptada do estudo de Candanedo, Feldheim e Deramaix (2017).

Objetivo: prever o consumo de eletrodomésticos 10 minutos à frente com uma MLP.
O arquivo é deliberadamente comentado para funcionar também como material didático.
"""

# Importa ferramentas para copiar o modelo no melhor ponto do treinamento.
from copy import deepcopy
# Importa ferramentas para gravar resultados em JSON, um formato fácil de reutilizar.
import json
# Importa caminhos portáveis entre sistemas operacionais.
from pathlib import Path
# Importa medição de tempo para registrar o custo computacional.
from time import perf_counter
# Importa o sistema de avisos para ocultar apenas o aviso esperado do ajuste época a época.
import warnings

# Importa o Matplotlib para construir os gráficos acadêmicos.
import matplotlib.pyplot as plt
# Importa o NumPy para cálculos numéricos e controle das sementes aleatórias.
import numpy as np
# Importa o Pandas para leitura, transformação e exportação das tabelas.
import pandas as pd
# Importa as três métricas de regressão exigidas pelo professor.
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
# Importa a classe do aviso que ocorre porque cada chamada executa intencionalmente uma época.
from sklearn.exceptions import ConvergenceWarning
# Importa a implementação de Perceptron de Múltiplas Camadas para regressão.
from sklearn.neural_network import MLPRegressor
# Importa a padronização, essencial quando variáveis possuem escalas diferentes.
from sklearn.preprocessing import StandardScaler


# Define a pasta onde este próprio programa está salvo.
BASE_DIR = Path(__file__).resolve().parent
# Define o CSV padrão fornecido com o trabalho.
CSV_PADRAO = BASE_DIR / "energydata_complete.csv"
# Define uma pasta exclusiva para resultados, preservando os dados originais.
PASTA_RESULTADOS = BASE_DIR / "resultados"
# Define quantos instantes à frente serão previstos: 1 passo equivale a 10 minutos.
HORIZONTE = 1
# Define o número máximo de épocas de cada treinamento.
MAX_EPOCAS = 80
# Interrompe quando a validação não melhora por este número de épocas.
PACIENCIA = 10
# Exige esta redução mínima de RMSE para considerar que houve melhoria.
MELHORIA_MINIMA = 0.01
# Repete cada arquitetura com três inicializações, pois redes neurais são estocásticas.
SEMENTES = [42, 123, 2026]
# Testa arquiteturas pequenas e médias, mantendo o experimento viável em um computador comum.
CONFIGURACOES = [
    {"nome": "MLP_32", "camadas": (32,), "alpha": 0.0001, "taxa": 0.001},
    {"nome": "MLP_64_32", "camadas": (64, 32), "alpha": 0.0001, "taxa": 0.001},
    {"nome": "MLP_64_32_reg", "camadas": (64, 32), "alpha": 0.001, "taxa": 0.0005},
]


def carregar_e_preparar(caminho_csv):
    """Lê, ordena e transforma a série em um problema supervisionado sem vazamento."""
    # Lê o CSV e converte a coluna date diretamente para o tipo data/hora.
    dados = pd.read_csv(caminho_csv, parse_dates=["date"])
    # Ordena cronologicamente para garantir a sequência correta dos eventos.
    dados = dados.sort_values("date").reset_index(drop=True)
    # Falha cedo se houver timestamps repetidos, pois eles confundiriam a ordem temporal.
    if dados["date"].duplicated().any():
        raise ValueError("Foram encontrados timestamps duplicados.")
    # Falha cedo se houver valores ausentes, tornando a decisão de imputação explícita.
    if dados.isna().any().any():
        raise ValueError("O dataset contém valores ausentes; revise antes de treinar.")
    # Remove as variáveis aleatórias criadas no artigo como controles negativos.
    dados = dados.drop(columns=["rv1", "rv2"])
    # Cria hora cíclica: 23h fica matematicamente próxima de 0h.
    dados["hora_sen"] = np.sin(2 * np.pi * dados["date"].dt.hour / 24)
    # Cria a segunda coordenada cíclica necessária para representar a hora sem ambiguidades.
    dados["hora_cos"] = np.cos(2 * np.pi * dados["date"].dt.hour / 24)
    # Cria dia da semana cíclico para representar hábitos semanais.
    dados["semana_sen"] = np.sin(2 * np.pi * dados["date"].dt.dayofweek / 7)
    # Completa a representação cíclica do dia da semana.
    dados["semana_cos"] = np.cos(2 * np.pi * dados["date"].dt.dayofweek / 7)
    # Cria uma variável binária para separar fins de semana dos dias úteis.
    dados["fim_de_semana"] = (dados["date"].dt.dayofweek >= 5).astype(int)
    # Cria defasagens de 10 min, 1 h, 2 h e 24 h para oferecer memória temporal à MLP.
    for atraso in [1, 6, 12, 144]:
        # Usa apenas consumo passado ou presente, nunca informação futura.
        dados[f"Appliances_lag_{atraso}"] = dados["Appliances"].shift(atraso)
    # Cria média móvel da última hora e desloca uma posição para evitar incluir o alvo futuro.
    dados["Appliances_media_1h"] = dados["Appliances"].shift(1).rolling(6).mean()
    # Cria média móvel das últimas 6 horas para representar o nível recente de consumo.
    dados["Appliances_media_6h"] = dados["Appliances"].shift(1).rolling(36).mean()
    # Copia o consumo atual, conhecido no momento da previsão e essencial para persistência.
    dados["Appliances_atual"] = dados["Appliances"]
    # Define o alvo como o consumo do próximo registro, isto é, 10 minutos à frente.
    dados["alvo_futuro"] = dados["Appliances"].shift(-HORIZONTE)
    # Remove linhas iniciais sem histórico e a linha final sem alvo futuro.
    dados = dados.dropna().reset_index(drop=True)
    # Calcula o timestamp futuro mesmo após a remoção da última linha sem alvo.
    datas_alvo = dados["date"] + pd.Timedelta(minutes=10 * HORIZONTE)
    # Remove data, alvo futuro e consumo contemporâneo da matriz de entrada.
    colunas_excluir = ["date", "alvo_futuro", "Appliances"]
    # Constrói a matriz X somente com variáveis disponíveis no instante da previsão.
    X = dados.drop(columns=colunas_excluir)
    # Constrói o vetor y com o consumo a ser previsto.
    y = dados["alvo_futuro"]
    # Retorna também datas e nomes para documentação e gráficos.
    return X, y, datas_alvo, list(X.columns), dados


def dividir_cronologicamente(X, y, datas):
    """Separa 70%/15%/15% sem embaralhar e, portanto, sem olhar o futuro."""
    # Calcula o último índice do conjunto de treino.
    fim_treino = int(len(X) * 0.70)
    # Calcula o último índice do conjunto de validação.
    fim_validacao = int(len(X) * 0.85)
    # Fatia treino usando o trecho mais antigo da série.
    X_treino, y_treino = X.iloc[:fim_treino], y.iloc[:fim_treino]
    # Fatia validação usando o trecho imediatamente posterior ao treino.
    X_valid, y_valid = X.iloc[fim_treino:fim_validacao], y.iloc[fim_treino:fim_validacao]
    # Reserva o trecho mais recente somente para a avaliação final.
    X_teste, y_teste = X.iloc[fim_validacao:], y.iloc[fim_validacao:]
    # Separa as datas do teste para o gráfico temporal.
    datas_teste = datas.iloc[fim_validacao:]
    # Retorna todas as partições de forma explícita.
    return X_treino, X_valid, X_teste, y_treino, y_valid, y_teste, datas_teste


def calcular_metricas(y_real, y_previsto):
    """Calcula MAE, RMSE e R², exatamente as métricas pedidas na especificação."""
    # Calcula o erro absoluto médio em Wh, fácil de interpretar.
    mae = mean_absolute_error(y_real, y_previsto)
    # Calcula a raiz do erro quadrático médio, que penaliza mais os grandes erros.
    rmse = np.sqrt(mean_squared_error(y_real, y_previsto))
    # Calcula a fração da variabilidade explicada pelo modelo.
    r2 = r2_score(y_real, y_previsto)
    # Converte os resultados para float nativo, compatível com JSON.
    return {"MAE": float(mae), "RMSE": float(rmse), "R2": float(r2)}


def treinar_uma_rodada(config, semente, X_treino, y_treino, X_valid, y_valid):
    """Treina época a época, escolhendo o ponto de menor RMSE na validação."""
    # Cria a MLP com ReLU, otimizador Adam e uma época por chamada.
    modelo = MLPRegressor(
        hidden_layer_sizes=config["camadas"],
        activation="relu",
        solver="adam",
        alpha=config["alpha"],
        learning_rate_init=config["taxa"],
        batch_size=256,
        max_iter=1,
        warm_start=True,
        shuffle=False,
        random_state=semente,
    )
    # Começa sem melhor modelo armazenado.
    melhor_modelo = None
    # Começa com erro infinito para que a primeira época seja aceita.
    melhor_rmse = np.inf
    # Guarda a época escolhida.
    melhor_epoca = 0
    # Conta épocas consecutivas sem melhoria.
    sem_melhora = 0
    # Guarda a curva de erro quadrático do treino.
    loss_treino = []
    # Guarda a curva de RMSE da validação.
    rmse_validacao = []
    # Repete o ajuste até o limite máximo de épocas.
    for epoca in range(1, MAX_EPOCAS + 1):
        # Oculta apenas o aviso esperado de max_iter=1, pois controlamos as épocas externamente.
        with warnings.catch_warnings():
            # Declara que o aviso de uma única época não representa falha de convergência global.
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            # Atualiza os pesos usando todo o conjunto de treino na ordem cronológica.
            modelo.fit(X_treino, y_treino)
        # Registra a loss interna da MLP para a curva de aprendizado.
        loss_treino.append(float(modelo.loss_))
        # Faz previsões no conjunto que orienta as decisões de treinamento.
        previsto_valid = modelo.predict(X_valid)
        # Calcula o RMSE da validação sem tocar no teste.
        rmse_atual = float(np.sqrt(mean_squared_error(y_valid, previsto_valid)))
        # Acrescenta o valor à curva de validação.
        rmse_validacao.append(rmse_atual)
        # Verifica se a melhoria supera a tolerância definida.
        if rmse_atual < melhor_rmse - MELHORIA_MINIMA:
            # Atualiza o menor erro conhecido.
            melhor_rmse = rmse_atual
            # Guarda a época correspondente.
            melhor_epoca = epoca
            # Copia pesos e estado do otimizador no melhor ponto.
            melhor_modelo = deepcopy(modelo)
            # Reinicia a contagem de estagnação.
            sem_melhora = 0
        else:
            # Incrementa a contagem quando não há melhoria relevante.
            sem_melhora += 1
        # Para cedo para reduzir sobreajuste e tempo computacional.
        if sem_melhora >= PACIENCIA:
            break
    # Empacota as curvas para posterior comparação.
    historico = {"loss_treino": loss_treino, "rmse_validacao": rmse_validacao}
    # Retorna o melhor estado, e não necessariamente o estado da última época.
    return melhor_modelo, melhor_rmse, melhor_epoca, historico


def salvar_graficos(melhor, tabela_rodadas, y_teste, previsao_teste, datas_teste):
    """Gera os gráficos essenciais do protocolo experimental."""
    # Cria uma figura para a curva de aprendizado.
    plt.figure(figsize=(9, 5))
    # Plota a loss de treino por época.
    plt.plot(melhor["historico"]["loss_treino"], label="Loss de treino (MSE/2)")
    # Plota o RMSE de validação em eixo secundário para manter unidades legíveis.
    eixo2 = plt.gca().twinx()
    # Desenha a curva que decide o early stopping.
    eixo2.plot(melhor["historico"]["rmse_validacao_Wh"], color="tab:orange", label="RMSE validação")
    # Nomeia o eixo horizontal.
    plt.gca().set_xlabel("Época")
    # Nomeia o eixo esquerdo.
    plt.gca().set_ylabel("Loss de treino")
    # Nomeia o eixo direito.
    eixo2.set_ylabel("RMSE de validação (Wh)")
    # Define um título informativo.
    plt.title("Curva de aprendizado do melhor MLP")
    # Ativa uma grade discreta.
    plt.gca().grid(alpha=0.25)
    # Ajusta margens automaticamente.
    plt.tight_layout()
    # Salva em alta resolução para uso no relatório e nos slides.
    plt.savefig(PASTA_RESULTADOS / "curva_aprendizado.png", dpi=180)
    # Fecha a figura para liberar memória.
    plt.close()
    # Seleciona uma semana do teste para evitar um gráfico visualmente congestionado.
    n_semana = min(7 * 24 * 6, len(y_teste))
    # Cria a figura da comparação temporal.
    plt.figure(figsize=(11, 5))
    # Plota os valores reais.
    plt.plot(datas_teste.iloc[:n_semana], y_teste.iloc[:n_semana], label="Real", linewidth=1.2)
    # Plota as previsões do modelo.
    plt.plot(datas_teste.iloc[:n_semana], previsao_teste[:n_semana], label="MLP", linewidth=1.1)
    # Nomeia os eixos.
    plt.ylabel("Consumo (Wh)")
    # Explica o recorte no título.
    plt.title("Real versus previsto — primeira semana do teste")
    # Exibe a legenda.
    plt.legend()
    # Inclina datas para melhorar a leitura.
    plt.xticks(rotation=25)
    # Ajusta margens.
    plt.tight_layout()
    # Salva o gráfico temporal.
    plt.savefig(PASTA_RESULTADOS / "real_vs_previsto.png", dpi=180)
    # Fecha a figura.
    plt.close()
    # Resume RMSE médio e desvio por configuração.
    resumo = tabela_rodadas.groupby("configuracao")["RMSE_validacao"].agg(["mean", "std"])
    # Cria o gráfico comparativo.
    plt.figure(figsize=(8, 5))
    # Desenha barras com dispersão entre sementes.
    plt.bar(resumo.index, resumo["mean"], yerr=resumo["std"], capsize=5)
    # Nomeia o eixo vertical.
    plt.ylabel("RMSE de validação (Wh)")
    # Explica a barra de erro.
    plt.title("Otimização: média ± desvio-padrão em 3 sementes")
    # Ajusta os rótulos.
    plt.xticks(rotation=12)
    # Ajusta margens.
    plt.tight_layout()
    # Salva a comparação das configurações.
    plt.savefig(PASTA_RESULTADOS / "comparacao_configuracoes.png", dpi=180)
    # Fecha a figura.
    plt.close()


def executar():
    """Executa o protocolo completo e salva resultados reproduzíveis."""
    # Cria a pasta de saída se ela ainda não existir.
    PASTA_RESULTADOS.mkdir(exist_ok=True)
    # Registra o início da execução.
    inicio = perf_counter()
    # Prepara a série como previsão supervisionada.
    X, y, datas, nomes_features, dados_modelados = carregar_e_preparar(CSV_PADRAO)
    # Faz a divisão temporal exigida.
    partes = dividir_cronologicamente(X, y, datas)
    # Desempacota as partes para tornar o restante do código legível.
    X_treino, X_valid, X_teste, y_treino, y_valid, y_teste, datas_teste = partes
    # Cria o padronizador das entradas.
    escala_X = StandardScaler()
    # Aprende médias e desvios somente no treino, impedindo vazamento.
    X_treino_z = escala_X.fit_transform(X_treino)
    # Aplica exatamente a transformação aprendida à validação.
    X_valid_z = escala_X.transform(X_valid)
    # Aplica exatamente a transformação aprendida ao teste.
    X_teste_z = escala_X.transform(X_teste)
    # Cria um padronizador separado para a variável-alvo.
    escala_y = StandardScaler()
    # Aprende a escala do alvo apenas no treino.
    y_treino_z = escala_y.fit_transform(y_treino.to_numpy().reshape(-1, 1)).ravel()
    # Transforma a validação sem aprender nada com ela.
    y_valid_z = escala_y.transform(y_valid.to_numpy().reshape(-1, 1)).ravel()
    # Inicializa a coleção de resultados de todas as rodadas.
    rodadas = []
    # Inicializa o melhor resultado global.
    melhor = None
    # Percorre as configurações candidatas.
    for config in CONFIGURACOES:
        # Repete cada configuração com diferentes inicializações dos pesos.
        for semente in SEMENTES:
            # Treina e seleciona a melhor época pela validação.
            modelo, rmse_z, epoca, historico = treinar_uma_rodada(
                config, semente, X_treino_z, y_treino_z, X_valid_z, y_valid_z
            )
            # Converte as previsões de validação de volta para Wh.
            previsto_valid = escala_y.inverse_transform(
                modelo.predict(X_valid_z).reshape(-1, 1)
            ).ravel()
            # Impõe o limite físico: consumo de energia não pode ser negativo.
            previsto_valid = np.clip(previsto_valid, 0, None)
            # Calcula métricas interpretáveis na unidade original.
            metricas_valid = calcular_metricas(y_valid, previsto_valid)
            # Registra os elementos necessários à auditoria da rodada.
            registro = {
                "configuracao": config["nome"],
                "camadas": str(config["camadas"]),
                "alpha": config["alpha"],
                "taxa": config["taxa"],
                "semente": semente,
                "melhor_epoca": epoca,
                "RMSE_validacao": metricas_valid["RMSE"],
                "MAE_validacao": metricas_valid["MAE"],
                "R2_validacao": metricas_valid["R2"],
            }
            # Acrescenta o registro à tabela experimental.
            rodadas.append(registro)
            # Atualiza o campeão usando apenas a validação.
            if melhor is None or metricas_valid["RMSE"] < melhor["RMSE_validacao"]:
                # Converte a curva padronizada para Wh antes de guardar o campeão.
                historico["rmse_validacao_Wh"] = [
                    valor * float(escala_y.scale_[0]) for valor in historico["rmse_validacao"]
                ]
                # Guarda modelo, configuração e histórico do novo campeão.
                melhor = {
                    **registro,
                    "modelo": modelo,
                    "historico": historico,
                    "config": config,
                }
    # Converte os registros em tabela.
    tabela_rodadas = pd.DataFrame(rodadas)
    # Salva todas as rodadas, não apenas a vencedora.
    tabela_rodadas.to_csv(PASTA_RESULTADOS / "rodadas_validacao.csv", index=False)
    # Faz uma única avaliação final no conjunto de teste intocado.
    previsao_z = melhor["modelo"].predict(X_teste_z)
    # Retorna as previsões à unidade Wh.
    previsao_teste = escala_y.inverse_transform(previsao_z.reshape(-1, 1)).ravel()
    # Impõe o mesmo limite físico usado durante a seleção.
    previsao_teste = np.clip(previsao_teste, 0, None)
    # Calcula as métricas finais.
    metricas_teste = calcular_metricas(y_teste, previsao_teste)
    # Cria uma baseline de persistência: o próximo consumo será igual ao consumo atual.
    baseline = dados_modelados["Appliances"].iloc[-len(y_teste):].to_numpy()
    # Calcula as métricas da referência simples.
    metricas_baseline = calcular_metricas(y_teste, baseline)
    # Monta uma tabela linha a linha para análise de erros.
    previsoes = pd.DataFrame(
        {"data": datas_teste, "real_Wh": y_teste, "previsto_Wh": previsao_teste}
    )
    # Acrescenta o erro assinado para diagnosticar sub e superestimação.
    previsoes["erro_Wh"] = previsoes["previsto_Wh"] - previsoes["real_Wh"]
    # Salva as previsões finais.
    previsoes.to_csv(PASTA_RESULTADOS / "previsoes_teste.csv", index=False)
    # Gera todos os gráficos requisitados.
    salvar_graficos(melhor, tabela_rodadas, y_teste, previsao_teste, datas_teste)
    # Calcula o tempo total de execução.
    duracao = perf_counter() - inicio
    # Cria um resumo autocontido para o relatório automático.
    resumo = {
        "dataset": {
            "linhas_originais": 19735,
            "linhas_modeladas": len(X),
            "numero_features": len(nomes_features),
            "inicio": str(datas.iloc[0]),
            "fim": str(datas.iloc[-1]),
        },
        "divisao": {
            "treino": len(X_treino),
            "validacao": len(X_valid),
            "teste": len(X_teste),
        },
        "melhor_modelo": {
            key: value for key, value in melhor.items() if key not in {"modelo", "historico", "config"}
        },
        "metricas_teste_mlp": metricas_teste,
        "metricas_teste_persistencia": metricas_baseline,
        "tempo_segundos": duracao,
        "features": nomes_features,
    }
    # Grava o resumo com acentos preservados.
    (PASTA_RESULTADOS / "resumo.json").write_text(
        json.dumps(resumo, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    # Exibe o resultado principal no terminal.
    print(json.dumps(resumo, ensure_ascii=False, indent=2))


# Executa somente quando o arquivo é chamado diretamente.
if __name__ == "__main__":
    # Inicia o experimento completo.
    executar()
