# Predição de consumo de energia com MLP

Projeto acadêmico de reprodução adaptada de Candanedo, Feldheim e Deramaix
(2017), usando o dataset Appliances Energy Prediction da UCI.

## Como executar

```powershell
python -m pip install -r requirements.txt
python analise_mlp_energia.py
python gerar_relatorio.py
```

O primeiro comando de análise realiza nove treinamentos, seleciona o modelo pela
validação cronológica e avalia o teste uma única vez. Os artefatos são gravados
em `resultados/`; o segundo script cria `RELATORIO.md` com os números medidos.

## Arquivos

- `analise_mlp_energia.py`: experimento completo e extensamente comentado.
- `gerar_relatorio.py`: relatório reproduzível baseado no JSON de resultados.
- `energydata_complete.csv`: dataset original, mantido inalterado.

## Referências

- Candanedo, L. M.; Feldheim, V.; Deramaix, D. (2017). *Data driven prediction
  models of energy use of appliances in a low-energy house*. Energy and
  Buildings, 140, 81–97. DOI: 10.1016/j.enbuild.2017.01.083.
- UCI Machine Learning Repository. *Appliances Energy Prediction*. DOI:
  10.24432/C5VC8G.
