# IA048 - Atividade 1 - Regressão Linear

Prof. Levy Boccato

**Alunos:** Pedro Henrique Pinheiro Linhares (175807) e Rafael Hoyos (175100).

---

## Introdução

Esta atividade consiste em utilizar o modelo de regressão linear para realizar previsões em uma série temporal que apresenta três momentos, sendo um deles muito diferente dos outros dois. Este momento corresponde à pandemia de covid-19, que afetou o mundo inteiro de forma devastadora. Uma das formas de controlar o alastramento desta doença foi através da quarentena, que consistiu em manter todas as pessoas isoladas em suas casas, permitindo a saída apenas para trabalhadores essenciais, como profissionais de saúde.

Esta realidade que durou os anos de 2020 e 2021, aproximadamente, marcam um acontecimento inédito na série temporal amostrada. Neste momento, houve uma queda abrupta no número dos vôos, que se manteve por mais de um ano, e que se modificou após a adoção de algumas medidas de flexibilização da quarentena.

Tendo este contexto, foi proposto pela atividade o uso do modelo de reressão linear para realizar tais previsões. Para isto, nossa dupla utilizou o modelo de regressão linear do scikit-learning para realizar as predições. Além disto, as métricas de qualidade do modelo foram extraídas da mesma biblioteca de acordo com o que nos foi pedido na atividade.

Por se tratar de uma série temporal, tivemos que utilizar a estratégia do uso de lag, como proposto na atividade, para realizar as predições, pois utilizar toda a série temporal para isto seria muito custoso computacionalmente, visto que iria gerar um vetor de pesos do tamanho da série temporal. Além disso, informações muito antigas tendem a não interferir de forma significativa na predição. Para isto, foram criadas matrizes com valores atrasados cujo tamanho dependia do valor do atraso.

Outra estratégia adotada foi o uso de duas normalizações diferentes para saber como elas iriam influenciar na previsão do modelo. As normalizações adotadas foram z-norma e minmax.

Por fim, para o treinamento e validação do modelo, foi adotada estratégia do holdout. Desta forma, utilizamos os anos de 2018 e 2019 para servirem de validação para o treinamento no item b. Já para o item c, foram utilizados os anos de 2020 e 2021 como propostos pelo enunciado.



## Item a)

Para gerar o gráfico, foi feito o plot *"line"* simples com os pontos do número total de vôos (Flt) no eixo y e o tempo no eixo x:
A figura gerada foi:
![Plot da série temporal](results/série-temporal-completa-2003-a-2023.png)

Obs: As labels no eixo x fazem referência ao mês de janeiro de cada respectivo ano.\

Observa-se que, assim como o enunciado diz, a série temporal apresenta três faixas distintas de comportamento, provavelmente associadas aos seguinte fatores:

1. **Jan/2003 a Ago/2008:** Os valores parecem se estabilizar no "alto" a partir de 2004 (até 2008). Isso deve se dar pela expansão econômica global mais a forte demanda no setor aéreo que ocorreram na época.

2. **Set/2008 a Dez/2019:** Nota-se uma queda a partir da metade de 2008, ano que já remete à famosa *Crise de 2008*, que deve ser um dos principais motivos dessa queda. Algumas companhias aéreas chegaram a se fundir para otimizar frotas e voar com (menos) aeronaves, mais cheias [[2]](#referências). A partir de então, o setor adotou uma rigorosa "disciplina de capacidade", a favor de aviões maiores e priorizando altas taxas de ocupação. Logo, embora a demanda de passageiros tenha se recuperado ao longo da década (2010-2019), o número total de voos permaneceu em um patamar inferior e estabilizado devido à maior eficiência operacional. Vale ressaltar, também, que, até 2015, aproximadamente, houve a crise do euro, o que acarretou uma tendência de queda durante a primeira metade deste período, o que só foi revertido a partir do acordo de recuperação econômica da Grécia. Após isso, houve uma tendência de aumento nos anos subsequentes.

3. **Jan/2020 a Set/2023:** Despencada histórica com a Pandemia de COVID-19 (2020) seguida por uma recuperação gradual com o avanço da vacinação e o fim dos lockdowns.

![Plot da série temporal apenas 2023](results/serie_temporal_2003.png)

---

## Item b)

![RMSE / lags validação](results/rmse_lagsb.png)

## b1
Este gráfico evidencia três platôs:
1. 1<=K<=6: Neste intervalo o modelo está prevendo muito mal, tendo em vista os RMSE médios de cada ponto. Isto pode ter relação com o fato de que menos da metade do ano está sendo usado. E como durante o ano existem muitas flutuações, a tendência anual não é levada em consideração.
2. 7<=K<=12: Neste intervalo já é possível obter um platô mediano, com valores para RMSE melhores em média. Uma possível justificativa é o fato de que o modelo está aprendendo com muito mais dados que anteriormente. Com isto, é possível que um lag de até um ano ainda não é bom o suficiente.
3. 13<=K<=24: Este intervalo é o melhor platô de todos, permitindo observar que adicionar mais de um ano de lag na série temporal é muito bom para reduzir o RMSE. Porém, a inferência de que aumentar o lag indiscriminadamente reduz o RMSE é incorreta, pelo que pode ser visto no gráfico.

Desta forma, conjecturamos que no caso 1 o modelo fica muito frágil a variações, pois ele não teve disponível um intervalo suficientemente satisfatório para prever flutuações. Já o caso 3 mostra que o modelo com pelo menos 13 meses de lag apresenta uma melhor previsão da série temporal por conta de ter tido dados suficientes para formar uma boa previsão, porém, aumentar este lag faz com que o modelo faça previsões levemente piores, provavelmente por conta de que os dados muito atrasados não possuem tanta relação com os dados que serão previstos, se tornando apenas ruído. Por fim, este excesso de pontos faz com que o modelo fique excessivamente ajustado a este período fornecido. Desta forma, dados novos não terão tanta qualidade na previsão.

![previsao_melhor_modelok14](results/previsao_melhor_modelob2.png)

```python
# Cálculo de métricas do modelo em relação aos dados de teste indo de 2020 até 2023

# RMSE da previsão com o melhor K
rmse_teste = root_mean_squared_error(
        y_teste,
        previsoes_teste
    )
print(f"O RMSE do modelo com o melhor K é {rmse_teste:.4f}.\n")

# MAPE da previsão com o melhor K
mape = mean_absolute_percentage_error(y_teste, previsoes_teste)
print(f"O MAPE do modelo com o melhor K é {mape:.4f}.\n")

# R^2 da previsão com o melhor K
r2 = r2_score(y_teste, previsoes_teste)
print(f"O R^2 do modelo com o melhor K é {r2:.4f}.")
```

```txt
O RMSE do modelo com o melhor K é 114566.1933.

O MAPE do modelo com o melhor K é 0.1353.

O R^2 do modelo com o melhor K é 0.3582.
```

## b2

Analisando o gráfico do predito para os dados do teste em função dos dados reais, é possível concluir que o modelo foi muito ineficiente para predizer os valores fora da normalidade do período, no caso, a pandemia. Após esse período, o modelo voltou a predizer de forma muito mais coerente.

Isto faz sentido quando se leva em consideração que todo o modelo foi treinado e validado com dados de um período de normalidade, em que o período da série estava estabilizado em 12 meses.

Analisando de forma mais detalhista, é possível observar que os dados previstos estão inicialmente seguindo a tendência em que eles foram treinados para seguir.Porém, ao lidar com a nova realidade dos dados pandêmicos, ele começa bem nos dois primeiros meses, antes da quarentena, e logo se observa um erro absurdo, pois o terceiro mês já trouxe um padrão em que o modelo não estava treinado para prever. Assim, ele passa os anos seguintes errando absurdamente, até que o contexto global volte à realidade em que ele estava acostumado.

Note que a recuperação da normalidade ocorre de forma muito rápida. Isto ocorre por que provavelmente o modelo considera que os meses recentes são muito influentes na tomada de decisão.

![Previsão (K-14) 2022 e 2023](results/previsao_22_23.png)

```python
# Cálculo de métricas do modelo em relação aos dados de teste indo de 2022 e 2023

# RMSE da previsão com o melhor K
rmse_teste = root_mean_squared_error(
        y_teste[-21:],
        previsoes_teste[-21:]
    )
print(f"O RMSE do modelo com o melhor K é {rmse_teste:.4f}.\n")

# MAPE da previsão com o melhor K
mape = mean_absolute_percentage_error(y_teste[-21:], previsoes_teste[-21:])
print(f"O MAPE do modelo com o melhor K é {mape:.4f}.\n")

# R^2 da previsão com o melhor K
r2 = r2_score(y_teste[-21:], previsoes_teste[-21:])
print(f"O R^2 do modelo com o melhor K é {r2:.4f}.")
```
```txt
O RMSE do modelo com o melhor K é 20377.0329.

O MAPE do modelo com o melhor K é 0.0213.

O R^2 do modelo com o melhor K é 0.7935.
```

## b2 e b3

Tomando agora os anos de 2022 e 2023 como referência, é possível observar que o modelo previu com muito mais qualidade os dados de teste. Isso é notável ao se comparar os dados reais com os dados previstos no gráfico.

Além disso, ao se comparar as métricas de performance, conseguimos tirar conclusões bem mais embasadas. Quando se considerou todo o período de 2020 até 2023, tivemos um R^2 de 0.3582, já para apenas o período de 2022 e 2023 tivemos um R^2 de 0.7935. Esta métrica foca na explicabilidade do modelo. Sabendo que, quanto mais próximo de 1, mais o modelo explica a realidade e quanto mais próximo de 0, menos o modelo explica a realidade. Desta forma, o segundo valor está evidenciando que o modelo está muito mais fiel à realidade quando se considera o período de normalidade global pós-pandêmico. E que ele está errando muito para a realidade pandêmica. O mesmo ocorre para o RMSE e para o MAPE, que corroboram para a mesma conclusão.


## Adição de normalização dos dados

A partir daqui serão adicionados diferentes tipos de normalização para analisar como o modelo se comporta a partir desta adição desta técnica.

Para realizar este estudo iremos utilizar o método mais comum Z-Score e um outro método não tão utilizado, o MinMax.

```python
def normalizar_serie(voos, anos, metodo):
    """
    Normaliza a série temporal utilizando SOMENTE os dados
    do período de treinmaento (2003-2017)
    """

    dados_treino = voos[anos <= 2017].reshape(-1, 1)

    if metodo == "zscore":
        scaler = StandardScaler()

    elif metodo == "minmax":
        scaler = MinMaxScaler()

    # Calcula os parâmetros SOMENTE com o treinamento(2003-2017)
    scaler.fit(dados_treino)

    # Aplica os mesmos parâmetros à série inteira
    voos_normalizados = scaler.transform(voos.reshape(-1, 1)).flatten()

    return voos_normalizados, scaler
```
Print da lógica:
```txt
Para o z-score, o K ideal encontrado na validação foi K = 14
Para o minmax, o K ideal encontrado na validação foi K = 14
```

![Comparação das normalizações na validação](results/compara_normalizacao.png)

## Conclusão sobre a normalização

A partir da investigação sobre a aplicação de diferentes técnicas de normalização dos dados, é possível concluir que, para este caso específico, não existe diferença significativa entre aplicar normalização ou não. Desta forma, mantemos para o próximo item a não aplicação da normalização.

O K ideal encontrado na validação foi K = 5

![RMSE / Lags Norm](results/rmse_lags_conc_norm.png)
![Previsão K=5 22-23 Norm](results/previsao_K5_22_23.png)

```python
# Cálculo de métricas do modelo em relação aos dados de teste indo de 2020 até 2023

# RMSE da previsão com o melhor K
rmse_teste = root_mean_squared_error(
        y_teste,
        previsoes_teste
    )
print(f"O RMSE do modelo com o melhor K é {rmse_teste:.4f}.\n")

# MAPE da previsão com o melhor K
mape = mean_absolute_percentage_error(y_teste, previsoes_teste)
print(f"O MAPE do modelo com o melhor K é {mape:.4f}.\n")

# R^2 da previsão com o melhor K
r2 = r2_score(y_teste, previsoes_teste)
print(f"O R^2 do modelo com o melhor K é {r2:.4f}.")
```

```txt
O RMSE do modelo com o melhor K é 39682.9311.

O MAPE do modelo com o melhor K é 0.0478.

O R^2 do modelo com o melhor K é 0.2167.
```

## Conclusão sobre o item c
Neste item foi pedido que o conjunto de validação fosse modificado para se usar os anos de 2020 e 2021, diferenciando do que anteriormente havia sido feito, pois usamos os anos 2018 e 2019 como validação.

A partir desta mudança foi possível observar o quanto o conjunto de validação influencia na arquitetura do modelo. No caso do item b, foi utilizado um período em que a série temporal ainda estava se comportando de forma normal. O período era o mesmo para toda a mostra de 2003 até 2019, assim, temos que o conjunto de treinamento e validação estavam se comportando da mesma maneira, ou seja, existia uma certa previsibilidade.

No caso do item c, incluímos no treinamento e validação algo que pode ser considerado como uma interferência ou um ruído na série temporal. Podemos considerar que o que aconteceu em 2008 também foi algo que pode ser considerado como um ruído, porém, além de ser muito menor em comparação à quarentena, ele se perde frente ao espaço amostral fornecido ao modelo.

A mudança do conjunto de validação afetou o modelo de forma que o K ideal ficou em torno de valores muito menores, o que pode ser interpretado como o modelo se tornando mais cauteloso e mais desconfiado dos dados antigos, visto que estes dados não forneceram informação suficiente para que o modelo pudesse prever o que estaria ocorrendo no conjunto de validação.

Outra mudança foi a qualidade das previsões, que pode ser evidenciada pelas métricas de desempenho do modelo. O R^2 ficou muito próximo de 0, o que permite interpretar que o modelo não foi capaz de prever de forma efetiva. Porém, o MAPE ficou muito próximo de zero, mostrando que os valores previstos não ficaram tão distantes da realidade. Isto tudo permite concluir que, por mais que o modelo tenha sido validado com uma amostra que não representa a série temporal por inteiro, o K escolhido permitiu que o modelo ainda performasse de forma satisfatória.

## Análise Adicional com Transformada de Fourier
O problema em questão se baseia na sazonalidade dos dados, ou seja, como ele se comporta a partir da passagem do tempo. Por ser um espectro temporal com comportamento que se repete com o tempo, nada mais natural que se considerar uma análise a partir da Transformada de Fourier. A partir disto, visa-se compreender o comportamento recorrente do espectro e os resultados do modelo de Regressão Linear adotado.

Porém, tendo em vista que o ano de 2020 possui uma quebra na sazonalidade do espectro, iremos realizar a análise apenas até o ano de 2019.

```python
# FFT somente de 2003 até 2019

# Seleciona os dados até dezembro de 2019
idx_2019 = df["Year"].to_numpy() <= 2019

voos_2019 = voos[idx_2019]
voos_2019 = voos_2019 - np.mean(voos_2019)

N_2019 = len(voos_2019)

# FFT
fft_2019 = np.fft.fft(voos_2019)

# Frequências
frequencias_2019 = np.fft.fftfreq(
    N_2019,
    d=1
)

# Magnitude
magnitude_2019 = np.abs(fft_2019)

# Apenas frequências positivas
idx = frequencias_2019 > 0

frequencias_2019 = frequencias_2019[idx]
magnitude_2019 = magnitude_2019[idx]

# Converter para período
periodos_2019 = 1 / frequencias_2019

# Considerar períodos até 25 meses
idx_periodo = periodos_2019 <= 25

plt.figure(figsize=(10, 5))

plt.plot(periodos_2019[idx_periodo], magnitude_2019[idx_periodo])

plt.title("Espectro da série de tráfego aéreo de 2003 a 2019")
plt.xlabel("Período (meses)")
plt.ylabel("Magnitude")
plt.grid(True)

plt.show()
```

![Espectro FFT](results/espectro_fft.png)

A partir da análise do espctro tranformado pela FFT, é possível observar um pico de magnitude da sazonalidade por volta de 12 meses. Isto serve para evidenciar que o espectro temporal possui período de aproximadamente 12 meses, o que faz sentido quando se considera o valor do lag(K), o qual alcança uma significativa melhora quando passa de 12 para 13, alcançando seu valor mínimo de RMSE em 14. Desta forma, a conjectura de que o valor ótimo de K está relacionado com o período em que o espectro se repete ganha mais força.


## Referências
1. Documentação do scikit-learn. https://scikit-learn.org/
2. International air passenger fares shrug off the recession - U.S. Bureau of Labor Statistics. https://www.bls.gov/opub/btn/volume-1/international-air-passenger-fares-shrug-off-the-recession.htm Acesso em 07/09/2026