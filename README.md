# Vinculação de endereços a setores censitários — CNEFE Bahia

Solução em Python para localizar o setor censitário de endereços usando o CNEFE
da Bahia como referência. A saída preserva os registros e valores originais da
base de busca e acrescenta uma decisão: `FOUND`, `AMBIGUOUS` ou `NOT_FOUND`.

## Arquitetura

```text
CNEFE CSV → ingestão Spark → Parquet → normalização Spark → Parquet normalizado
                                                               ↓
                                                       indexação bulk
                                                               ↓
                                                         Elasticsearch

Busca CSV → normalização Pandas → baseline exata Spark → recuperação de candidatos
                                                               ↓
                                                      seleção e consolidação
                                                               ↓
                                                      validação → CSV final
```

A preparação do CNEFE e a indexação são executadas separadamente. O pipeline
principal reutiliza o Parquet normalizado e o índice persistente. Spark funciona
em modo local dentro da aplicação; Elasticsearch funciona em um serviço próprio.

| Componente | Versão / função |
|---|---|
| Python | 3.11.16; CLI, regras de decisão e exportação |
| Java | OpenJDK 21; execução do Spark |
| PySpark | 4.2.0; processamento do CNEFE e baseline, padrão `local[4]` |
| Pandas | 2.3.3; processamento da base pequena de busca |
| Elasticsearch | Servidor 9.5.2 e cliente Python 9.5.0; recuperação de candidatos |
| Docker Compose | Ambiente da aplicação e Elasticsearch single-node |

## Estrutura e dados de entrada

```text
.
├── src/                  # ingestão, normalização, indexação, matching e exportação
├── README.md
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .dockerignore
└── .gitignore
```

Os dados reais não estão incluídos e não são baixados automaticamente. Para
reproduzir o desafio, obtenha os arquivos de entrada fornecidos pelo responsável
pela avaliação e coloque-os na mesma pasta deste README:

```text
29_BA.csv
2026_teste_tecnico_relinke_cidacs(in).csv
```

Os CSVs usam UTF-8 e separador `;`. O schema completo do CNEFE está definido em
`src/ingestion/schemas.py`; a ingestão confere os nomes e a ordem das 34 colunas,
filtra `COD_UF=29` e mantém 23 colunas relevantes. `COD_SETOR` é preservado como
texto. A busca exige as colunas `consulta_logradouro`, `consulta_numero`,
`consulta_bairro`, `consulta_cep`, `consulta_complemento` e `consulta_municipio`.
Colunas adicionais da busca são preservadas, desde que não usem nomes reservados
para campos gerados, como `search_id` e `cep_normalizado`. Colisões são rejeitadas
com uma mensagem de erro, sem sobrescrever valores originais. O volume esperado
é de 107 registros e pode ser configurado.

Durante a execução, os comandos criam `data/processed/`, `data/output/` e `docs/`
dentro deste projeto. São pastas de saída: seu conteúdo é gerado a partir dos
arquivos de entrada fornecidos pelo usuário.

## Pré-requisitos e configuração

- Docker com suporte a containers Linux e Docker Compose 2.24 ou superior.
- Aproximadamente 8 GB de memória disponíveis para o Docker.
- Pelo menos 8 GB livres em disco, além de espaço para os arquivos de entrada.
  O uso efetivo depende do volume processado.

Python, Java e Spark são instalados na imagem da aplicação. O Elasticsearch tem
heap padrão de 1 GiB, limite de container de 2 GiB e porta `9200` publicada somente
em `127.0.0.1`. Se outra instância já estiver usando essa porta, pare-a antes de
iniciar este Compose.

O uso de `.env` é opcional: os valores padrão estão no código. Para configurá-lo,
copie `.env.example` para `.env` e ajuste somente as opções necessárias. Exemplo
para entradas com outros nomes, armazenadas dentro desta pasta:

```dotenv
CNEFE_INPUT_PATH=data/raw/cnefe.csv
SEARCH_INPUT_PATH=data/raw/busca.csv
EXPECTED_SEARCH_RECORDS=107
```

Coloque os arquivos nos caminhos escolhidos, dentro deste projeto. Caminhos
relativos são resolvidos pelo diretório que contém este README. O Compose monta
esse diretório em `/app`, disponibilizando os arquivos aos containers.
Entradas e saídas devem usar caminhos distintos; a configuração também rejeita
um artefato dentro do caminho de outro, como dois Parquets sobrepostos.

O Compose carrega `.env` quando presente e define `ELASTICSEARCH_URL` como
`http://elasticsearch:9200` dentro da aplicação. O valor `localhost` do exemplo
se aplica à execução Python no host, que precisa receber as variáveis pelo shell.
`SPARK_MASTER=local[4]` permite até quatro tarefas Spark simultâneas.

## Primeira execução

Após obter o repositório e posicionar os dois CSVs, execute os comandos abaixo no
diretório que contém este README, `Dockerfile` e `docker-compose.yml`:

```bash
docker compose config --quiet
docker compose build app
docker compose up -d --wait elasticsearch
docker compose run --rm app python -m src.main check-elasticsearch
docker compose run --rm app python -m src.main validate-search
docker compose run --rm app python -m src.main prepare-cnefe
docker compose run --rm app python -m src.main normalize-data
docker compose run --rm app python -m src.main index-cnefe
docker compose run --rm app python -m src.main validate-index
docker compose run --rm app python -m src.main run-pipeline
```

A ingestão grava o CNEFE selecionado em Parquet. `normalize-data` normaliza as duas
bases, preserva os campos originais e gera o relatório de impacto. `index-cnefe`
envia os campos necessários em lotes ao Elasticsearch e registra a carga.
`validate-index` verifica consultas técnicas. `run-pipeline` renormaliza a busca
pequena, executa baseline, retrieval, selection e exportação.

Para gerar a avaliação dos resultados após o pipeline:

```bash
docker compose run --rm app python -m src.main evaluate-results
```

A preparação inicial do CNEFE exige mais tempo e recursos do que o matching.
Parquets preservam tipos e permitem leitura colunar nos processamentos seguintes.

## Execuções seguintes e persistência

Com o Parquet normalizado, o relatório de indexação e o índice já preparados:

```bash
docker compose up -d --wait elasticsearch
docker compose run --rm app python -m src.main run-pipeline
```

O pipeline verifica existência dos pré-requisitos, versão do mapping, completude
da carga bulk e contagem do índice. A preparação e a indexação devem ser refeitas
quando a fonte ou as regras correspondentes mudarem. A avaliação pode ser
executada novamente depois do pipeline.

O índice fica no volume Docker nomeado `elasticsearch-data`, normalmente
prefixado pelo nome do projeto Compose. Ele não está na pasta do repositório nem
acompanha uma cópia ou clone. Um ambiente novo precisa criar seu próprio índice.
`docker compose stop` e `docker compose down` preservam o volume;
`docker compose down -v` o remove e exige nova indexação.

A reconstrução de um índice existente exige a opção explícita abaixo, que apaga
seu conteúdo antes da nova carga:

```bash
docker compose run --rm app python -m src.main index-cnefe --recreate-index
```

## Matching e saída

A normalização padroniza caixa, acentos, espaços, pontuação e algumas abreviações
observadas, mantendo componentes estruturados e valores originais. A baseline
tenta chaves exatas de município, CEP, logradouro e número, com fallbacks próprios
para endereços sem número. Resultados exatos `FOUND` e `AMBIGUOUS` são preservados
na consolidação.

O retrieval consulta todos os registros para diagnóstico, com até três estratégias
progressivas e limite padrão de dez candidatos únicos. Fuzzy é limitado ao
logradouro. O selector compara novamente município, logradouro, número, CEP e
bairro/localidade. A política usa similaridade mínima de logradouro de 0,90 para
evidência forte e 0,85 para plausível, combinada com concordâncias estruturadas.
O `_score` é registrado para auditoria e não decide sozinho o setor.

| Status | Significado |
|---|---|
| `FOUND` | A regra exata ou a evidência forte apoia um único setor. |
| `AMBIGUOUS` | Existem candidatos plausíveis sem evidência suficiente ou setores conflitantes. |
| `NOT_FOUND` | Não existem candidatos ou a evidência recuperada é fraca. |

O setor é preenchido somente para `FOUND`. Antes da exportação, o código confere
cardinalidade, identificadores, ordem, valores originais e coerência entre setor e
status. Os CSVs usam UTF-8, separador `;` e não possuem índice Pandas adicional.

| Arquivo gerado | Conteúdo |
|---|---|
| `data/output/result.csv` | Colunas originais mais `search_id`, `setor_censitario`, `status_match` e `metodo_match` |
| `data/output/result_audit.csv` | Campos normalizados, baseline, proposta Elasticsearch, decisão e evidências |
| `data/output/result_metrics.json` | Contagens por status, cobertura e validações da exportação |
| `data/processed/normalization_report.json` | Impacto da normalização e colisões |
| `data/output/elasticsearch_index_report.json` | Operações bulk, falhas e quantidade de documentos indexados |

O pipeline também gera arquivos intermediários de baseline, candidatos e
evidências em `data/output/`, além de novos relatórios de retrieval e matching em
`docs/`. `evaluate-results` gera avaliação por caso, casos de baixa confiança,
métricas e o relatório `docs/evaluation.md`. Todos esses caminhos ficam dentro
deste projeto; os arquivos são produzidos pela execução e ignorados pelo Git.
O relatório compara baseline e resultado final da mesma execução, não versões
históricas. A ausência de candidatos é válida e gera evidências com cabeçalho,
mesmo quando não há linhas de candidatos para avaliar.
Os logs agregados são apresentados no terminal.

## Resultados

### Execução de referência

Uma execução registrada em **09/10/2026**, com os dados do desafio, apresentou
107 registros de entrada e 107 de saída. As métricas de baseline, seleção e
exportação registraram:

| Etapa | FOUND | AMBIGUOUS | NOT_FOUND | Cobertura |
|---|---:|---:|---:|---:|
| Baseline exata | 6 | 16 | 85 | 5,61% |
| Pipeline completo | 11 | 45 | 51 | 10,28% |

Dos 11 endereços classificados como `FOUND`, seis vieram da baseline e cinco
foram aceitos pela seleção estruturada após a recuperação no Elasticsearch.
As 16 ambiguidades exatas foram preservadas; outros 29 casos tinham candidatos plausíveis, mas
evidências insuficientes para uma decisão segura. Dos 51 `NOT_FOUND`, 35 ficaram
sem candidatos e 16 tiveram candidatos com evidências insuficientes.

Esses números são referência histórica, não validação da cópia portátil após
as correções. A versão entregue ainda precisa de uma execução completa para
confirmar suas métricas atuais. Os arquivos daquela execução não acompanham
a entrega; não são pré-requisitos. Não há ground truth para concluir que o
aumento de cobertura representou aumento de precisão.

### Como interpretar uma nova execução

Esta entrega não inclui arquivos de resultados pré-calculados. Após executar
`run-pipeline`, `data/output/result.csv` contém uma linha para cada endereço de
entrada, na ordem original. O setor censitário é preenchido apenas nos casos
`FOUND`; nos demais, fica vazio e `status_match` distingue ausência de evidência
de ambiguidade.

Um `AMBIGUOUS` pode representar candidatos em setores diferentes ou candidatos
em um único setor sem evidências suficientes para aceitar a vinculação. Já
`NOT_FOUND` não prova que o endereço inexiste no CNEFE: ele pode não ter sido
recuperado pelas consultas ou ter recebido somente evidências fracas.

As contagens da execução ficam em `data/output/result_metrics.json`:

- `input_records` e `output_records`: quantidade de entradas e saídas, que deve
  ser igual;
- `found`, `ambiguous` e `not_found`: quantidade de endereços em cada status;
- `found_percentage`: cobertura, calculada como `found / input_records * 100`.

O arquivo `data/output/candidate_selection_metrics.json` permite comparar a
baseline exata com a decisão final da mesma execução e identificar quantos
`FOUND` adicionais vieram da recuperação Elasticsearch e da seleção estruturada.
Para investigar uma decisão, use `search_id` no arquivo `result_audit.csv` e
`input_id` no arquivo `candidate_selection_evidence.csv`, ambos em `data/output/`.
Eles registram método, candidatos e concordâncias entre componentes do endereço.

Percentual de encontrados não é acurácia: sem uma base rotulada de referência,
não é possível medir quantos setores atribuídos estão realmente corretos. Um
score alto ou mais casos `FOUND` não demonstram, sozinhos, melhora de precisão.

## Limitações

- Não existe ground truth legítimo incluído: percentual `FOUND` mede cobertura,
  e não acurácia. As métricas devem ser obtidas pela execução com os dados.
- O limite de candidatos pode ocultar setores adicionais; ambiguidades exatas
  conhecidas permanecem preservadas.
- Bairro da busca e localidade do CNEFE fornecem evidência adicional, mas não são
  equivalentes semânticos em geral.
- Endereços rurais, sem número ou com componentes misturados continuam difíceis.
- Elasticsearch é single-node, sem autenticação e destinado ao uso local.
- As versões diretas estão fixadas; dependências transitivas e a atualização
  exata do pacote Java não estão travadas para builds idênticos byte a byte.
