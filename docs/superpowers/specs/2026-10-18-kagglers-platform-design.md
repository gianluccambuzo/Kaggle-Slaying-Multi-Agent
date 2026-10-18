# Kagglers: plataforma de agentes para competições do Kaggle — Design

Data: 2026-10-18
Status: aguardando revisão do usuário

## 1. Objetivo e escopo

Construir uma **plataforma reutilizável** em que um orquestrador determinístico, com um LLM apenas como proponente de hipóteses, participa de forma autônoma de competições do Kaggle: coleta dados, treina e avalia candidatos, envia submissões pela CLI oficial e monitora o leaderboard.

- **Caso de teste:** Playground Series S6E10 "Predicting Airline Satisfaction" (prazo 31/out/2026).
- **Produto real:** a plataforma, para ser reaproveitada depois na competição "PTCG AI Battle Challenge Playground" (submissão `.tar.gz`, 5 por dia, prazo 8/jan/2027).
- **Meta em S6E10:** top 20% é desejável, mas não é critério de sucesso do projeto. O critério é o ciclo completo funcionar de ponta a ponta com uma submissão válida.
- **Fora do escopo deste spec:** o pack do Pokémon TCG (só a interface é desenhada para ele), execução em containers, integração com MLflow.

### Fatos sobre a competição (NÃO oficiais — vieram de repositórios da comunidade)

Verificar na página (Overview, Evaluation, Rules) antes de implementar o que depende deles:

| Item | Valor reportado | Depende disso |
|---|---|---|
| Métrica | ROC AUC | `Evaluator` |
| Tamanho | ~700 mil linhas de treino, ~300 mil de teste, 21 features | orçamento de compute |
| Limite diário | 10 (padrão do Playground costuma ser 5) | `Pack.rules` |
| Dado original como extra | usado por alguns, regra não confirmada | operador desligado por padrão |
| Features que usam o teste | regra não confirmada | desligado por padrão |
| Regras sobre automação | não verificadas | decisão de seguir |

Referência de scores públicos (auto-reportados): baseline Random Forest ≈ 0,956 AUC; melhores blends ≈ 0,9613.

## 2. Premissas de design

1. Validação confiável primeiro; o leaderboard público nunca decide seleção de modelos.
2. Baseline forte e rápido antes de qualquer sofisticação.
3. O LLM propõe, o código decide e mede. O orquestrador não é um agente LLM livre.
4. Código gerado roda em sandbox com limites.
5. Tetos rígidos de custo (LLM, compute) e de submissões por dia.
6. Estado persistido, execução retomável.

Evidência da pesquisa (AIDE, AutoKaggle, MLE-STAR, KompeteAI): o ganho vem de busca guiada por validação e de boas ferramentas, não do framework de agentes. Taxas de medalha em benchmarks podem estar infladas em relação ao Kaggle real.

## 3. Arquitetura

Abordagem escolhida: **núcleo genérico + "Competition Pack" por competição**, em Python, sem framework de agentes. A estratégia de busca (candidatos como nós de uma árvore, estilo AIDE) vive dentro do pack.

```
kagglers/
  core/
    orchestrator.py   # máquina de estados + critério de parada
    budget.py         # LLM US$, compute (min), submissões/dia
    store.py          # SQLite: experimentos, candidatos, submissões
    sandbox.py        # roda código gerado com timeout/memória
    kaggle_client.py  # wrapper da CLI: download, submit, leaderboard
    llm.py            # único ponto de chamada ao LLM (SDK Anthropic)
    types.py          # Candidate, Score, Artifact, Hypothesis
  packs/
    airline_s6e10/    # data, evaluator, packager, operadores, prompts
    (ptcg/ no futuro)
  runs/               # artefatos e logs por execução
```

### Interfaces que o pack implementa

- `Evaluator.evaluate(candidate) -> Score`: `value`, `std`, `n_runs` e o artefato de predição (OOF e teste). O núcleo vê só "candidato → score". No Pokémon, o mesmo contrato roda partidas simuladas em vez de CV.
- `Packager.build(candidates) -> Artifact`: gera o CSV (ou `.tar.gz`) e valida o formato antes do envio.
- `Operator`: transformação aplicada a um candidato (adicionar feature, trocar hiperparâmetros, rodar ablação). Schema de entrada fixo, executa no sandbox.
- `Pack.rules`: limites declarados (submissões/dia, dados externos permitidos). O orçamento lê daqui.

### Decisões embutidas

1. Estado no SQLite, não em memória; o orquestrador é retomável.
2. Só `llm.py` chama o LLM (contagem de custo, teto, troca de modelo).
3. O LB nunca entra no `Score` usado para decidir; é metadado de monitoramento.

## 4. Ciclo do orquestrador

LLM só nos passos marcados com 🧠.

```
INIT → PREPARE → BASELINE → PROPOSE 🧠 → EXECUTE → EVALUATE → DECIDE
                                ↑                                 │
                                └───────────── (loop) ────────────┤
                                                                  ↓
                            ENSEMBLE → SUBMIT → MONITOR → STOP
```

1. **PREPARE:** baixa dados, valida esquema, gera folds fixos e salva `fold_id` por linha.
2. **BASELINE:** LightGBM padrão, 5 folds; define a referência e prova o ciclo de ponta a ponta antes de gastar LLM.
3. **PROPOSE 🧠:** o LLM recebe o histórico resumido do `store` e devolve 1..N hipóteses estruturadas (operador + parâmetros). Duplicatas são descartadas por `config_hash`.
4. **EXECUTE:** cada hipótese roda no sandbox; falha ou timeout vira registro `failed` sem derrubar o ciclo.
5. **EVALUATE:** CV via `Evaluator`; promoção só se passar o limiar de ruído (seção 5).
6. **DECIDE:** mantém, descarta ou promove ao conjunto do ensemble; checa orçamento e parada.
7. **ENSEMBLE:** blending sobre OOF dos promovidos, pesos ajustados só no OOF.
8. **SUBMIT:** gera e valida o artefato; envia somente se o CV do ensemble superar o da última submissão por uma margem e houver cota.
9. **MONITOR:** lê o leaderboard, calcula o percentil, compara CV vs LB. Divergência grande gera alerta e **não altera a seleção**.

### Orçamento e parada

- Tetos rígidos: LLM em US$, minutos de compute, submissões/dia (de `Pack.rules`). Estouro → STOP com motivo registrado.
- Saturação: N rodadas seguidas de PROPOSE sem ganho acima do limiar.
- Meta: percentil ≤ 20% e CV estável (percentil é informativo; não aciona seleção).
- Reserva: pelo menos 1 submissão/dia é guardada para a execução final.

### Erros

- Falha de sandbox/operador: registra e segue.
- Rede na CLI do Kaggle: retry com backoff; depois STOP com estado salvo.
- Submissão rejeitada: o status final lido em `submissions` (`ERROR`/falha) é gravado e a execução para com o motivo `submission_rejected` (status e descrição impressos), sem retry cego; retomar a mesma execução não reenvia, é preciso uma nova execução.
- O `submit` nunca é repetido às cegas: num erro transitório (5xx, timeout) o cliente consulta `submissions` pela mensagem; se ela aparece, conta como enviada; se não, devolve o erro transitório sem segundo upload (a retomada decide).

### Comandos da CLI do Kaggle (`kaggle>=2.2,<3`)

- `kaggle competitions download <c> -p <dest>`: grava `<dest>/<c>.zip` (a CLI 2.x não tem `--unzip`); o `kaggle_client` extrai o zip em `<dest>` recusando caminhos fora de `<dest>`. Repetido com backoff em erro transitório.
- `kaggle competitions submit <c> -f <arquivo> -m <mensagem>`: chamado uma única vez (a CLI 2.x não tem `--wait`). Depois o cliente consulta `kaggle competitions submissions <c> -v` até a entrada com `description == mensagem` ter status final, com limite de tentativas (padrão 60 × 10 s). Status (`SubmissionStatus.COMPLETE`, `COMPLETE`, `ERROR`, `PENDING`...) é classificado sem diferenciar maiúsculas: contém "complete" → concluída; contém "error" ou "fail" → rejeitada; senão → pendente. Se ainda pendente ao fim das tentativas, a submissão fica `pending` e o MONITOR grava o percentil como nulo.
- `kaggle competitions submissions <c> -v`: CSV lido por nome de coluna (`ref`, `description`, `status`, `publicScore`).
- `kaggle competitions leaderboard <c> -d -p <dest>`: baixa o zip do leaderboard, extraído do mesmo modo seguro.
- Cada chamada tem timeout (padrão 1800 s, configurável), saída lida em UTF-8 (`PYTHONIOENCODING=utf-8`, caracteres inválidos substituídos); timeout conta como erro transitório, exceto no `submit`, que passa pela consulta acima.

## 5. Estratégia de validação (pack S6E10)

- **Folds:** `StratifiedKFold` 5 folds, gerado uma vez em PREPARE, `fold_id` salvo por linha e compartilhado por todos os candidatos (comparações pareadas).
- **Seeds:** o `fold_id` é fixo; a variação vem da seed do modelo. Funil de dois estágios: triagem com 1 seed; quem passa um limiar frouxo roda 3 seeds.
- **Limiar de ruído:** diferença de AUC pareada por fold contra o melhor atual. Promove se a média das diferenças > 2 × erro-padrão **e** positiva em ≥ 4 de 5 folds. O fator 2 é parâmetro do pack, calibrado no baseline com seeds idênticas.
- **Leakage:**
  - Target encoding e qualquer estatística aprendida da label são ajustados dentro de cada fold de treino (operador com contrato fixo; o sandbox rejeita ajuste de encoders no conjunto completo).
  - Features que usam o teste (frequências) ficam desligadas por padrão até as regras permitirem.
  - Adversarial validation rápida treino vs teste em PREPARE; AUC adversarial muito acima de 0,5 gera alerta.
  - Dado original como extra: desligado até as regras permitirem; se permitido, entra só no treino de cada fold, nunca na validação.
- **Ensemble:** nested CV simples (pesos ajustados em 4 folds, medidos no quinto).
- **LB público:** só alimenta o MONITOR; `|CV − LB|` acima da tolerância gera aviso, nunca muda a seleção.

## 6. Sandbox e segurança

- Subprocesso `python -I` em diretório próprio; dados em modo somente leitura (na fase 1 isso não é imposto pelo sandbox, ver "Desvios" no plano).
- Layout real dos artefatos (raiz = diretório atual): o operador roda com cwd `runs/<exp_id>/` (lá fica `operator_config.json`) e grava `oof.npy`, `test.npy` e `metrics.json` em `runs/<exp_id>/out/`; o orquestrador grava o manifesto do score em `runs/<run_id>/<exp_id>/score.json`, o CSV em `runs/<run_id>/submission/submission.csv` e o leaderboard em `runs/<run_id>/leaderboard/`; o banco é `runs/kagglers.db` e os dados ficam em `data/<competition>/` (com `folds_k{n}_s{seed}.npy`).
- Limites de timeout, memória e CPU configuráveis no pack. No Windows, Job Objects (`psutil`/`pywin32`) limitam memória e matam a árvore de processos no timeout. O Windows não tem o isolamento do Linux; para código não confiável o ideal é Docker/WSL2, deixado como evolução.
- Contrato: o operador recebe um JSON de config, lê os folds e devolve OOF e teste em `.npy`/`.parquet` mais um JSON de métricas. O sandbox valida o formato.
- Sem rede no código de operador; download e submissão passam só pelo `kaggle_client`, no processo do orquestrador.
- Segredos (token do Kaggle, chave do LLM) em variáveis de ambiente, nunca repassados ao subprocesso.
- Análise estática antes de executar, para recusar ajuste de encoders fora do contrato.

## 7. Memória de experimentos (SQLite)

- `runs`: `run_id`, `competition`, `state` (último estado concluído, padrão `INIT`), início/fim, motivo de parada. (O esquema ainda tem uma coluna `config_hash` em `runs`, não usada.)
- `experiments`: `exp_id`, `run_id`, `parent_id`, `hypothesis`, `operator`, `config_json`, `config_hash` (único por `(run_id, config_hash)`, regra R2), `status` (`running` ao começar; depois `ok`/`failed`/`timeout`), `cv_mean`, `cv_std`, `seeds`, `delta_vs_best`, `promoted`, `cost_usd`, `compute_s`.
- `submissions`: `sub_id`, `exp_id`, `kaggle_ref`, `message`, `cv_at_submit`, `lb_public`, `status`, `submitted_at`.
- `leaderboard_snapshots`: `run_id`, `ts`, `n_teams`, `our_score`, `percentil` (nulo quando o score público não está disponível).

O PROPOSE recebe um resumo dessas tabelas (o que foi tentado, ganho de cada tentativa, falhas). MLflow fica de fora; o schema permite exportar depois.

## 8. Escopo da fase 1, testes e critério de pronto

**Fase 1:** `core/` (store, kaggle_client, budget, sandbox, orchestrator em modo mínimo) e `packs/airline_s6e10/` (PREPARE, BASELINE LightGBM 5 folds, `Packager`). PROPOSE com LLM **não** entra. O ciclo para depois do BASELINE, faz uma submissão e uma leitura do leaderboard.

**Pronto quando:**
1. `kagglers run airline_s6e10 --phase1` baixa os dados, gera os folds, treina o baseline, imprime o AUC de CV e valida o CSV.
2. A submissão é enviada uma vez, o status é acompanhado por `submissions` até ficar final, o `kaggle_ref` é gravado e o MONITOR grava o percentil.
3. Interromper no meio e rodar de novo retoma do estado salvo.

**Testes (pytest, TDD):**
- Unitários: `budget` (estouro de cada teto), `store` (unicidade de `(run_id, config_hash)`), validador de formato do CSV, percentil sobre CSV de leaderboard de exemplo, geração e estabilidade dos folds.
- Integração sem rede: `kaggle_client` é uma interface com implementação real e um fake que lê arquivos locais; o fluxo completo roda contra o fake e um dataset sintético pequeno.
- Sandbox: teste de timeout e de estouro de memória (processo morto, orquestrador continua).
- A chamada real ao Kaggle é verificação manual.

## 9. Fases seguintes (fora deste plano)

2. Orquestrador completo com PROPOSE 🧠 e registro de experimentos.
3. Operadores de features, tuning e ensemble.
4. Monitoramento e iteração até o top 20% (se o prazo permitir).
5. Pack do PTCG reaproveitando o núcleo.

## 10. Pendências a resolver antes do plano

- Confirmar na página oficial: métrica, prazo, limite diário, regras de automação, dados externos e features que usam o teste.
- Configurar o token do Kaggle e aceitar as regras da competição na conta.
- Definir os tetos numéricos de orçamento (US$ de LLM, minutos de compute) para a primeira execução.
