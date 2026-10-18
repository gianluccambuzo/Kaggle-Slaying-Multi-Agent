# Kagglers Fase 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Entregar o esqueleto da fase 1: `kagglers run airline_s6e10 --phase1` baixa os dados, gera folds, treina o baseline LightGBM (5 folds) no sandbox, valida e envia a submissão, lê o leaderboard e é retomável.

**Architecture:** Núcleo genérico (`kagglers/core`) com store SQLite, orçamento, sandbox, cliente Kaggle e orquestrador em máquina de estados; um pack (`kagglers/packs/airline_s6e10`) implementa `Evaluator`, `Packager` e o operador baseline. Sem LLM e sem framework de agentes nesta fase. A CLI real do Kaggle fica atrás de uma interface com um fake para testes sem rede.

**Tech Stack:** Python, SQLite (stdlib), pandas, numpy, pyarrow, scikit-learn, lightgbm, psutil, `kaggle` (CLI), pytest.

**Spec:** `docs/superpowers/specs/2026-10-18-kagglers-platform-design.md`

## Global Constraints

- Pacote Python `kagglers` na raiz do repositório, com `core/` e `packs/`; testes em `tests/`.
- Ambiente: `.venv` na raiz, hoje Python 3.14. Se `lightgbm` não instalar nessa versão (Task 1, Step 3), recriar o `.venv` com Python 3.12 e fixar `requires-python = ">=3.12"`.
- Só `core/llm.py` chama o LLM; ele **não** é criado na fase 1.
- O LB público nunca entra no `Score` usado para decidir; é gravado só como metadado.
- Folds: `StratifiedKFold` com 5 folds, geração única, `fold_id` por linha, compartilhado por todos os candidatos.
- Código de operador roda em subprocesso `python -I`, sem rede e sem segredos; token do Kaggle e chave de LLM só no processo do orquestrador.
- Limite diário de submissões lido de `PackRules`; padrão conservador de **5** (o valor 10 reportado pela comunidade não foi confirmado). Reserva de 1 submissão/dia para a execução final.
- Métrica: ROC AUC (não oficial; confirmar na página). Maior é melhor.
- Dado original como extra e features derivadas do teste: **desligados** em `PackRules`.
- Cada chamada real ao Kaggle é verificação manual; nenhum teste automático usa rede.
- Commits em inglês e **sem** linha `Co-Authored-By` (nem outra atribuição ao Claude).

## Desvios e esclarecimentos em relação à spec

- A tabela `runs` ganha a coluna `state` (necessária para retomar). A spec deve receber a mesma coluna (Task 2, Step 4).
- Limite de memória do sandbox é imposto por **polling de RSS da árvore de processos via psutil** (não Job Objects/pywin32). Limite de CPU = teto de threads (`OMP_NUM_THREADS`/`n_jobs`).
- A análise estática anti-leakage do sandbox fica para a fase 2, quando houver código gerado por LLM. O baseline é código nosso.
- Calibração do fator 2× de ruído, nested CV do ensemble e PROPOSE ficam para a fase 2 ou 3. A fase 1 só registra o `cv_std` e os scores por fold.
- Os nomes das colunas do dataset ainda não foram confirmados. O pack lê `id_col` e `target_col` da 1ª e 2ª colunas de `sample_submission.csv`; `positive_label` é config explícita (ver Review Focus).

## Review Focus

1. Rótulos de treino em texto (ex.: `"satisfied"`) e `positive_label` não configurado: o PREPARE deve falhar com mensagem listando os valores, nunca inverter o AUC em silêncio. (Task 6)
2. Categorias do teste ausentes no treino e valores nulos: o treino e o packager não podem quebrar nem produzir NaN no CSV. (Tasks 6 e 7)
3. Submissão rejeitada por regras não aceitas (HTTP 403) ou por erro de formato: erro claro, sem retry cego. Falha de rede: retry com backoff e depois STOP com estado salvo. (Task 4)
4. `id` do CSV diferente do `sample_submission` (ordem, faltantes, duplicados): o validador rejeita antes do envio. (Task 7)
5. Queda entre o sucesso da CLI e a gravação no banco: ao retomar, não pode reenviar. A mensagem da submissão carrega o `exp_id` e a retomada consulta `submissions` do Kaggle antes de enviar. (Task 8)

Também cobertos: cota diária esgotada → STOP sem enviar (Task 3); leaderboard vazio ou com empates (Task 4).

---

### Task 1: Scaffold, tipos e protocolos

**Files:**
- Create: `pyproject.toml`, `.gitignore`, `kagglers/__init__.py`, `kagglers/core/__init__.py`, `kagglers/packs/__init__.py`, `kagglers/core/types.py`, `tests/__init__.py`, `tests/test_types.py`

**Interfaces:**
- Produces, em `kagglers/core/types.py`:
  - `@dataclass(frozen=True) Hypothesis(text: str, operator: str, params: dict)`
  - `@dataclass(frozen=True) Candidate(exp_id: str, operator: str, params: dict)` com propriedade `config_hash: str` (sha256 de JSON canônico de `operator` + `params`, 16 hex)
  - `@dataclass(frozen=True) Score(value: float, std: float, n_runs: int, fold_scores: list[float], oof_path: Path, test_path: Path)`
  - `@dataclass(frozen=True) Artifact(path: Path, kind: str)`
  - `@dataclass(frozen=True) PackRules(metric_name: str, higher_is_better: bool, daily_submission_limit: int, allow_external_data: bool, allow_test_derived_features: bool)`
  - `Protocol Evaluator`: `evaluate(self, candidate: Candidate) -> Score`
  - `Protocol Packager`: `build(self, score: Score, out_dir: Path) -> Artifact`
  - `Protocol Pack`: atributos `name: str`, `competition: str`, `rules: PackRules`, `evaluator: Evaluator`, `packager: Packager`, método `prepare(self) -> None`

- [ ] **Step 1: Write the failing test** em `tests/test_types.py`: `test_config_hash_is_stable_across_param_order` (`Candidate("a","baseline",{"x":1,"y":2}).config_hash == Candidate("b","baseline",{"y":2,"x":1}).config_hash`, independente de `exp_id`), `test_config_hash_differs_on_params`, `test_config_hash_has_16_hex_chars`.
- [ ] **Step 2: Run** `pytest tests/test_types.py -v` → FAIL (`ModuleNotFoundError`).
- [ ] **Step 3: Create** `pyproject.toml` (build `setuptools`, pacotes `kagglers*`, dependências listadas em Tech Stack, extra `dev` com pytest, script `kagglers = "kagglers.cli:main"`), `.gitignore` (`.venv/`, `data/`, `runs/`, `__pycache__/`, `.env`, `.idea/`) e `types.py` como acima. Instalar com `pip install -e ".[dev]"`.
- [ ] **Step 4: Verify the toolchain**: `python -c "import lightgbm, sklearn, pandas, pyarrow, psutil"` deve passar. Se `lightgbm` falhar, aplicar a regra de Global Constraints (Python 3.12) antes de seguir.
- [ ] **Step 5: Run** `pytest tests/test_types.py -v` → PASS.
- [ ] **Step 6: Commit** `feat: scaffold package and core types`.

---

### Task 2: Store (SQLite)

**Files:**
- Create: `kagglers/core/store.py`, `tests/test_store.py`
- Modify: `docs/superpowers/specs/2026-10-18-kagglers-platform-design.md` (seção 7: adicionar `state` a `runs`)

**Interfaces:**
- Consumes: `Candidate` (Task 1).
- Produces, em `store.py`, classe `Store(db_path: Path)` (cria o schema se ausente) com:
  - `create_run(competition: str) -> str` (run_id) e `get_run(run_id) -> dict`
  - `set_state(run_id: str, state: str) -> None`, `finish_run(run_id: str, stop_reason: str) -> None`
  - `add_experiment(run_id: str, candidate: Candidate, hypothesis: str, parent_id: str | None = None) -> bool` (False se o `config_hash` já existe na competição; índice único)
  - `finish_experiment(exp_id: str, *, status: str, cv_mean: float | None, cv_std: float | None, seeds: int, compute_s: float, cost_usd: float = 0.0, promoted: bool = False) -> None`
  - `get_experiment(exp_id) -> dict | None`, `list_experiments(run_id) -> list[dict]`
  - `add_submission(exp_id: str, kaggle_ref: str, message: str, cv_at_submit: float) -> str` (sub_id), `update_submission(sub_id: str, *, status: str, lb_public: float | None) -> None`, `list_submissions(run_id) -> list[dict]`, `count_submissions_today(competition: str, now: datetime) -> int` (dia UTC)
  - `add_leaderboard_snapshot(run_id: str, n_teams: int, our_score: float | None, percentile: float | None) -> None`
  - Tabelas e colunas exatamente as da spec seção 7.

- [ ] **Step 1: Write the failing tests**: `test_create_run_starts_in_init_state`, `test_set_state_persists_across_reopen` (fechar e reabrir o `Store` no mesmo arquivo), `test_add_experiment_rejects_duplicate_config_hash` (segundo `add_experiment` com `Candidate` de mesmo hash devolve `False` e só 1 linha existe), `test_finish_experiment_records_cv_and_status`, `test_count_submissions_today_counts_only_current_utc_day` (duas submissões, uma ontem, uma hoje → 1).
- [ ] **Step 2: Run** `pytest tests/test_store.py -v` → FAIL.
- [ ] **Step 3: Implement** `Store` com `sqlite3` da stdlib; `exp_id`/`run_id`/`sub_id` como uuid4 hex curto; `row_factory` devolvendo dict; chaves estrangeiras ligadas.
- [ ] **Step 4: Update the spec** seção 7: `runs` passa a incluir `state`.
- [ ] **Step 5: Run** `pytest tests/test_store.py -v` → PASS.
- [ ] **Step 6: Commit** `feat: add SQLite experiment store`.

---

### Task 3: Budget

**Files:**
- Create: `kagglers/core/budget.py`, `tests/test_budget.py`

**Interfaces:**
- Consumes: `PackRules` (Task 1).
- Produces:
  - `@dataclass(frozen=True) BudgetLimits(llm_usd: float, compute_minutes: float, submission_reserve: int = 1)`
  - `class Budget(limits: BudgetLimits, rules: PackRules)` com `charge_llm(usd: float)`, `charge_compute(seconds: float)`, `exhausted() -> str | None` (devolve o motivo, `"llm_budget"` ou `"compute_budget"`, ou `None`), `can_submit(submissions_today: int, final: bool = False) -> bool`
  - `can_submit` com `final=False` exige `submissions_today < daily_submission_limit - submission_reserve`; com `final=True`, `< daily_submission_limit`.

- [ ] **Step 1: Write the failing tests**: `test_exhausted_none_when_under_limits`, `test_exhausted_reports_llm_budget`, `test_exhausted_reports_compute_budget_after_charging_seconds` (limite de 1 min, `charge_compute(61)`), `test_can_submit_respects_reserve` (limite diário 5, reserva 1: `can_submit(3)` é True e `can_submit(4)` é False), `test_final_submission_may_use_reserve` (`can_submit(4, final=True)` True e `can_submit(5, final=True)` False).
- [ ] **Step 2: Run** `pytest tests/test_budget.py -v` → FAIL.
- [ ] **Step 3: Implement** `Budget` como acima.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: add budget limits and submission quota`.

---

### Task 4: Cliente Kaggle e percentil do leaderboard

**Files:**
- Create: `kagglers/core/kaggle_client.py`, `kagglers/core/leaderboard.py`, `tests/test_kaggle_client.py`, `tests/test_leaderboard.py`, `tests/fakes.py`

**Interfaces:**
- Consumes: nada de tarefas anteriores.
- Produces, em `kaggle_client.py`:
  - `class KaggleError(Exception)`; `class KaggleRulesNotAccepted(KaggleError)` (erros 403/"accept the rules"); `class KaggleTransientError(KaggleError)`
  - `@dataclass(frozen=True) SubmissionInfo(ref: str, description: str, status: str, public_score: float | None)`
  - `Protocol KaggleClient`: `download(competition: str, dest: Path) -> None`, `submit(competition: str, file: Path, message: str) -> None`, `submissions(competition: str) -> list[SubmissionInfo]`, `leaderboard(competition: str, dest: Path) -> Path`
  - `class CliKaggleClient(runner: Callable[[list[str]], subprocess.CompletedProcess] = default_runner, max_retries: int = 3, sleep: Callable[[float], None] = time.sleep)`: usa os subcomandos da spec (`competitions download -p <dest> --unzip`, `competitions submit -f -m --wait`, `competitions submissions <comp> -v`, `competitions leaderboard <comp> -d -p <dest>`). `submit` não devolve a ref; o chamador consulta `submissions()` e acha a ref pela `description`.
  - `class FakeKaggleClient` em `tests/fakes.py`: copia arquivos de um diretório local no `download`, grava `submit` em memória com ref incremental, `leaderboard` devolve um CSV pré-fabricado; aceita injeção de falhas (`fail_submit_after_success: bool` para simular a queda do Review Focus 5).
- Produces, em `leaderboard.py`:
  - `top_percent(our_score: float, leaderboard_csv: Path, higher_is_better: bool = True) -> float` = `100 * (n_better + 1) / n_teams`, com `n_better` = nº de equipes com score estritamente melhor; colunas do CSV `TeamId, TeamName, SubmissionDate, Score`; leaderboard vazio levanta `ValueError`.

- [ ] **Step 1: Write the failing tests**:
  - `test_leaderboard.py`: `test_top_percent_best_score_is_one_over_n` (10 equipes, nosso score maior que todos → 10.0), `test_top_percent_ties_do_not_count_as_better`, `test_top_percent_lower_is_better`, `test_top_percent_empty_leaderboard_raises`.
  - `test_kaggle_client.py` (com `runner` falso): `test_submit_builds_expected_command` (contém `competitions`, `submit`, `-f`, `-m`, `--wait`), `test_transient_failure_retries_with_backoff_then_raises` (3 chamadas, `sleep` chamado com valores crescentes, no fim `KaggleTransientError`), `test_rules_not_accepted_raises_without_retry` (saída com "403"/"accept" → `KaggleRulesNotAccepted` com 1 única chamada), `test_submissions_parses_verbose_csv` (stdout CSV com `ref,fileName,date,description,status,publicScore,privateScore` → lista de `SubmissionInfo`; `publicScore` vazio → `None`).
- [ ] **Step 2: Run** `pytest tests/test_leaderboard.py tests/test_kaggle_client.py -v` → FAIL.
- [ ] **Step 3: Implement** os dois módulos. O CSV de `submissions -v` varia por versão da CLI: o parser lê por nome de coluna e tolera colunas extras. Ao implementar, rode `kaggle competitions submissions -h` localmente (instalar `kaggle`) e confira os nomes reais das colunas; ajuste o teste com a saída real.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: add Kaggle client, fake and leaderboard percentile`.

---

### Task 5: Sandbox

**Files:**
- Create: `kagglers/core/sandbox.py`, `tests/test_sandbox.py`, `tests/sandbox_scripts/` (scripts de teste: `ok.py`, `sleep.py`, `alloc.py`, `echo_env.py`)

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) SandboxLimits(timeout_s: float, memory_mb: int, cpu_threads: int)`
  - `@dataclass(frozen=True) SandboxResult(status: str, returncode: int | None, duration_s: float, stdout: str, stderr: str)` com `status` em `{"ok","failed","timeout","oom"}`
  - `run_in_sandbox(args: list[str], workdir: Path, limits: SandboxLimits) -> SandboxResult`: executa `[sys.executable, "-I", *args]` com `cwd=workdir` e `env` mínimo (`PATH`, `SYSTEMROOT`, `TEMP`, `OMP_NUM_THREADS=cpu_threads`; **sem** `KAGGLE_*`, `ANTHROPIC_*` nem outras variáveis herdadas); thread de monitoramento faz polling a cada 0,2 s do RSS somado da árvore (psutil) e mata a árvore ao exceder `memory_mb` (`status="oom"`) ou `timeout_s` (`status="timeout"`).
  - `validate_operator_output(out_dir: Path) -> dict`: exige `oof.npy`, `test.npy` e `metrics.json` com as chaves `cv_mean`, `cv_std`, `fold_scores`, `n_runs`; levanta `SandboxOutputError` com a lista do que falta e devolve o JSON de métricas.

- [ ] **Step 1: Write the failing tests**: `test_ok_script_returns_ok_and_stdout`, `test_timeout_kills_process_and_reports_timeout` (`sleep.py` com `timeout_s=1`, termina em menos de 5 s), `test_memory_limit_kills_process_and_reports_oom` (`alloc.py` aloca ~500 MB, limite 150 MB), `test_child_processes_are_killed_on_timeout` (o script lança um filho; após o timeout, nenhum processo da árvore segue vivo), `test_secrets_not_inherited` (define `KAGGLE_KEY` no ambiente do teste via `monkeypatch`; `echo_env.py` imprime as variáveis e o valor não aparece), `test_validate_operator_output_lists_missing_files`, `test_validate_operator_output_returns_metrics`.
- [ ] **Step 2: Run** `pytest tests/test_sandbox.py -v` → FAIL.
- [ ] **Step 3: Implement** `sandbox.py` como acima. `subprocess.Popen` com `creationflags=CREATE_NEW_PROCESS_GROUP` no Windows; matar via `psutil.Process.children(recursive=True)` e depois o pai.
- [ ] **Step 4: Run** → PASS. Se o teste de memória for instável no CI local, aumentar a margem (alocar bem acima do limite) em vez de afrouxar a asserção.
- [ ] **Step 5: Commit** `feat: add subprocess sandbox with time, memory and env limits`.

---

### Task 6: Pack S6E10 — dados, folds e validação adversarial

**Files:**
- Create: `kagglers/packs/airline_s6e10/__init__.py`, `kagglers/packs/airline_s6e10/config.py`, `kagglers/packs/airline_s6e10/data.py`, `tests/conftest.py`, `tests/test_airline_data.py`

**Interfaces:**
- Consumes: `PackRules` (Task 1).
- Produces:
  - `config.py`: `@dataclass(frozen=True) AirlineConfig(competition: str = "playground-series-s6e10", data_dir: Path = Path("data/playground-series-s6e10"), positive_label: str | int | None = None, n_splits: int = 5, fold_seed: int = 42)` e `RULES = PackRules(metric_name="roc_auc", higher_is_better=True, daily_submission_limit=5, allow_external_data=False, allow_test_derived_features=False)`
  - `data.py`:
    - `@dataclass(frozen=True) Schema(id_col: str, target_col: str)` e `read_schema(data_dir: Path) -> Schema` (1ª e 2ª colunas de `sample_submission.csv`)
    - `load_train_test(data_dir: Path, schema: Schema) -> tuple[pd.DataFrame, pd.DataFrame]` (valida que `train.csv` tem `id_col` e `target_col`, e que `test.csv` tem `id_col` e **não** tem `target_col`; colunas de features iguais nos dois)
    - `encode_target(y: pd.Series, positive_label: str | int | None) -> np.ndarray` (0/1). Rótulos já em `{0,1}` dispensam `positive_label`; qualquer outro caso sem `positive_label` levanta `ValueError` listando os valores únicos (Review Focus 1).
    - `make_folds(y: np.ndarray, n_splits: int, seed: int) -> np.ndarray` (`fold_id` por linha, `StratifiedKFold(shuffle=True)`)
    - `save_folds(folds: np.ndarray, path: Path)` / `load_folds(path: Path) -> np.ndarray`
    - `adversarial_auc(train: pd.DataFrame, test: pd.DataFrame, sample: int = 50_000, seed: int = 0) -> float` (LightGBM pequeno, 3 folds, amostra balanceada)
- `tests/conftest.py` produz a fixture `synthetic_comp(tmp_path)` que escreve `train.csv` (600 linhas, `id`, 4 numéricas, 2 categóricas em texto, alvo `satisfaction` com `"satisfied"`/`"neutral or dissatisfied"` dependendo das features), `test.csv` (200 linhas, 1 categoria não vista no treino, alguns nulos) e `sample_submission.csv` (`id,satisfaction`), e devolve o diretório.

- [ ] **Step 1: Write the failing tests** em `tests/test_airline_data.py`: `test_read_schema_uses_first_two_sample_submission_columns`, `test_load_train_test_rejects_target_in_test`, `test_encode_target_numeric_labels_need_no_positive_label`, `test_encode_target_text_labels_without_positive_label_raises_listing_values` (a mensagem contém ambos os rótulos), `test_encode_target_maps_positive_label_to_one`, `test_make_folds_is_deterministic_for_seed`, `test_make_folds_is_stratified` (taxa do alvo em cada fold difere da global em menos de 0,02), `test_save_load_folds_roundtrip`, `test_adversarial_auc_near_half_for_same_distribution` (< 0,6), `test_adversarial_auc_high_for_shifted_distribution` (> 0,8).
- [ ] **Step 2: Run** `pytest tests/test_airline_data.py -v` → FAIL.
- [ ] **Step 3: Implement** `config.py` e `data.py`. Categóricas em texto viram `category` com o conjunto de categorias da **união** treino+teste, para não quebrar em categorias não vistas (Review Focus 2); essa conversão fica numa função `to_model_frame(train, test) -> tuple[pd.DataFrame, pd.DataFrame]` que o Task 7 também usa, e ganha o teste `test_to_model_frame_handles_unseen_categories_and_nulls`.
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: add S6E10 pack data loading, folds and adversarial check`.

---

### Task 7: Pack S6E10 — operador baseline, Evaluator e Packager

**Files:**
- Create: `kagglers/packs/airline_s6e10/baseline_operator.py`, `kagglers/packs/airline_s6e10/evaluator.py`, `kagglers/packs/airline_s6e10/packager.py`, `tests/test_airline_evaluator.py`, `tests/test_airline_packager.py`
- Modify: `kagglers/packs/airline_s6e10/__init__.py` (exporta `build_pack(config: AirlineConfig, workdir: Path) -> Pack` e registra o nome `airline_s6e10`)

**Interfaces:**
- Consumes: `run_in_sandbox`, `validate_operator_output`, `SandboxLimits` (Task 5), funções de `data.py` (Task 6), `Score`/`Artifact`/`Candidate` (Task 1).
- Produces:
  - `baseline_operator.py` com `main(config_path: Path) -> None`, executável como `python -I -m kagglers.packs.airline_s6e10.baseline_operator <config.json>`. O JSON traz `data_dir`, `folds_path`, `out_dir`, `positive_label`, `seeds: list[int]`, `params: dict` (LightGBM) e `n_jobs`. Para cada seed e cada fold treina `LGBMClassifier`, acumula `oof.npy` (média entre seeds) e `test.npy` (média entre folds e seeds) e grava `metrics.json` com `cv_mean` (AUC do OOF por fold, média), `cv_std` (desvio dos AUCs por fold), `fold_scores`, `n_runs = len(seeds) * n_splits`.
  - `evaluator.py`: `class AirlineEvaluator(config: AirlineConfig, runs_dir: Path, limits: SandboxLimits)` com `evaluate(candidate: Candidate) -> Score`. Escreve o JSON de config do operador em `runs_dir/<exp_id>/`, chama `run_in_sandbox`, e se `status != "ok"` levanta `EvaluationFailed(status, stderr)`; valida a saída com `validate_operator_output` e confere formas (`oof.shape == (n_train,)`, `test.shape == (n_test,)`).
  - `packager.py`: `validate_submission(df: pd.DataFrame, sample: pd.DataFrame) -> None` (levanta `SubmissionFormatError` com a razão) e `class AirlinePackager(config)` com `build(score: Score, out_dir: Path) -> Artifact`, que escreve `submission.csv` (`id` + probabilidade da classe positiva, na ordem do `sample_submission`) e chama `validate_submission` antes de devolver.

- [ ] **Step 1: Write the failing tests**:
  - `test_airline_evaluator.py` (usa `synthetic_comp`, `params` pequenos e `seeds=[0]`): `test_baseline_evaluate_returns_score_with_oof_and_test_shapes`, `test_baseline_cv_auc_beats_chance_on_synthetic_signal` (`score.value > 0.7`), `test_baseline_evaluates_with_unseen_category_and_nulls_in_test`, `test_evaluator_raises_evaluation_failed_on_sandbox_timeout` (limite de 0,1 s), `test_evaluator_is_reproducible_for_same_seed` (duas execuções dão o mesmo `value`).
  - `test_airline_packager.py`: `test_build_writes_csv_matching_sample_submission_columns_and_order`, `test_build_has_no_nan_and_probabilities_in_unit_interval`, `test_validate_submission_rejects_missing_ids`, `test_validate_submission_rejects_duplicate_ids`, `test_validate_submission_rejects_reordered_ids`, `test_validate_submission_rejects_nan`, `test_validate_submission_rejects_wrong_row_count`.
- [ ] **Step 2: Run** `pytest tests/test_airline_evaluator.py tests/test_airline_packager.py -v` → FAIL.
- [ ] **Step 3: Implement** os três módulos. O operador usa `to_model_frame` do Task 6 e `n_jobs` do JSON, e termina com código ≠ 0 em qualquer exceção (o sandbox traduz isso em `failed`).
- [ ] **Step 4: Run** → PASS.
- [ ] **Step 5: Commit** `feat: add S6E10 baseline operator, evaluator and packager`.

---

### Task 8: Orquestrador, CLI e teste de ponta a ponta

**Files:**
- Create: `kagglers/core/orchestrator.py`, `kagglers/cli.py`, `tests/test_orchestrator.py`, `README.md`
- Modify: `kagglers/packs/__init__.py` (`get_pack(name: str, **kwargs) -> Pack` com registro por nome)

**Interfaces:**
- Consumes: `Store` (Task 2), `Budget` (Task 3), `KaggleClient`/`SubmissionInfo` e `top_percent` (Task 4), `Pack` e demais tipos (Task 1), `build_pack` e `AirlineConfig` (Task 7).
- Produces:
  - `class Orchestrator(pack: Pack, store: Store, budget: Budget, kaggle: KaggleClient, work_dir: Path, now: Callable[[], datetime] = utcnow)` com `run(run_id: str | None = None) -> str` (devolve o `run_id`; com `run_id` dado, retoma). Estados da fase 1, em ordem: `INIT → PREPARE → BASELINE → SUBMIT → MONITOR → STOP`. Cada estado grava `store.set_state` ao **concluir** e é idempotente: um estado já concluído é pulado na retomada.
  - PREPARE: chama `kaggle.download`, depois `pack.prepare()` (folds + adversarial; alerta registrado no log se o AUC adversarial passar de 0,6).
  - BASELINE: `add_experiment` + `pack.evaluator.evaluate` + `finish_experiment`; falha do avaliador grava `status` `failed`/`timeout` e vai para STOP com o motivo. Soma o tempo ao `budget.charge_compute`.
  - SUBMIT: só envia se `budget.can_submit(store.count_submissions_today(...), final=True)`. A mensagem é `f"kagglers {exp_id}"`. Antes de enviar, consulta `kaggle.submissions` e, se existir uma com essa descrição, só reconcilia o registro no banco (Review Focus 5). Depois do envio, procura a `SubmissionInfo` pela descrição e grava `kaggle_ref`, `status` e `lb_public`.
  - MONITOR: baixa o leaderboard, calcula `top_percent` com o `lb_public` da submissão e grava o snapshot.
  - STOP: `finish_run` com o motivo (`"phase1_complete"`, `"quota_exhausted"`, `"evaluation_failed"`, `"compute_budget"`, ...).
  - `cli.py`: `main(argv: list[str] | None = None) -> int` com `kagglers run <pack> --phase1 [--resume RUN_ID] [--compute-minutes N]`; imprime o CV (média ± desvio), o caminho do CSV, a `kaggle_ref` e o percentil.
  - `README.md`: instalação, configuração do token (`~/.kaggle/kaggle.json`), aceitar as regras na página da competição, comando de execução e a verificação manual.

- [ ] **Step 1: Write the failing tests** em `tests/test_orchestrator.py` (com `synthetic_comp`, `FakeKaggleClient` e `positive_label="satisfied"`): `test_full_phase1_run_reaches_stop_with_submission_and_snapshot` (1 experimento `ok`, 1 submissão com `kaggle_ref` e 1 snapshot; `stop_reason == "phase1_complete"`), `test_resume_after_crash_in_monitor_does_not_retrain` (o fake falha no `leaderboard` na primeira passada; ao retomar com o mesmo `run_id` a tabela `experiments` continua com 1 linha e a run termina), `test_resume_after_submit_success_does_not_resubmit` (`fail_submit_after_success=True`: a CLI aceita o envio, mas a escrita falha; ao retomar o fake registra exatamente 1 submissão), `test_quota_exhausted_stops_without_submitting` (cota diária de 1 já consumida, `stop_reason == "quota_exhausted"`, nenhuma chamada a `submit`), `test_failed_baseline_stops_with_evaluation_failed` (limite de timeout absurdamente baixo), `test_rules_not_accepted_stops_with_clear_reason` (o fake levanta `KaggleRulesNotAccepted` no `download`; a run termina com esse motivo sem retry) e, em `tests/test_cli.py`, `test_cli_parses_run_command_and_returns_zero_on_success` (injetando o cliente fake).
- [ ] **Step 2: Run** `pytest tests/test_orchestrator.py -v` → FAIL.
- [ ] **Step 3: Implement** `orchestrator.py`, `get_pack` e `cli.py`. Em caso de erro não tratado num estado, a run grava o motivo e o estado atual (não conclui o estado) para permitir a retomada.
- [ ] **Step 4: Run the whole suite** `pytest -v` → todos PASS.
- [ ] **Step 5: Manual verification (precisa das suas credenciais; não automatizar)**: com o token em `~/.kaggle/kaggle.json` e as regras aceitas, rodar `kagglers run airline_s6e10 --phase1`. Esperado: dados baixados em `data/playground-series-s6e10/`, CV AUC impresso na faixa 0,95+, `submission.csv` validado, `kaggle_ref` impressa e percentil gravado. Registrar no README o AUC de CV e o LB do baseline.
- [ ] **Step 6: Commit** `feat: add phase 1 orchestrator, CLI and end-to-end tests`.

---

## Self-review

- **Cobertura da spec:** arquitetura/interfaces (T1), store e esquema (T2), orçamento/cota e reserva (T3), cliente Kaggle, erros e percentil (T4), sandbox (T5), folds, leakage por label (`positive_label`) e adversarial (T6), baseline/Evaluator/Packager e validação de formato (T7), ciclo, retomada, parada e MONITOR (T8). Itens da spec deliberadamente fora da fase 1 e listados em "Desvios": PROPOSE, limiar de ruído e seu calibrador, ensemble com nested CV, análise estática anti-leakage.
- **Consistência de tipos:** `Candidate/Score/Artifact/PackRules/Pack` definidos na T1 e usados com os mesmos nomes nas T2 a T8; `validate_operator_output`/`SandboxLimits` (T5) usados na T7; `top_percent` (T4) usado na T8; `to_model_frame` (T6) usado na T7.
- **Pendências externas ao código:** confirmação dos fatos da competição, token do Kaggle, regras aceitas e `positive_label` real (descobrir olhando os valores únicos do alvo no primeiro PREPARE).
