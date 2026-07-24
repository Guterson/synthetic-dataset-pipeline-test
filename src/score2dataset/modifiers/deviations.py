import time

import numpy as np

from src.repository.musicxml_to_events import parse_musicxml_to_midi_ticks


def apply_event_level_deviations(canonical_events, config):
    """
    Aplica desvios estocásticos baseados em falhas motoras humanas
    a uma lista de eventos canônicos.
    """
    # -----------------------------------------------------------------
    # 1. MODIFICAÇÃO DISCRETA DE IDENTIDADES (Keep, Omit, Sub)
    # -----------------------------------------------------------------
    modified_sequence = []

    # Contadores para validação do Limiar de Integridade
    N_sub = 0
    N_omit = 0

    # Distribuição Categorizada: define as probabilidades de cada ação
    p_keep = config["p_keep"]
    p_sub = config["p_sub"]
    p_omit = config["p_omit"]

    # Validação das probabilidades
    assert np.isclose(p_keep + p_sub + p_omit, 1.0)

    outcomes = ["keep", "sub", "omit"]
    probabilities = [p_keep, p_sub, p_omit]

    for e in canonical_events:
        # Sorteia o destino da nota baseado na Distribuição Categorizada
        y_i = np.random.choice(outcomes, p=probabilities)

        if y_i == "keep":
            # Preserva a nota original intacta
            modified_sequence.append(e.copy())

        elif y_i == "sub":
            # Substituição: Mantém tempo, duração e volume, mas desloca o Pitch
            N_sub += 1
            e_sub = e.copy()

            # Amostragem da Gaussiana Discreta para o desvio do pitch (\delta_p)
            delta_p = int(round(np.random.normal(0, config["sigma_p"])))
            # Garante que o pitch fique dentro dos limites MIDI (0-127)
            e_sub["pitch"] = max(0, min(127, e_sub["pitch"] + delta_p))

            modified_sequence.append(e_sub)

        elif y_i == "omit":
            # Omissão: A nota simplesmente desaparece (\emptyset)
            N_omit += 1
            continue

    # -----------------------------------------------------------------
    # 2. INSERÇÕES ADITIVAS (Ghost Strikes / Notas Fantasmas)
    # -----------------------------------------------------------------
    # Sorteia o número total de inserções baseado na Distribuição de Poisson (\lambda)
    N_ins = np.random.poisson(config["lambda_poisson"])

    # Se houver notas na sequência modificada e inserções a fazer
    if len(modified_sequence) > 0 and N_ins > 0:
        # Sorteia de forma Uniforme quais índices da sequência vão gerar o erro aditivo
        target_indices = np.random.randint(0, len(modified_sequence), size=N_ins)

        for idx in target_indices:
            parent_note = modified_sequence[idx]

            # Cria a nota fantasma herdando tempo, duração e volume da nota mãe
            e_ins = parent_note.copy()

            # Desloca o pitch usando a mesma Gaussiana Discreta (\delta_ins)
            delta_ins = int(round(np.random.normal(0, config["sigma_p"])))
            e_ins["pitch"] = max(0, min(127, e_ins["pitch"] + delta_ins))

            # Insere o ghost strike de volta na sequência
            modified_sequence.append(e_ins)

    # Reordena a sequência final por start_time para manter a ordem cronológica
    modified_sequence.sort(key=lambda x: x["start_time"])

    # -----------------------------------------------------------------
    # 3. VALIDAÇÃO: LIMIAR DE INTEGRIDADE ESTRUTURAL (\Gamma)
    # -----------------------------------------------------------------
    N_neutral = len(canonical_events)
    w_s = config["w_s"]
    w_o = config["w_o"]
    w_i = config["w_i"]
    gamma = config["gamma_threshold"]

    cumulative_noise = (w_s * N_sub + w_o * N_omit + w_i * N_ins) / N_neutral

    is_valid_performance = cumulative_noise < gamma

    # Metadados do processamento para auditoria científica
    metrics = {
        "N_sub": N_sub,
        "N_omit": N_omit,
        "N_ins": N_ins,
        "cumulative_noise": round(cumulative_noise, 4),
        "is_valid": is_valid_performance,
    }

    return modified_sequence, metrics


# =====================================================================
# CONFIGURAÇÃO DO BENCHMARK (801 TENTATIVAS)
# =====================================================================
if __name__ == "__main__":
    # Garantindo reprodutibilidade científica
    np.random.seed(42)

    # Nosso template canônico de teste baseado no "The Entertainer"
    hungarian_canonical = parse_musicxml_to_midi_ticks("hungarian_dance_n5.musicxml")

    pipeline_config = {
        "p_keep": 0.85,
        "p_sub": 0.10,
        "p_omit": 0.05,
        "sigma_p": 1.2,
        "lambda_poisson": 2.0,
        "w_s": 0.2,
        "w_o": 0.5,
        "w_i": 1.5,
        "gamma_threshold": 0.40,
    }

    TOTAL_TRIES = 801

    # Listas para acumular os resultados de cada iteração
    substitutions_history = []
    omissions_history = []
    insertions_history = []
    noise_history = []
    valid_count = 0

    print(f"Iniciando simulação de {TOTAL_TRIES} execuções ruidosas...")

    # Cronômetro de alta precisão ativado para o loop inteiro
    start_benchmark = time.perf_counter()

    for _ in range(TOTAL_TRIES):
        # Executa o pipeline
        _, run_report = apply_event_level_deviations(
            hungarian_canonical, pipeline_config
        )

        # Armazena os dados brutos da rodada
        substitutions_history.append(run_report["N_sub"])
        omissions_history.append(run_report["N_omit"])
        insertions_history.append(run_report["N_ins"])
        noise_history.append(run_report["cumulative_noise"])

        if run_report["is_valid"]:
            valid_count += 1

    end_benchmark = time.perf_counter()

    # -----------------------------------------------------------------
    # CÁLCULO DAS MÉTRICAS AGREGADAS
    # -----------------------------------------------------------------
    total_time = end_benchmark - start_benchmark
    avg_time_ms = (total_time / TOTAL_TRIES) * 1000  # Convertido para milissegundos

    avg_subs = np.mean(substitutions_history)
    avg_omits = np.mean(omissions_history)
    avg_ins = np.mean(insertions_history)
    avg_noise = np.mean(noise_history)

    success_rate = (valid_count / TOTAL_TRIES) * 100

    # -----------------------------------------------------------------
    # EXIBIÇÃO DO RELATÓRIO FINAL
    # -----------------------------------------------------------------
    print("\n" + "=" * 50)
    print(f"📊 RELATÓRIO DO PIPELINE ESTOCÁSTICO ({TOTAL_TRIES} TENTATIVAS)")
    print("=" * 50)
    print(f"⏱️ Tempo Total de Processamento: {total_time:.4f} segundos")
    print(f"⚡ Tempo Médio por Execução:     {avg_time_ms:.3f} milissegundos")
    print("-" * 50)
    print(f"🎵 Média de Substituições:       {avg_subs:.2f} notas por música")
    print(f"🔇 Média de Omissões:            {avg_omits:.2f} notas por música")
    print(f"👻 Média de Inserções (Ghosts):  {avg_ins:.2f} notas por música")
    print(f"📉 Índice de Ruído Médio:        {avg_noise:.3f}")
    print("-" * 50)
    print(f"✅ Performances Válidas (Γ):    {valid_count} de {TOTAL_TRIES}")
    print(f"📈 Taxa de Aproveitamento:       {success_rate:.2f}%")
    print("=" * 50)
