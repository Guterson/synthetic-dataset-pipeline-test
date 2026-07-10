import xml.etree.ElementTree as ET
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_musicxml_complet_pipeline(xml_path, tpqn=480):
    """
    Varre o arquivo MusicXML uma única vez (paralelismo de dados).
    Extrai tanto a sequência de eventos canônicos quanto as diretrizes de andamento.
    """
    tree = ET.parse(xml_path)
    root = tree.getroot()

    canonical_events = []
    score_directives = []

    note_base_midi = {"C": 12, "D": 14, "E": 16, "F": 17, "G": 19, "A": 21, "B": 23}

    for part in root.findall("part"):
        current_time_ticks = 0
        divisions = 1
        last_note_duration_ticks = 0

        # Mapeamento dinâmico para fechar o end_tick de ritardandos/accelerandos
        active_ramps = []

        for measure in part.findall("measure"):
            # 1. Captura atributos estruturais (divisions, etc.)
            attributes = measure.find("attributes")
            if attributes is not None:
                div_elem = attributes.find("divisions")
                if div_elem is not None:
                    divisions = int(div_elem.text)

            # 2. Captura metadados de andamento na partitura (<direction>)
            for direction in measure.findall("direction"):
                # Verifica andamento nominal explícito (ex: <sound tempo="76"/>)
                sound = direction.find("sound")
                if sound is not None and "tempo" in sound.attrib:
                    bpm_value = float(sound.attrib["tempo"])
                    score_directives.append(
                        {
                            "tick": current_time_ticks,
                            "type": "nominal",
                            "value": bpm_value,
                        }
                    )

                # Verifica marcações textuais de rampa (ritardando/accelerando) em <words>
                # Nota: Softwares reais costumam ancorar isso em tags tipo <direction-type><words>
                words = direction.find(".//words")
                if words is not None and words.text:
                    texto = words.text.lower()
                    if "rit" in texto or "ritardando" in texto:
                        active_ramps.append(
                            {"start_tick": current_time_ticks, "type": "ritardando"}
                        )
                    elif "accel" in texto or "accelerando" in texto:
                        active_ramps.append(
                            {"start_tick": current_time_ticks, "type": "accelerando"}
                        )
                    elif "a tempo" in texto or "tempo i" in texto:
                        score_directives.append(
                            {"tick": current_time_ticks, "type": "a_tempo"}
                        )

            # 3. Processamento sequencial de elementos de nota do compasso
            measure_duration_ticks = 0
            for element in measure:
                if element.tag == "note":
                    is_chord = element.find("chord") is not None
                    is_rest = element.find("rest") is not None

                    duration_elem = element.find("duration")
                    if duration_elem is not None:
                        xml_duration = float(duration_elem.text)
                        duration_ticks = int(round((xml_duration / divisions) * tpqn))
                    else:
                        duration_ticks = 0

                    if not is_chord:
                        if not is_rest:
                            start_time = current_time_ticks
                        measure_duration_ticks += duration_ticks
                    else:
                        start_time = current_time_ticks - last_note_duration_ticks

                    if duration_ticks > 0 and not is_rest:
                        pitch_elem = element.find("pitch")
                        if pitch_elem is not None:
                            step = pitch_elem.find("step").text
                            octave = int(pitch_elem.find("octave").text)
                            midi_note = note_base_midi[step] + (octave * 12)

                            alter_elem = pitch_elem.find("alter")
                            if alter_elem is not None:
                                midi_note += int(alter_elem.text)

                            canonical_events.append(
                                {
                                    "pitch": midi_note,
                                    "start_time": start_time,
                                    "duration": duration_ticks,
                                    "volume": 90,
                                }
                            )

                    if not is_chord:
                        current_time_ticks += duration_ticks
                        last_note_duration_ticks = duration_ticks

            # Lógica elástica: se uma rampa foi aberta nesta medida, definimos que ela
            # estende-se e termina exatamente no final do compasso corrente (limite local)
            while active_ramps:
                ramp = active_ramps.pop(0)
                score_directives.append(
                    {
                        "tick": ramp["start_tick"],
                        "type": ramp["type"],
                        "end_tick": current_time_ticks,  # O compasso fecha a rampa se não houver alvo numérico
                    }
                )

    return canonical_events, score_directives, current_time_ticks


# [As funções matemáticas puras desenvolvidas anteriormente entram aqui]
def calculate_pure_plateau_ramp(start_bpm, duration_ticks, direction="ritardando"):
    t = np.arange(duration_ticks)
    if duration_ticks <= 0:
        return np.array([start_bpm])
    if direction == "ritardando":
        target_bpm = max(40.0, start_bpm * 0.70)
        logistic_filter = 1.0 / (1.0 + np.exp(-0.005 * (t - (duration_ticks * 0.4))))
        return start_bpm - (start_bpm - target_bpm) * logistic_filter
    else:
        target_bpm = min(240.0, start_bpm * 1.30)
        logistic_filter = 1.0 / (1.0 + np.exp(-0.003 * (t - (duration_ticks * 0.5))))
        return start_bpm + (target_bpm - start_bpm) * logistic_filter


def generate_score_only_tempo_curve(score_directives, total_ticks):
    base_curve = np.zeros(total_ticks)
    current_nominal_bpm = 120.0
    tempo_stack = [current_nominal_bpm]
    directives = sorted(score_directives, key=lambda x: x["tick"])

    i = 0
    while i < total_ticks:
        current_dir = next((d for d in directives if d["tick"] == i), None)
        if current_dir:
            if current_dir["type"] == "nominal":
                current_nominal_bpm = current_dir["value"]
                tempo_stack.append(current_nominal_bpm)
                base_curve[i] = current_nominal_bpm
                i += 1
            elif current_dir["type"] == "a_tempo":
                if len(tempo_stack) > 1:
                    current_nominal_bpm = tempo_stack[-1]
                base_curve[i] = current_nominal_bpm
                i += 1
            elif current_dir["type"] in ["ritardando", "accelerando"]:
                start_tick = i
                end_tick = current_dir["end_tick"]
                ramp_bpms = calculate_pure_plateau_ramp(
                    current_nominal_bpm,
                    end_tick - start_tick,
                    direction=current_dir["type"],
                )
                for t_idx, bpm_val in enumerate(ramp_bpms):
                    if start_tick + t_idx >= total_ticks:
                        break
                    base_curve[start_tick + t_idx] = bpm_val
                current_nominal_bpm = ramp_bpms[-1]
                i = end_tick
        else:
            base_curve[i] = current_nominal_bpm
            i += 1
    return base_curve


def generate_continuous_human_walk(base_curve, config):
    total_ticks = len(base_curve)
    expressive_curve = np.zeros(total_ticks)
    random_walk_state = 0.0
    inertia = 0.9995
    for tick in range(total_ticks):
        score_bpm = base_curve[tick]
        innovation = np.random.normal(0, score_bpm * config["random_walk_volatility"])
        random_walk_state = (inertia * random_walk_state) + (
            (1.0 - inertia) * innovation
        )
        expressive_curve[tick] = np.clip(score_bpm + random_walk_state, 40.0, 240.0)
    return expressive_curve


# =====================================================================
# MAIN PIPELINE & PLOT GRAPH
# =====================================================================
if __name__ == "__main__":
    np.random.seed(42)
    xml_filename = (
        "hungarian_dance_n5.musicxml"  # Arquivo MusicXML gerado anteriormente
    )

    # 1. Extração Dinâmica e Paralela de Notas e Diretrizes diretamente do arquivo
    try:
        eventos_canonicos, diretivas_extraidas, total_ticks_musica = (
            parse_musicxml_complet_pipeline(xml_filename)
        )

        print(
            f"Sucesso! {len(eventos_canonicos)} notas extraídas e {len(diretivas_extraidas)} diretrizes de andamento catalogadas."
        )
        print("Diretrizes detectadas no XML:", diretivas_extraidas)

        # 2. Computação das Curvas Base e Humana
        pipeline_config = {"random_walk_volatility": 0.04}
        curva_base_partitura = generate_score_only_tempo_curve(
            diretivas_extraidas, total_ticks_musica
        )
        curva_humana_expressiva = generate_continuous_human_walk(
            curva_base_partitura, pipeline_config
        )

        # 3. LOOKUP: Associa cada nota canônica ao seu andamento localizado na grade
        for ev in eventos_canonicos:
            ev["local_bpm"] = curva_humana_expressiva[
                min(ev["start_time"], total_ticks_musica - 1)
            ]

        # =================================================================
        # REQUISITO: GERAÇÃO DO GRÁFICO (MATPLOTLIB)
        # =================================================================
        plt.figure(figsize=(12, 6))
        ticks_eixo_x = np.arange(total_ticks_musica)

        plt.savefig("dump")

        # Desenha a linha determinística da partitura (Platô)
        plt.plot(
            ticks_eixo_x,
            curva_base_partitura,
            label="Curva Alvo da Partitura (Semântica com Platô)",
            color="blue",
            linestyle="--",
            linewidth=2,
        )  # Desenha a curva humana contínua com o Random Walk integrado
        plt.plot(
            ticks_eixo_x,
            curva_humana_expressiva,
            label="Curva Expressiva Humana (Random Walk + Platô)",
            color="darkred",
            alpha=0.8,
            linewidth=1.5,
        )  # Plota marcadores onde as notas musicais de fato acontecem (Amostragem/Lookup)
        onsets_x = [e["start_time"] for e in eventos_canonicos]
        onsets_y = [e["local_bpm"] for e in eventos_canonicos]
        plt.scatter(
            onsets_x,
            onsets_y,
            color="black",
            zorder=5,
            label="Ataques das Notas (Onsets)",
        )  # Destaca as diretrizes semânticas da música no gráfico
        for d in diretivas_extraidas:
            plt.axvline(x=d["tick"], color="gray", alpha=0.5, linestyle=":")
            plt.text(
                d["tick"] + 10,
                plt.ylim()[0] + 5,
                f"[{d['type'].upper()}]",
                rotation=90,
                color="gray",
                fontsize=9,
            )
        plt.title(
            "Evolução Contínua do Andamento Expressivo (Tempo Curve Model)",
            fontsize=14,
            fontweight="bold",
        )
        plt.xlabel("Tempo de Linha (Ticks MIDI)", fontsize=12)
        plt.ylabel("Andamento (BPM)", fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc="upper right")
        plt.tight_layout()  # Exibe o gráfico na tela
        caminho_imagem = "curva_tempo_expressiva.png"

        plt.savefig(
            caminho_imagem, dpi=300
        )  # Salva o gráfico em alta resolução na pasta
        print(f"\n[SUCESSO] Gráfico gerado e salvo em: {caminho_imagem}")
        print(
            "-> DICA: Basta clicar no arquivo 'curva_tempo_expressiva.png' na barra lateral esquerda do VS Code para ver o gráfico!"
        )
        plt.close()
    except FileNotFoundError:
        print(f"Erro: Salve o arquivo '{xml_filename}' no diretório antes de rodar.")
