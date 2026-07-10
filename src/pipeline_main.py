import os
import logging
import random
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List

import psutil  # Biblioteca útil para ler informações do sistema (pip install psutil)
from lxml import etree


def init_worker_process():
    """
    Função executada IMEDIATAMENTE quando um novo processo trabalhador é criado.
    Ela serve para reduzir a prioridade do processo no sistema operacional.
    """
    try:
        # No Linux/macOS (Muda o "nice" do processo. Valores maiores = menor prioridade)
        if hasattr(os, "nice"):
            os.nice(
                10
            )  # Deixa o processo "gentil", cedendo CPU para outras tarefas do usuário
        else:
            # No Windows
            p = psutil.Process(os.getpid())
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    except Exception:
        pass  # Se falhar, o script continua rodando normalmente


def process_single_variation(
    xml_path: Path,
    variation_id: int,
    output_dir: Path,
    sfz_pool: List[Path],
    rir_pool: List[Path],
) -> bool:
    file_prefix = f"{xml_path.stem}_var_{variation_id:04d}"

    midi_path = output_dir / f"{file_prefix}.mid"
    dry_wav_path = output_dir / f"{file_prefix}_dry.wav"
    wet_wav_path = output_dir / f"{file_prefix}_wet.wav"

    try:
        # [Passos 1 a 3: Parser, Modificadores e Geração do MIDI - Omitidos para focar no Áudio]
        # ... código de geração do midi_path ...

        # 4. Renderização do Áudio Seco (SFZ)
        chosen_sfz = random.choice(sfz_pool)
        # TODO: Sua função real vai gerar o arquivo dry_wav_path aqui
        dry_wav_path.write_text(f"MOCK AUDIO SECO USANDO {chosen_sfz.name}")

        # 5. Convolução de Espaço (RIR)
        chosen_rir = random.choice(rir_pool)
        # TODO: Sua função real vai ler o dry_wav_path e gerar o wet_wav_path aqui
        wet_wav_path.write_text(f"MOCK AUDIO CONVOLVIDO USANDO {chosen_rir.name}")

        # 6. OTIMIZAÇÃO DE ESPAÇO: Deleta o arquivo seco imediatamente após o uso
        if dry_wav_path.exists():
            dry_wav_path.unlink()  # Remove o arquivo físico do disco na hora

        logging.info(f"[{file_prefix}] Variação gerada e otimizada com sucesso.")
        return True

    except Exception as e:
        logging.error(f"[{file_prefix}] Falha na variação: {e}")
        # Garante limpeza do arquivo temporário mesmo se o passo da RIR falhar no meio
        if dry_wav_path.exists():
            dry_wav_path.unlink()
        return False


def run_bulk_dataset_generation(
    input_xml_file: Path,
    num_variations: int,
    output_directory: Path,
    sfz_pool_directory: Path,
    rir_pool_directory: Path,
    cpu_utilization_ratio: float = 0.75,  # 0.75 significa usar no máximo 75% da sua CPU
):
    if not input_xml_file.is_file():
        logging.critical(f"Arquivo inválido: {input_xml_file}")
        return

    output_directory.mkdir(parents=True, exist_ok=True)
    sfz_files = list(sfz_pool_directory.glob("**/*.sfz"))
    rir_files = list(rir_pool_directory.glob("**/*.wav"))

    # --- CÁLCULO SEGURO DE CORES DA CPU ---
    total_cores = os.cpu_count() or 1
    # Garante que vai usar pelo menos 1 core, mas limita pela proporção (ex: 8 * 0.75 = 6 cores)
    calculated_workers = max(1, int(total_cores * cpu_utilization_ratio))

    # Se o cálculo der que vai usar todos os cores, força deixar pelo menos 1 livre pro sistema
    if calculated_workers >= total_cores and total_cores > 1:
        calculated_workers = total_cores - 1

    logging.info(f"Máquina possui {total_cores} núcleos lógicos.")
    logging.info(
        f"Configurando pipeline para usar {calculated_workers} núcleos (Segurança de Sistema Ativa)."
    )

    success_count = 0
    failure_count = 0

    # Passamos a função init_worker_process para o initializer.
    # Cada novo processo criado vai rodar ela antes de começar o trabalho pesado.
    with ProcessPoolExecutor(
        max_workers=calculated_workers, initializer=init_worker_process
    ) as executor:
        futures = {
            executor.submit(
                process_single_variation,
                input_xml_file,
                v_id,
                output_directory,
                sfz_files,
                rir_files,
            ): v_id
            for v_id in range(num_variations)
        }

        for future in as_completed(futures):
            v_id = futures[future]
            try:
                if future.result():
                    success_count += 1
                else:
                    failure_count += 1
            except Exception as exc:
                logging.critical(f"Variação {v_id} causou um crash fatal: {exc}")
                failure_count += 1

    logging.info(f"Lote concluído. Sucessos: {success_count} | Falhas: {failure_count}")
