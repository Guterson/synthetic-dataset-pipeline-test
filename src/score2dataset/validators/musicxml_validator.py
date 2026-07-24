from pathlib import Path
import logging

from lxml import etree


def is_strict_musicxml(file_path: Path, xsd_schema_path: Path) -> bool:
    """
    Realiza uma validação estrutural profunda contra o esquema oficial XSD do MusicXML.
    Retorna True apenas se o arquivo for semanticamente um MusicXML válido.
    """
    if not file_path.is_file() or not xsd_schema_path.is_file():
        logging.error("[Validador] Arquivos de entrada ou de esquema XSD ausentes.")
        return False

    try:
        # 1. Parse do arquivo de regras oficial (.xsd)
        xsd_doc = etree.parse(str(xsd_schema_path))
        musicxml_schema = etree.XMLSchema(
            xsd_doc
        )  # Transforma lxml em um motor validador específico

        # 2. Parse do seu arquivo de partitura (.musicxml)
        target_xml_doc = etree.parse(str(file_path))

        # 3. Executa a validação profunda baseada em regras
        # Se houver qualquer tag inválida ou fora de ordem, retorna False
        is_valid = musicxml_schema.validate(target_xml_doc)

        if not is_valid:
            # Captura e exibe no log exatamente qual tag ou linha musical violou a regra
            error_log = musicxml_schema.error_log.filter_from_errors()
            logging.warning(
                f"[Validador] O arquivo {file_path.name} quebrou as regras do MusicXML: {error_log}"
            )

        return is_valid

    except etree.XMLSyntaxError as syntax_err:
        logging.error(
            f"[Validador] Erro crasso de sintaxe XML no arquivo: {syntax_err}"
        )
        return False
    except Exception as general_err:
        logging.error(f"[Validador] Falha inesperada durante a checagem: {general_err}")
        return False
