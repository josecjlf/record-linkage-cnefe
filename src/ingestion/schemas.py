"""Contratos de colunas observados nos arquivos de entrada."""

from __future__ import annotations

from pyspark.sql.types import StringType, StructField, StructType


CNEFE_COLUMNS = (
    "COD_UNICO_ENDERECO",
    "COD_UF",
    "COD_MUNICIPIO",
    "COD_DISTRITO",
    "COD_SUBDISTRITO",
    "COD_SETOR",
    "NUM_QUADRA",
    "NUM_FACE",
    "CEP",
    "DSC_LOCALIDADE",
    "NOM_TIPO_SEGLOGR",
    "NOM_TITULO_SEGLOGR",
    "NOM_SEGLOGR",
    "NUM_ENDERECO",
    "DSC_MODIFICADOR",
    "NOM_COMP_ELEM1",
    "VAL_COMP_ELEM1",
    "NOM_COMP_ELEM2",
    "VAL_COMP_ELEM2",
    "NOM_COMP_ELEM3",
    "VAL_COMP_ELEM3",
    "NOM_COMP_ELEM4",
    "VAL_COMP_ELEM4",
    "NOM_COMP_ELEM5",
    "VAL_COMP_ELEM5",
    "LATITUDE",
    "LONGITUDE",
    "NV_GEO_COORD",
    "COD_ESPECIE",
    "DSC_ESTABELECIMENTO",
    "COD_INDICADOR_ESTAB_ENDERECO",
    "COD_INDICADOR_CONST_ENDERECO",
    "COD_INDICADOR_FINALIDADE_CONST",
    "COD_TIPO_ESPECI",
)

CNEFE_SELECTED_COLUMNS = (
    "COD_UNICO_ENDERECO",
    "COD_UF",
    "COD_MUNICIPIO",
    "COD_DISTRITO",
    "COD_SUBDISTRITO",
    "COD_SETOR",
    "CEP",
    "DSC_LOCALIDADE",
    "NOM_TIPO_SEGLOGR",
    "NOM_TITULO_SEGLOGR",
    "NOM_SEGLOGR",
    "NUM_ENDERECO",
    "DSC_MODIFICADOR",
    "NOM_COMP_ELEM1",
    "VAL_COMP_ELEM1",
    "NOM_COMP_ELEM2",
    "VAL_COMP_ELEM2",
    "NOM_COMP_ELEM3",
    "VAL_COMP_ELEM3",
    "NOM_COMP_ELEM4",
    "VAL_COMP_ELEM4",
    "NOM_COMP_ELEM5",
    "VAL_COMP_ELEM5",
)

SEARCH_COLUMNS = (
    "consulta_logradouro",
    "consulta_numero",
    "consulta_bairro",
    "consulta_cep",
    "consulta_complemento",
    "consulta_municipio",
)

SEARCH_ID_COLUMN = "search_id"

CNEFE_SPARK_SCHEMA = StructType(
    [StructField(name, StringType(), nullable=True) for name in CNEFE_COLUMNS]
)
