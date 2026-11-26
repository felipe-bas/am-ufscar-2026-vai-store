# ################################################################
# PROJETO FINAL
#
# Universidade Federal de Sao Carlos (UFSCAR)
# Departamento de Computacao - Sorocaba (DComp-So)
# Disciplina: Aprendizado de Maquina
# Prof. Tiago A. Almeida
#
#
# Nome: Felipe Rodrigues Bastos (RA: 815406) - felipe.bastos@estudante.ufscar.br
# Nome: João Henrique de Camargo Brants (RA: 802553) - joaobrants@estudante.ufscar.br
# RA: 815406 / 802553
# ################################################################

# Arquivo com todas as funcoes e codigos referentes ao preprocessamento e engenharia de atributos

from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import gammaln
from scipy.stats import norm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    SplineTransformer,
    StandardScaler,
    TargetEncoder,
)


CUTOFF_DEFAULT = "2024-09-30"
DATA_INICIO_DEFAULT = "2024-01-01"

TOP_CATEGORIAS_VENDAS = [
    "Temperos & Condimentos",
    "Castanhas & Oleaginosas",
    "Grãos",
    "Mixes da Casa",
    "Matinais",
    "Frutas",
    "Lanchonete",
    "Bacalhau & Pescados",
    "Superalimentos & Energéticos",
    "Ervas & Chás",
    "Conservas",
    "Azeites & Vinagres",
]

TOP_12_SKUS = [13805, 2987, 199, 191, 6720, 17048, 13778, 1062, 15844, 264, 86, 16923]

MAPA_CATEGORIAS_AUSENTES_SKU = {
    264: ("Frutas", "Frutas Secas"),
    3407: ("Castanhas & Oleaginosas", "Castanhas do Para"),
    2033: ("Bacalhau & Pescados", "Bacalhau"),
    3641: ("Bacalhau & Pescados", "Bacalhau"),
    23243: ("Bacalhau & Pescados", "Bacalhau"),
}

BGNBD_FEATURE_NAMES = [
    "bgnbd_p_alive",
    "bgnbd_lam_day",
    "bgnbd_expected_tx_92d",
    "nb_p0",
    "nb_p1",
    "nb_p2",
    "nb_p3",
    "log_nb_p0",
    "log_nb_p1",
    "log_nb_p2",
    "log_nb_p3",
    "rec_p0",
    "rec_p1",
    "rec_p2",
    "rec_p3",
    "ren_p0",
    "ren_p1",
    "ren_p2",
    "ren_p3",
]

SPLINE_KEY_COLS = [
    "recencia_dias",
    "antiguidade_dias",
    "ciclo_medio_dias",
    "recencia_sobre_ciclo",
    "janela_frac_c0",
    "janela_frac_c1",
    "janela_frac_c2",
    "log1p_faturamento",
    "log1p_n_compras",
]


def _slugify(texto: str) -> str:
    norm = unicodedata.normalize("NFKD", str(texto)).encode("ascii", "ignore").decode("ascii")
    norm = re.sub(r"[^a-zA-Z0-9]+", "_", norm.lower()).strip("_")
    return norm


def _max_consecutive_ones_9bit(mask: int) -> int:
    melhor = 0
    atual = 0
    for bit in range(9):
        if (mask >> bit) & 1:
            atual += 1
            if atual > melhor:
                melhor = atual
        else:
            atual = 0
    return melhor


_STREAK_LOOKUP = np.array([_max_consecutive_ones_9bit(m) for m in range(512)], dtype=np.int8)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Limita valores extremos por quantil calculado apenas no conjunto de treino."""

    def __init__(self, lower_q: float = 0.0, upper_q: float = 0.995):
        self.lower_q = lower_q
        self.upper_q = upper_q

    def fit(self, X, y=None):
        arr = np.asarray(X, dtype=np.float64)
        self.n_features_in_ = arr.shape[1]
        self.lower_bounds_ = np.nanquantile(arr, self.lower_q, axis=0) if self.lower_q > 0.0 else None
        self.upper_bounds_ = np.nanquantile(arr, self.upper_q, axis=0) if self.upper_q < 1.0 else None
        return self

    def transform(self, X):
        arr = np.asarray(X, dtype=np.float64).copy()
        if self.lower_bounds_ is not None:
            for j in range(arr.shape[1]):
                val = self.lower_bounds_[j]
                if np.isfinite(val):
                    mask = ~np.isnan(arr[:, j]) & (arr[:, j] < val)
                    arr[mask, j] = val
        if self.upper_bounds_ is not None:
            for j in range(arr.shape[1]):
                val = self.upper_bounds_[j]
                if np.isfinite(val):
                    mask = ~np.isnan(arr[:, j]) & (arr[:, j] > val)
                    arr[mask, j] = val
        return arr

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray([f"x{i}" for i in range(self.n_features_in_)], dtype=object)
        return np.asarray(input_features, dtype=object)


class BGNBDFeatureTransformer(BaseEstimator, TransformerMixin):
    """
    Ajusta o modelo probabilistico BG/NBD (Fader, Hardie & Lee 2005) exclusivamente
    sobre o historico RFM pre-cutoff (sem usar y!) no fold de treino e gera 19 atributos
    analiticos de sobrevivencia, Binomial Negativa e Renovacao Log-Normal.
    """

    def __init__(self, reg_l2: float = 1e-4):
        self.reg_l2 = reg_l2

    @staticmethod
    def _extract_rfm(X):
        if isinstance(X, pd.DataFrame):
            n_dias = X["n_dias_com_compra"].to_numpy(dtype=np.float64)
            antig = X["antiguidade_dias"].to_numpy(dtype=np.float64)
            rec = X["recencia_dias"].to_numpy(dtype=np.float64)
            ciclo = X["ciclo_medio_dias"].to_numpy(dtype=np.float64)
            var_c = X["variacao_ciclo"].to_numpy(dtype=np.float64)
            v90 = X["visitas_90d"].to_numpy(dtype=np.float64) if "visitas_90d" in X.columns else n_dias
        else:
            arr = np.asarray(X, dtype=np.float64)
            n_dias, antig, rec, ciclo, var_c = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3], arr[:, 4]
            v90 = arr[:, 5] if arr.shape[1] >= 6 else n_dias
        return n_dias, antig, rec, ciclo, var_c, v90

    def fit(self, X, y=None):
        # y e explicitamente ignorado: ajuste 100% nao supervisionado sobre RFM pre-cutoff
        n_dias, antig, rec, _, _, _ = self._extract_rfm(X)
        x = np.maximum(n_dias - 1.0, 0.0)
        T_w = (np.maximum(antig, 0.0) + 1.0) / 7.0
        tx_w = np.maximum(antig - rec, 0.0) / 7.0
        has_rep = x > 0
        reg = self.reg_l2

        def neg_ll(log_p):
            r, alpha, a, b = np.exp(np.clip(log_p, -8.0, 8.0))
            a1 = gammaln(r + x) - gammaln(r) + r * np.log(alpha)
            a2 = gammaln(a + b) + gammaln(b + x) - gammaln(b) - gammaln(a + b + x)
            a3 = -(r + x) * np.log(alpha + T_w)
            a4 = np.zeros_like(x)
            a4[has_rep] = (
                np.log(a)
                - np.log(b + np.maximum(x[has_rep] - 1.0, 1e-9))
                - (r + x[has_rep]) * np.log(alpha + tx_w[has_rep])
            )
            ll = a1 + a2 + np.where(has_rep, np.logaddexp(a3, a4), a3)
            return -float(np.mean(ll)) + reg * float(np.sum(log_p ** 2))

        res = minimize(neg_ll, np.log([0.5, 5.0, 0.5, 2.0]), method="L-BFGS-B")
        self.params_ = tuple(np.exp(np.clip(res.x, -8.0, 8.0)).tolist())
        return self

    def transform(self, X):
        r, alpha, a, b = self.params_
        n_dias, antig, rec, ciclo, var_c, v90 = self._extract_rfm(X)

        x = np.maximum(n_dias - 1.0, 0.0)
        T_d = np.maximum(antig, 0.0) + 1.0
        T_w = T_d / 7.0
        tx_w = np.maximum(antig - rec, 0.0) / 7.0
        has_rep = x > 0

        log_ratio = np.where(
            has_rep,
            np.log(a)
            - np.log(b + np.maximum(x - 1.0, 1e-9))
            + (r + x) * (np.log(alpha + T_w) - np.log(alpha + tx_w)),
            -50.0,
        )
        p_alive = np.where(has_rep, 1.0 / (1.0 + np.exp(np.clip(log_ratio, -50.0, 50.0))), 1.0)
        lam_day = (r + x) / ((alpha + T_w) * 7.0)
        expected_tx_92d = p_alive * lam_day * 92.0

        ciclo_ef = np.where(np.isfinite(ciclo) & (ciclo > 0), ciclo, 46.0)
        t0 = np.minimum(1.2 * ciclo_ef, 92.0) / 7.0
        t1 = np.minimum(2.0 * ciclo_ef, 92.0) / 7.0
        t2 = np.full_like(t0, 92.0 / 7.0)

        def _s_nb(dt_w):
            return ((alpha + T_w) / (alpha + T_w + dt_w)) ** (r + x)

        nb_p0 = p_alive * (1.0 - _s_nb(t0))
        nb_p1 = p_alive * (_s_nb(t0) - _s_nb(t1))
        nb_p2 = np.where(has_rep, p_alive * (_s_nb(t1) - _s_nb(t2)), 0.0)
        nb_p3 = np.clip(1.0 - (nb_p0 + nb_p1 + nb_p2), 0.0, 1.0)

        log_nb_p0 = np.log(np.clip(nb_p0, 1e-6, 1.0))
        log_nb_p1 = np.log(np.clip(nb_p1, 1e-6, 1.0))
        log_nb_p2 = np.log(np.clip(nb_p2, 1e-6, 1.0))
        log_nb_p3 = np.log(np.clip(nb_p3, 1e-6, 1.0))

        # Taxa recente nos ultimos 90 dias
        x_rec = np.maximum(np.where(np.isfinite(v90), v90, n_dias) - 1.0, 0.0)
        T_rec_w = np.minimum(T_d, 90.0) / 7.0

        def _s_rec(dt_w):
            return ((alpha + T_rec_w) / (alpha + T_rec_w + dt_w)) ** (r + x_rec)

        rec_p0 = p_alive * (1.0 - _s_rec(t0))
        rec_p1 = p_alive * (_s_rec(t0) - _s_rec(t1))
        rec_p2 = np.where(has_rep, p_alive * (_s_rec(t1) - _s_rec(t2)), 0.0)
        rec_p3 = np.clip(1.0 - (rec_p0 + rec_p1 + rec_p2), 0.0, 1.0)

        # Processo de Renovacao Log-Normal condicional a ja ter esperado rec dias
        rec_pos = np.maximum(rec, 0.0)
        cv_est = np.where(np.isfinite(var_c) & (ciclo_ef > 0), var_c / (ciclo_ef + 1e-5), 0.65)
        sigma_log = np.clip(cv_est, 0.35, 1.2)
        mu_log = np.log(np.maximum(ciclo_ef, 1.0))

        f_rec = norm.cdf((np.log(np.maximum(rec_pos, 0.5)) - mu_log) / sigma_log)
        s_rec = np.maximum(1.0 - f_rec, 1e-4)

        def _cond_cdf(t_end_dias):
            f_end = norm.cdf((np.log(np.maximum(t_end_dias, 0.5)) - mu_log) / sigma_log)
            return np.clip((f_end - f_rec) / s_rec, 0.0, 1.0)

        c0_end = np.minimum(1.2 * ciclo_ef, rec_pos + 92.0)
        c1_end = np.minimum(2.0 * ciclo_ef, rec_pos + 92.0)
        c2_end = rec_pos + 92.0

        ren_cdf0 = np.where(c0_end > rec_pos, _cond_cdf(c0_end), 0.0)
        ren_cdf1 = np.where(c1_end > rec_pos, _cond_cdf(c1_end), 0.0)
        ren_cdf2 = _cond_cdf(c2_end)

        ren_p0 = p_alive * ren_cdf0
        ren_p1 = p_alive * np.maximum(ren_cdf1 - ren_cdf0, 0.0)
        ren_p2 = np.where(has_rep, p_alive * np.maximum(ren_cdf2 - ren_cdf1, 0.0), 0.0)
        ren_p3 = np.clip(1.0 - (ren_p0 + ren_p1 + ren_p2), 0.0, 1.0)

        mat = np.column_stack([
            p_alive, lam_day, expected_tx_92d,
            nb_p0, nb_p1, nb_p2, nb_p3,
            log_nb_p0, log_nb_p1, log_nb_p2, log_nb_p3,
            rec_p0, rec_p1, rec_p2, rec_p3,
            ren_p0, ren_p1, ren_p2, ren_p3,
        ])
        return np.nan_to_num(mat, nan=0.0, posinf=1.0, neginf=-14.0)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(BGNBD_FEATURE_NAMES, dtype=object)


ARQUIVOS_DADOS_OBRIGATORIOS = (
    "train.csv",
    "test.csv",
    "sample_submission.csv",
    "vendas_dirty.csv",
    "produto_dirty.csv",
)


def resolver_diretorio_dados(data_dir_config: str | Path = "ufs-car-am-2026-projeto-final") -> Path:
    """
    Resolve o diretorio que contem os 5 CSVs originais do projeto.
    Verifica primeiro o caminho configurado e faz fallback automatico para diretorios comuns.
    """
    brutos = [
        Path(data_dir_config),
        Path("."),
        Path("data"),
        Path("dados"),
        Path("bases"),
        Path("dataset"),
        Path("..") / "ufs-car-am-2026-projeto-final",
        Path("..") / "data",
        Path("..") / "dados",
        Path(".."),
    ]
    candidatos: List[Path] = []
    vistos = set()
    for p in brutos:
        chave = str(p.resolve()) if p.exists() else str(p)
        if chave not in vistos:
            vistos.add(chave)
            candidatos.append(p)

    for cand in candidatos:
        if cand.is_dir() and all((cand / arq).exists() for arq in ARQUIVOS_DADOS_OBRIGATORIOS):
            return cand

    faltando_cfg = [arq for arq in ARQUIVOS_DADOS_OBRIGATORIOS if not (Path(data_dir_config) / arq).exists()]
    raise FileNotFoundError(
        f"Bases originais nao encontradas. Caminho configurado: '{data_dir_config}' "
        f"(ausentes: {faltando_cfg}). Diretorios verificados: {[str(c) for c in candidatos]}. "
        f"Arquivos obrigatorios: {list(ARQUIVOS_DADOS_OBRIGATORIOS)}."
    )


def carregar_bases(data_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Carrega train.csv, test.csv e sample_submission.csv com validacoes basicas."""
    base_path = Path(data_dir)
    train_df = pd.read_csv(base_path / "train.csv")
    test_df = pd.read_csv(base_path / "test.csv")
    sample_sub = pd.read_csv(base_path / "sample_submission.csv")

    train_df.columns = train_df.columns.str.strip()
    test_df.columns = test_df.columns.str.strip()
    sample_sub.columns = sample_sub.columns.str.strip()

    for df in (train_df, test_df):
        df["CLI_CPF"] = df["CLI_CPF"].astype(str).str.strip()
        df["filial"] = (
            df["filial"]
            .astype("string")
            .str.strip()
            .str.upper()
            .replace("", pd.NA)
            .astype(object)
        )
        df["filial"] = df["filial"].where(pd.notna(df["filial"]), np.nan)

    assert train_df["CLI_CPF"].duplicated().sum() == 0, "CPFs duplicados em train.csv"
    assert test_df["CLI_CPF"].duplicated().sum() == 0, "CPFs duplicados em test.csv"
    assert set(train_df["target"].unique()).issubset({0, 1, 2, 3}), "Classes invalidas em target"
    assert (test_df["CLI_CPF"].values == sample_sub["CLI_CPF"].astype(str).str.strip().values).all(), (
        "Ordem de CLI_CPF diverge entre test.csv e sample_submission.csv"
    )

    train_df["target"] = train_df["target"].astype(int)
    return train_df, test_df, sample_sub


def limpar_vendas_e_produtos_pre_cutoff(
    data_dir: str | Path,
    cutoff: str = CUTOFF_DEFAULT,
    data_inicio: str = DATA_INICIO_DEFAULT,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int | str]]:
    """Limpa vendas_dirty.csv e produto_dirty.csv aplicando corte temporal estrito <= cutoff."""
    base_path = Path(data_dir)
    vendas = pd.read_csv(base_path / "vendas_dirty.csv", low_memory=False)
    prod = pd.read_csv(base_path / "produto_dirty.csv", low_memory=False)

    resumo: Dict[str, int | str] = {"vendas_linhas_brutas": int(len(vendas))}

    dt_raw = pd.to_datetime(vendas["DATA_ATEND"].astype(str).str.strip(), format="%Y-%m-%d", errors="coerce")
    dt_cod = pd.to_datetime(vendas["COD_ATEND"].astype(str).str.split(".").str[-1], format="%Y%m%d", errors="coerce")

    datas_corrompidas = int(((dt_raw < "2024-01-01") | (dt_raw > "2024-12-31") | dt_raw.isna()).sum())
    vendas["dt"] = dt_cod.where(dt_cod.notna(), dt_raw)
    resumo["datas_corrompidas_recuperadas_via_cod_atend"] = datas_corrompidas

    dt_ini = pd.Timestamp(data_inicio)
    dt_cut = pd.Timestamp(cutoff)
    vendas = vendas[(vendas["dt"] >= dt_ini) & (vendas["dt"] <= dt_cut)].copy()
    assert vendas["dt"].max() <= dt_cut, "VAZAMENTO TEMPORAL DETECTADO!"
    resumo["vendas_linhas_pre_cutoff"] = int(len(vendas))

    vendas["CLI_CPF"] = vendas["CLI_CPF"].astype("string").str.strip()
    n_sem_cpf = int(vendas["CLI_CPF"].isna().sum())
    vendas = vendas[vendas["CLI_CPF"].notna() & (vendas["CLI_CPF"] != "")].copy()
    vendas["CLI_CPF"] = vendas["CLI_CPF"].astype(str)
    resumo["linhas_pre_cutoff_sem_cpf_removidas"] = n_sem_cpf

    n_dup_exata = int(vendas.duplicated().sum())
    vendas = vendas.drop_duplicates().copy()
    n_dup_cod_venda = int(vendas["COD_VENDA"].duplicated().sum())
    vendas = vendas.drop_duplicates(subset=["COD_VENDA"], keep="first").copy()
    resumo["duplicatas_exatas_removidas"] = n_dup_exata
    resumo["duplicatas_cod_venda_removidas"] = n_dup_cod_venda

    mapa_filial = {101032: "RUA", 101042: "SHOPPING"}
    filial_raw = vendas["FILIAL"].astype("string").str.strip().str.upper()
    n_filial_suja = int((~filial_raw.isin(["RUA", "SHOPPING"])).sum())
    vendas["FILIAL_LIMPA"] = vendas["COD_FILIAL"].map(mapa_filial).fillna("OUTRA")
    resumo["filiais_sujas_corrigidas_via_cod_filial"] = n_filial_suja

    unid_norm = vendas["UNID"].astype(str).str.strip().str.upper().str.replace(".", "", regex=False)
    vendas["UNID_LIMPA"] = np.where(unid_norm == "UN", "UN", "KG")

    qtd = pd.to_numeric(vendas["QTD_VENDA"], errors="coerce")
    fat = pd.to_numeric(vendas["FATUR_VENDA"], errors="coerce")
    resumo["qtd_invalida_ou_nula"] = int((qtd.isna() | (qtd <= 0)).sum())
    resumo["fatur_invalido_ou_nulo"] = int((fat.isna() | (fat <= 0)).sum())

    qtd_pos = qtd.where(qtd > 0, np.nan)
    q9995_kg = float(qtd_pos[vendas["UNID_LIMPA"] == "KG"].quantile(0.9995))
    q9995_un = float(qtd_pos[vendas["UNID_LIMPA"] == "UN"].quantile(0.9995))
    qtd_cap = np.where(vendas["UNID_LIMPA"] == "UN", np.minimum(qtd_pos, q9995_un), np.minimum(qtd_pos, q9995_kg))
    vendas["qtd_limpa"] = np.where(qtd_pos.isna(), np.nan, qtd_cap)
    vendas["fatur_limpo"] = fat.where(fat > 0, np.nan)
    vendas["SKU"] = pd.to_numeric(vendas["SKU"], errors="coerce").fillna(-1).astype(int)

    prod_limpo = prod.copy()
    prod_limpo = prod_limpo[prod_limpo["SKU"].notna()].copy()
    prod_limpo["SKU"] = pd.to_numeric(prod_limpo["SKU"], errors="coerce").astype(int)
    prod_limpo = prod_limpo.drop_duplicates(subset=["SKU"], keep="first").copy()
    prod_limpo["NOME_PRODUTO"] = prod_limpo["NOME_PRODUTO"].fillna("DESCONHECIDO").astype(str).str.strip()
    prod_limpo["CATEGORIA"] = prod_limpo["CATEGORIA"].astype("string").str.strip()
    prod_limpo["SUBCATEGORIA"] = prod_limpo["SUBCATEGORIA"].astype("string").str.strip()

    for sku_id, (cat_fix, sub_fix) in MAPA_CATEGORIAS_AUSENTES_SKU.items():
        mask_sku = prod_limpo["SKU"] == sku_id
        if mask_sku.any():
            prod_limpo.loc[mask_sku & prod_limpo["CATEGORIA"].isna(), "CATEGORIA"] = cat_fix
            prod_limpo.loc[mask_sku & prod_limpo["SUBCATEGORIA"].isna(), "SUBCATEGORIA"] = sub_fix

    prod_limpo["CATEGORIA"] = prod_limpo["CATEGORIA"].fillna("DESCONHECIDA").astype(str)
    prod_limpo["SUBCATEGORIA"] = prod_limpo["SUBCATEGORIA"].fillna("DESCONHECIDA").astype(str)

    resumo["vendas_linhas_limpas_pre_cutoff"] = int(len(vendas))
    resumo["produtos_unicos_limpos"] = int(len(prod_limpo))
    return vendas, prod_limpo, resumo


def construir_atributos_base(df: pd.DataFrame) -> pd.DataFrame:
    """Grupo A: cria atributos estruturais e comportamentais usando apenas colunas de train/test."""
    out = df.copy()
    eps = 1e-5

    out["is_single_day"] = (out["n_dias_com_compra"] == 1).astype(np.int8)
    out["is_single_purchase"] = (out["n_compras"] == 1).astype(np.int8)
    out["variacao_ciclo_ausente"] = out["variacao_ciclo"].isna().astype(np.int8)
    out["can_be_class2"] = ((out["n_dias_com_compra"] >= 2) & (out["ciclo_medio_dias"] <= 46.5)).astype(np.int8)
    out["can_be_class1"] = ((out["n_dias_com_compra"] == 1) | (out["ciclo_medio_dias"] <= 77.0)).astype(np.int8)

    ciclo = out["ciclo_medio_dias"].astype(np.float64)
    out["recencia_sobre_ciclo"] = out["recencia_dias"] / (ciclo + eps)
    out["atraso_dias_abs"] = out["recencia_dias"] - ciclo
    out["atraso_zscore"] = (out["recencia_dias"] - ciclo) / (out["variacao_ciclo"] + 1.0)
    out["cv_ciclo"] = out["variacao_ciclo"] / (ciclo + eps)
    out["max_delay_ratio_92d"] = 92.0 / (ciclo + eps)
    out["dias_esperados_prox_compra"] = ciclo - out["recencia_dias"]

    ciclo_ef = ciclo.fillna(46.0)
    lim_c0 = np.minimum(1.2 * ciclo_ef, 92.0)
    lim_c1 = np.minimum(2.0 * ciclo_ef, 92.0)
    out["janela_frac_c0"] = np.clip(lim_c0 / 92.0, 0.0, 1.0)
    out["janela_frac_c1"] = np.clip((lim_c1 - lim_c0) / 92.0, 0.0, 1.0)
    out["janela_frac_c2"] = np.where(
        out["n_dias_com_compra"] == 1, 0.0, np.clip((92.0 - lim_c1) / 92.0, 0.0, 1.0)
    )

    out["span_ativo_dias"] = np.maximum(out["antiguidade_dias"] - out["recencia_dias"], 0.0)
    out["ratio_recencia_antiguidade"] = out["recencia_dias"] / (out["antiguidade_dias"] + 1.0)
    out["frequencia_mensal"] = out["n_compras"] / ((out["antiguidade_dias"] + 1.0) / 30.0)
    out["taxa_dias_ativos"] = out["n_dias_com_compra"] / (out["antiguidade_dias"] + 1.0)
    out["compras_por_dia_compra"] = out["n_compras"] / np.maximum(out["n_dias_com_compra"], 1.0)
    out["itens_por_compra"] = out["total_itens"] / np.maximum(out["n_compras"], 1.0)
    out["sku_por_compra"] = out["n_sku"] / np.maximum(out["n_compras"], 1.0)
    out["sku_por_categoria"] = out["n_sku"] / np.maximum(out["n_categorias"], 1.0)
    out["faturamento_por_item"] = out["faturamento"] / (out["total_itens"] + eps)
    out["faturamento_por_dia_antig"] = out["faturamento"] / (out["antiguidade_dias"] + 1.0)
    out["faturamento_por_sku"] = out["faturamento"] / np.maximum(out["n_sku"], 1.0)
    out["tendencia_relativa"] = out["tendencia_faturamento"] / (out["ticket_medio"] + eps)
    out["tendencia_ausente"] = out["tendencia_faturamento"].isna().astype(np.int8)

    out["log1p_faturamento"] = np.log1p(np.maximum(out["faturamento"], 0.0))
    out["log1p_ticket_medio"] = np.log1p(np.maximum(out["ticket_medio"], 0.0))
    out["log1p_total_itens"] = np.log1p(np.maximum(out["total_itens"], 0.0))
    out["log1p_n_compras"] = np.log1p(np.maximum(out["n_compras"], 0.0))
    out["filial_rua"] = (out["filial"] == "RUA").astype(np.int8)

    return out


def construir_atributos_transacionais(
    vendas_pre: pd.DataFrame,
    prod_limpo: pd.DataFrame,
    cutoff: str = CUTOFF_DEFAULT,
) -> pd.DataFrame:
    """Grupo B: extrai atributos temporais, janelas curtas, gaps, SKUs, caixas e entropia 100% vetorizados."""
    dt_cut = pd.Timestamp(cutoff)
    assert vendas_pre["dt"].max() <= dt_cut, "VAZAMENTO TEMPORAL DETECTADO em vendas_pre!"

    v = vendas_pre.merge(prod_limpo[["SKU", "CATEGORIA", "SUBCATEGORIA"]], on="SKU", how="left")
    v["CATEGORIA"] = v["CATEGORIA"].fillna("DESCONHECIDA")
    v["SUBCATEGORIA"] = v["SUBCATEGORIA"].fillna("DESCONHECIDA")
    v["fatur_zero"] = v["fatur_limpo"].fillna(0.0)
    v["qtd_zero"] = v["qtd_limpa"].fillna(0.0)
    v["mes"] = v["dt"].dt.month.astype(np.int8)
    v["dow"] = v["dt"].dt.dayofweek.astype(np.int8)
    v["dia_mes"] = v["dt"].dt.day.astype(np.int8)
    v["doy"] = v["dt"].dt.dayofyear.astype(np.int16)
    v["is_weekend"] = (v["dow"] >= 5).astype(np.int8)
    v["is_domingo"] = (v["dow"] == 6).astype(np.int8)
    v["is_inicio_mes"] = (v["dia_mes"] <= 10).astype(np.int8)
    v["is_fim_mes"] = (v["dia_mes"] >= 21).astype(np.int8)
    v["is_rua"] = (v["FILIAL_LIMPA"] == "RUA").astype(np.int8)
    v["is_un"] = (v["UNID_LIMPA"] == "UN").astype(np.int8)
    v["is_linha_invalida"] = (v["qtd_limpa"].isna() | v["fatur_limpo"].isna()).astype(np.int8)
    v["caixa"] = v["COD_ATEND"].astype(str).str.split(".").str[1].fillna("NA")
    v["is_bacalhau_pascoa"] = ((v["CATEGORIA"] == "Bacalhau & Pescados") & (v["mes"].isin([3, 4]))).astype(np.int8)
    v["is_semana_pascoa"] = ((v["dt"] >= "2024-03-15") & (v["dt"] <= "2024-03-31")).astype(np.int8)
    v["fat_kg"] = np.where(v["UNID_LIMPA"] == "KG", v["fatur_zero"], 0.0)
    v["qtd_kg"] = np.where(v["UNID_LIMPA"] == "KG", v["qtd_zero"], 0.0)

    # 1. Nivel de visita (COD_ATEND)
    visitas = (
        v.groupby(["CLI_CPF", "COD_ATEND", "dt"], sort=False, as_index=False)
        .agg(
            fat_visita=("fatur_zero", "sum"),
            itens_visita=("qtd_zero", "sum"),
            linhas_visita=("COD_VENDA", "count"),
            n_skus_visita=("SKU", "nunique"),
            is_rua=("is_rua", "max"),
            is_weekend=("is_weekend", "max"),
            is_domingo=("is_domingo", "max"),
            is_inicio_mes=("is_inicio_mes", "max"),
            is_fim_mes=("is_fim_mes", "max"),
            mes=("mes", "first"),
            dow=("dow", "first"),
            dia_mes=("dia_mes", "first"),
            doy=("doy", "first"),
            filial=("FILIAL_LIMPA", "first"),
            caixa=("caixa", "first"),
        )
        .sort_values(["CLI_CPF", "dt", "COD_ATEND"], kind="mergesort")
        .reset_index(drop=True)
    )

    dias_cut = (dt_cut - visitas["dt"]).dt.days.astype(np.int16)
    visitas["dias_ate_cutoff"] = dias_cut
    for d_win in (7, 14, 21, 30, 45, 60, 90, 180):
        visitas[f"v_{d_win}d"] = (dias_cut <= d_win).astype(np.int8)
    visitas["v_q1"] = (visitas["mes"] <= 3).astype(np.int8)
    visitas["v_q2"] = ((visitas["mes"] >= 4) & (visitas["mes"] <= 6)).astype(np.int8)
    visitas["v_q3"] = (visitas["mes"] >= 7).astype(np.int8)

    for d_win in (14, 30, 45, 60, 90):
        visitas[f"f_{d_win}d"] = visitas["fat_visita"] * visitas[f"v_{d_win}d"]
    visitas["f_q1"] = visitas["fat_visita"] * visitas["v_q1"]
    visitas["f_q2"] = visitas["fat_visita"] * visitas["v_q2"]
    visitas["f_q3"] = visitas["fat_visita"] * visitas["v_q3"]
    visitas["prev_rec"] = visitas.groupby("CLI_CPF", sort=False)["dias_ate_cutoff"].shift(1)
    visitas["prev_fat"] = visitas.groupby("CLI_CPF", sort=False)["fat_visita"].shift(1)
    visitas["prev_lines"] = visitas.groupby("CLI_CPF", sort=False)["linhas_visita"].shift(1)

    cli_vis = visitas.groupby("CLI_CPF", sort=False).agg(
        visitas_vendas=("COD_ATEND", "count"),
        visitas_7d=("v_7d", "sum"),
        visitas_14d=("v_14d", "sum"),
        visitas_21d=("v_21d", "sum"),
        visitas_30d=("v_30d", "sum"),
        visitas_45d=("v_45d", "sum"),
        visitas_60d=("v_60d", "sum"),
        visitas_90d=("v_90d", "sum"),
        visitas_180d=("v_180d", "sum"),
        visitas_q1=("v_q1", "sum"),
        visitas_q2=("v_q2", "sum"),
        visitas_q3=("v_q3", "sum"),
        fat_14d=("f_14d", "sum"),
        fat_30d=("f_30d", "sum"),
        fat_45d=("f_45d", "sum"),
        fat_60d=("f_60d", "sum"),
        fat_90d=("f_90d", "sum"),
        fat_q1=("f_q1", "sum"),
        fat_q2=("f_q2", "sum"),
        fat_q3=("f_q3", "sum"),
        fat_total_vendas=("fat_visita", "sum"),
        linhas_total=("linhas_visita", "sum"),
        linhas_por_visita=("linhas_visita", "mean"),
        fat_ultima_visita=("fat_visita", "last"),
        fat_primeira_visita=("fat_visita", "first"),
        std_fat_visita=("fat_visita", "std"),
        max_fat_visita=("fat_visita", "max"),
        min_fat_visita=("fat_visita", "min"),
        pct_visitas_rua=("is_rua", "mean"),
        n_filiais_distintas=("filial", "nunique"),
        pct_fim_de_semana=("is_weekend", "mean"),
        pct_domingo=("is_domingo", "mean"),
        pct_inicio_mes=("is_inicio_mes", "mean"),
        pct_fim_mes=("is_fim_mes", "mean"),
        dow_ultima_visita=("dow", "last"),
        dia_mes_ultima_visita=("dia_mes", "last"),
        dia_do_ano_ultima=("doy", "last"),
        dia_do_ano_primeira=("doy", "first"),
        ultimo_mes=("mes", "last"),
        primeiro_mes=("mes", "first"),
        n_meses_ativos=("mes", "nunique"),
        n_caixas_distintos=("caixa", "nunique"),
        rec_2nd_last=("prev_rec", "last"),
        fat_2nd_last=("prev_fat", "last"),
        lines_last=("linhas_visita", "last"),
        lines_2nd_last=("prev_lines", "last"),
        max_n_skus_visita=("n_skus_visita", "max"),
        std_n_skus_visita=("n_skus_visita", "std"),
    )

    # Sinais ciclicos / calendario da ultima visita
    cli_vis["ultima_visita_domingo"] = (cli_vis["dow_ultima_visita"] == 6).astype(np.int8)
    cli_vis["ultima_visita_qui_sex"] = cli_vis["dow_ultima_visita"].isin([3, 4]).astype(np.int8)
    cli_vis["sin_mes_ultima"] = np.sin(2.0 * np.pi * cli_vis["ultimo_mes"] / 12.0)
    cli_vis["cos_mes_ultima"] = np.cos(2.0 * np.pi * cli_vis["ultimo_mes"] / 12.0)

    # Tendencia linear com sinal real (slope_fat_real) vetorizada
    visitas["v_idx"] = visitas.groupby("CLI_CPF", sort=False).cumcount().astype(np.float64)
    mean_x = visitas.groupby("CLI_CPF", sort=False)["v_idx"].transform("mean")
    mean_y = visitas.groupby("CLI_CPF", sort=False)["fat_visita"].transform("mean")
    visitas["_dx2"] = (visitas["v_idx"] - mean_x) ** 2
    visitas["_dxdy"] = (visitas["v_idx"] - mean_x) * (visitas["fat_visita"] - mean_y)
    slope_df = visitas.groupby("CLI_CPF", sort=False)[["_dxdy", "_dx2"]].sum()
    cli_vis["slope_fat_real"] = np.where(slope_df["_dx2"] > 0, slope_df["_dxdy"] / slope_df["_dx2"], np.nan)

    # Streak maximo de meses consecutivos ativos via bitmask de 9 bits
    meses_unicos = visitas[["CLI_CPF", "mes"]].drop_duplicates().copy()
    meses_unicos["bit"] = np.left_shift(1, meses_unicos["mes"].astype(np.int32) - 1)
    bitmasks = meses_unicos.groupby("CLI_CPF", sort=False)["bit"].sum().astype(np.int32)
    cli_vis["max_streak_meses"] = _STREAK_LOOKUP[np.clip(bitmasks.values, 0, 511)]
    cli_vis["comprou_setembro"] = ((bitmasks.values >> 8) & 1).astype(np.int8)
    cli_vis["comprou_agosto"] = ((bitmasks.values >> 7) & 1).astype(np.int8)
    cli_vis["prop_meses_ativos"] = cli_vis["n_meses_ativos"] / 9.0
    cli_vis["trocou_filial"] = (cli_vis["n_filiais_distintas"] > 1).astype(np.int8)

    eps = 1e-5
    cli_vis["ratio_visitas_30_90"] = cli_vis["visitas_30d"] / (cli_vis["visitas_90d"] + eps)
    cli_vis["ratio_visitas_45_90"] = cli_vis["visitas_45d"] / (cli_vis["visitas_90d"] + eps)
    cli_vis["ratio_visitas_90_total"] = cli_vis["visitas_90d"] / (cli_vis["visitas_vendas"] + eps)
    cli_vis["ratio_visitas_q3_q12"] = (cli_vis["visitas_q3"] + 0.5) / (cli_vis["visitas_q1"] + cli_vis["visitas_q2"] + 0.5)
    cli_vis["ratio_fat_30_90"] = cli_vis["fat_30d"] / (cli_vis["fat_90d"] + eps)
    cli_vis["ratio_fat_90_total"] = cli_vis["fat_90d"] / (cli_vis["fat_total_vendas"] + eps)
    cli_vis["ratio_fat_q3_q12"] = (cli_vis["fat_q3"] + 1.0) / (cli_vis["fat_q1"] + cli_vis["fat_q2"] + 1.0)

    # 2. Intervalos reais entre dias distintos de compra (gaps)
    dias_unicos = visitas[["CLI_CPF", "dt"]].drop_duplicates().sort_values(["CLI_CPF", "dt"], kind="mergesort")
    dias_unicos["prev_dt"] = dias_unicos.groupby("CLI_CPF", sort=False)["dt"].shift(1)
    dias_unicos["gap"] = (dias_unicos["dt"] - dias_unicos["prev_dt"]).dt.days
    gaps = dias_unicos.dropna(subset=["gap"]).copy()
    gaps["prev_gap"] = gaps.groupby("CLI_CPF", sort=False)["gap"].shift(1)

    cli_gaps = gaps.groupby("CLI_CPF", sort=False).agg(
        ultimo_gap=("gap", "last"),
        penultimo_gap=("prev_gap", "last"),
        primeiro_gap=("gap", "first"),
        min_gap=("gap", "min"),
        max_gap=("gap", "max"),
        median_gap=("gap", "median"),
    )
    cli_gaps["gap_accel_ratio"] = cli_gaps["ultimo_gap"] / (cli_gaps["median_gap"] + 1.0)
    cli_gaps["gap_trend_abs"] = cli_gaps["ultimo_gap"] - cli_gaps["primeiro_gap"]
    cli_gaps["gap_delta_ultimos"] = cli_gaps["ultimo_gap"] - cli_gaps["penultimo_gap"]
    cli_gaps["ewm_gap_recent"] = np.where(
        cli_gaps["penultimo_gap"].notna(),
        0.6 * cli_gaps["ultimo_gap"] + 0.4 * cli_gaps["penultimo_gap"],
        cli_gaps["ultimo_gap"],
    )

    # 3. Mix de produtos, SKUs, categorias, preco/kg, entropia e terminal de caixa
    sku_dias = v[["CLI_CPF", "SKU", "dt"]].drop_duplicates()
    sku_cnt = sku_dias.groupby(["CLI_CPF", "SKU"], sort=False).size().reset_index(name="n_dias_sku")
    sku_cnt["recomprado"] = (sku_cnt["n_dias_sku"] >= 2).astype(np.int8)
    cli_sku = sku_cnt.groupby("CLI_CPF", sort=False).agg(
        n_skus_vendas=("SKU", "count"),
        n_skus_recomprados=("recomprado", "sum"),
    )
    cli_sku["taxa_recompra_sku"] = cli_sku["n_skus_recomprados"] / np.maximum(cli_sku["n_skus_vendas"], 1)

    cli_prod_extra = v.groupby("CLI_CPF", sort=False).agg(
        n_subcategorias_distintas=("SUBCATEGORIA", "nunique"),
        comprou_bacalhau_pascoa=("is_bacalhau_pascoa", "max"),
        comprou_semana_pascoa=("is_semana_pascoa", "max"),
        pct_linhas_un=("is_un", "mean"),
        n_linhas_invalidas=("is_linha_invalida", "sum"),
        fat_kg_sum=("fat_kg", "sum"),
        qtd_kg_sum=("qtd_kg", "sum"),
    )
    cli_prod_extra["preco_medio_kg"] = np.where(
        cli_prod_extra["qtd_kg_sum"] > 0.01,
        cli_prod_extra["fat_kg_sum"] / cli_prod_extra["qtd_kg_sum"],
        np.nan,
    )
    cli_prod_extra = cli_prod_extra.drop(columns=["fat_kg_sum", "qtd_kg_sum"])

    # Share por categoria (top 12) + Entropia de Shannon das categorias
    cat_fat = v.groupby(["CLI_CPF", "CATEGORIA"], sort=False)["fatur_zero"].sum().unstack(fill_value=0.0)
    total_cat_fat = cat_fat.sum(axis=1).replace(0.0, np.nan)
    p_cat = cat_fat.div(total_cat_fat, axis=0)
    log_p_cat = np.log(p_cat.where(p_cat > 0.0))
    entropia = -(p_cat * log_p_cat).sum(axis=1).fillna(0.0)

    share_cols = pd.DataFrame(index=cat_fat.index)
    share_cols["entropia_categorias"] = entropia
    for cat in TOP_CATEGORIAS_VENDAS:
        col_name = f"share_cat_{_slugify(cat)}"
        share_cols[col_name] = (cat_fat[cat] / total_cat_fat).fillna(0.0) if cat in cat_fat.columns else 0.0

    share_cols["share_bacalhau"] = share_cols.get("share_cat_bacalhau_pescados", 0.0)
    share_cols["share_castanhas_frutas"] = (
        share_cols.get("share_cat_castanhas_oleaginosas", 0.0) + share_cols.get("share_cat_frutas", 0.0)
    )

    # Share dos Top 12 SKUs individuais (forte discriminacao para n_dias == 1)
    v_top12 = v[v["SKU"].isin(TOP_12_SKUS)]
    sku12_fat = v_top12.groupby(["CLI_CPF", "SKU"], sort=False)["fatur_zero"].sum().unstack(fill_value=0.0)
    sku12_share = pd.DataFrame(index=cli_vis.index)
    fat_tot_ser = cli_vis["fat_total_vendas"].replace(0.0, np.nan)
    for sku_id in TOP_12_SKUS:
        if sku_id in sku12_fat.columns:
            sku12_share[f"share_sku_{sku_id}"] = (sku12_fat[sku_id].reindex(cli_vis.index, fill_value=0.0) / fat_tot_ser).fillna(0.0)
        else:
            sku12_share[f"share_sku_{sku_id}"] = 0.0

    # SKU, Subcategoria, Categoria e Caixa principais
    sku_fat = v.groupby(["CLI_CPF", "SKU"], sort=False)["fatur_zero"].sum().reset_index()
    sku_top = sku_fat.sort_values(["CLI_CPF", "fatur_zero"], ascending=[True, False]).drop_duplicates("CLI_CPF")
    sku_top = sku_top.set_index("CLI_CPF")[["SKU"]].rename(columns={"SKU": "sku_principal"})
    sku_top["sku_principal"] = sku_top["sku_principal"].astype(str).astype(object)

    sub_fat = v.groupby(["CLI_CPF", "SUBCATEGORIA"], sort=False)["fatur_zero"].sum().reset_index()
    sub_top = sub_fat.sort_values(["CLI_CPF", "fatur_zero"], ascending=[True, False]).drop_duplicates("CLI_CPF")
    sub_top = sub_top.set_index("CLI_CPF")[["SUBCATEGORIA"]].rename(columns={"SUBCATEGORIA": "subcat_principal"})
    sub_top["subcat_principal"] = sub_top["subcat_principal"].astype(str).astype(object)

    cat_fat_long = v.groupby(["CLI_CPF", "CATEGORIA"], sort=False)["fatur_zero"].sum().reset_index()
    cat_top = cat_fat_long.sort_values(["CLI_CPF", "fatur_zero"], ascending=[True, False]).drop_duplicates("CLI_CPF")
    cat_top = cat_top.set_index("CLI_CPF")[["CATEGORIA"]].rename(columns={"CATEGORIA": "cat_principal"})
    cat_top["cat_principal"] = cat_top["cat_principal"].astype(str).astype(object)

    caixa_cnt = visitas.groupby(["CLI_CPF", "caixa"], sort=False).agg(cnt=("COD_ATEND", "count"), last_dt=("dt", "max")).reset_index()
    caixa_top = caixa_cnt.sort_values(["CLI_CPF", "cnt", "last_dt"], ascending=[True, False, False]).drop_duplicates("CLI_CPF")
    caixa_top = caixa_top.set_index("CLI_CPF")[["caixa"]].rename(columns={"caixa": "caixa_principal"})
    caixa_top["caixa_principal"] = caixa_top["caixa_principal"].astype(str).astype(object)

    features_transacionais = (
        cli_vis.join(cli_gaps, how="left")
        .join(cli_sku, how="left")
        .join(cli_prod_extra, how="left")
        .join(share_cols, how="left")
        .join(sku12_share, how="left")
        .join(sku_top, how="left")
        .join(sub_top, how="left")
        .join(cat_top, how="left")
        .join(caixa_top, how="left")
        .reset_index()
    )
    features_transacionais["encontrado_em_vendas"] = np.int8(1)
    return features_transacionais


def montar_matriz_features(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    vendas_pre: Optional[pd.DataFrame] = None,
    prod_limpo: Optional[pd.DataFrame] = None,
    nivel: str = "completo",
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame, Dict[str, List[str]]]:
    """Combina features de nivel 'original', 'grupo_a' ou 'completo' sem leakage."""
    y_train = train_df["target"].to_numpy(dtype=int)
    colunas_orig = [c for c in train_df.columns if c not in ("CLI_CPF", "target")]

    if nivel == "original":
        X_tr = train_df[colunas_orig].copy()
        X_te = test_df[colunas_orig].copy()
        grupos = {"numericas": [c for c in colunas_orig if c != "filial"], "categoricas": ["filial"], "alta_card": []}
        return X_tr, y_train, X_te, grupos

    tr_a = construir_atributos_base(train_df)
    te_a = construir_atributos_base(test_df)

    if nivel == "grupo_a":
        cols = [c for c in tr_a.columns if c not in ("CLI_CPF", "target")]
        X_tr = tr_a[cols].copy()
        X_te = te_a[cols].copy()
        grupos = {"numericas": [c for c in cols if c != "filial"], "categoricas": ["filial"], "alta_card": []}
        return X_tr, y_train, X_te, grupos

    if vendas_pre is None or prod_limpo is None:
        raise ValueError("vendas_pre e prod_limpo sao obrigatorios para nivel='completo'.")

    feat_tx = construir_atributos_transacionais(vendas_pre, prod_limpo)
    tr_full = tr_a.merge(feat_tx, on="CLI_CPF", how="left")
    te_full = te_a.merge(feat_tx, on="CLI_CPF", how="left")

    for df_m in (tr_full, te_full):
        df_m["encontrado_em_vendas"] = df_m["encontrado_em_vendas"].fillna(0).astype(np.int8)
        df_m["recencia_sobre_ultimo_gap"] = df_m["recencia_dias"] / (df_m["ultimo_gap"] + 1.0)
        df_m["recencia_sobre_max_gap"] = df_m["recencia_dias"] / (df_m["max_gap"] + 1.0)
        df_m["recencia_sobre_median_gap"] = df_m["recencia_dias"] / (df_m["median_gap"] + 1.0)
        df_m["recencia_sobre_ewm_gap"] = df_m["recencia_dias"] / (df_m["ewm_gap_recent"] + 1.0)
        df_m["ratio_fat_ultima_media"] = df_m["fat_ultima_visita"] / (df_m["ticket_medio"] + 1e-5)
        df_m["diff_n_compras_vendas"] = df_m["n_compras"] - df_m["visitas_vendas"]
        df_m["rec_2nd_over_ciclo"] = df_m["rec_2nd_last"] / (df_m["ciclo_medio_dias"] + 1.0)
        df_m["ratio_fat_last_2nd"] = df_m["fat_ultima_visita"] / (np.maximum(df_m["fat_2nd_last"], 0.0) + 1.0)
        for col_cat in ("filial", "cat_principal", "subcat_principal", "sku_principal", "caixa_principal"):
            df_m[col_cat] = df_m[col_cat].where(pd.notna(df_m[col_cat]), "AUSENTE").astype(str).astype(object)

    cols = [c for c in tr_full.columns if c not in ("CLI_CPF", "target")]
    cat_cols = ["filial"]
    alta_card = ["cat_principal", "subcat_principal", "sku_principal", "caixa_principal"]
    num_cols = [c for c in cols if c not in cat_cols + alta_card]

    X_tr = tr_full[cols].copy()
    X_te = te_full[cols].copy()
    grupos = {"numericas": num_cols, "categoricas": cat_cols, "alta_card": alta_card}
    return X_tr, y_train, X_te, grupos


def criar_preprocessador(
    colunas_numericas: List[str],
    colunas_categoricas: List[str],
    colunas_alta_card: Optional[List[str]] = None,
    modo: str = "tree_native",
    winsorize_upper: Optional[float] = 0.995,
    usar_bgnbd: bool = True,
    target_type: str = "auto",
) -> ColumnTransformer:
    """
    Cria ColumnTransformer modular adequado a familia do algoritmo e modo de imputacao.
    Usa TargetEncoder(cv=5) interno para evitar overfitting in-fold e BGNBDFeatureTransformer
    nao supervisionado (zero target).
    """
    colunas_alta_card = colunas_alta_card or []
    bgnbd_cols_cand = ["n_dias_com_compra", "antiguidade_dias", "recencia_dias", "ciclo_medio_dias", "variacao_ciclo", "visitas_90d"]
    bgnbd_cols = [c for c in bgnbd_cols_cand if c in colunas_numericas]
    habilitar_bgnbd = usar_bgnbd and len(bgnbd_cols) >= 5

    te_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    if modo == "gam_spline":
        spline_cols = [c for c in SPLINE_KEY_COLS if c in colunas_numericas]
        other_nums = [c for c in colunas_numericas if c not in spline_cols]

        spline_steps = []
        if winsorize_upper is not None:
            spline_steps.append(("winsorizar", Winsorizer(lower_q=0.0, upper_q=winsorize_upper)))
        spline_steps.extend([
            ("imputar", SimpleImputer(strategy="median")),
            ("splines", SplineTransformer(n_knots=5, degree=3, include_bias=False)),
            ("escalar", StandardScaler()),
        ])

        other_steps = []
        if winsorize_upper is not None:
            other_steps.append(("winsorizar", Winsorizer(lower_q=0.0, upper_q=winsorize_upper)))
        other_steps.extend([
            ("imputar", SimpleImputer(strategy="median")),
            ("escalar", StandardScaler()),
        ])

        transformers = [
            ("splines_chave", Pipeline(spline_steps), spline_cols),
            ("numericas", Pipeline(other_steps), other_nums),
            ("categoricas", Pipeline([
                ("imputar", SimpleImputer(strategy="constant", fill_value="AUSENTE")),
                ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]), colunas_categoricas),
        ]
        if habilitar_bgnbd:
            transformers.append((
                "bgnbd",
                Pipeline([("tx", BGNBDFeatureTransformer()), ("escalar", StandardScaler())]),
                bgnbd_cols,
            ))
        if colunas_alta_card:
            transformers.append((
                "alta_card_oof",
                TargetEncoder(cv=te_cv, smooth="auto", target_type=target_type),
                colunas_alta_card,
            ))
        return ColumnTransformer(transformers, remainder="drop")

    if modo == "linear":
        num_steps = []
        if winsorize_upper is not None:
            num_steps.append(("winsorizar", Winsorizer(lower_q=0.0, upper_q=winsorize_upper)))
        num_steps.extend([
            ("imputar", SimpleImputer(strategy="median", add_indicator=False)),
            ("escalar", StandardScaler()),
        ])
        cat_pipe = Pipeline([
            ("imputar", SimpleImputer(strategy="constant", fill_value="AUSENTE")),
            ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers = [
            ("numericas", Pipeline(num_steps), colunas_numericas),
            ("categoricas", cat_pipe, colunas_categoricas),
        ]
        if habilitar_bgnbd:
            transformers.append((
                "bgnbd",
                Pipeline([("tx", BGNBDFeatureTransformer()), ("escalar", StandardScaler())]),
                bgnbd_cols,
            ))
        if colunas_alta_card:
            transformers.append((
                "alta_card_oof",
                TargetEncoder(cv=te_cv, smooth="auto", target_type=target_type),
                colunas_alta_card,
            ))
        return ColumnTransformer(transformers, remainder="drop")

    if modo == "tree_imputed":
        num_pipe = Pipeline([
            ("imputar", SimpleImputer(strategy="median", add_indicator=False)),
        ])
        cat_pipe = Pipeline([
            ("imputar", SimpleImputer(strategy="constant", fill_value="AUSENTE")),
            ("codificar", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers = [
            ("numericas", num_pipe, colunas_numericas),
            ("categoricas", cat_pipe, colunas_categoricas),
        ]
        if habilitar_bgnbd:
            transformers.append(("bgnbd", BGNBDFeatureTransformer(), bgnbd_cols))
        if colunas_alta_card:
            transformers.append((
                "alta_card_oof",
                TargetEncoder(cv=te_cv, smooth="auto", target_type=target_type),
                colunas_alta_card,
            ))
        return ColumnTransformer(transformers, remainder="drop")

    if modo == "tree_native":
        cat_pipe = Pipeline([
            ("imputar", SimpleImputer(strategy="constant", fill_value="AUSENTE")),
            ("codificar", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ])
        transformers = [
            ("numericas", "passthrough", colunas_numericas),
            ("categoricas", cat_pipe, colunas_categoricas),
        ]
        if habilitar_bgnbd:
            transformers.append(("bgnbd", BGNBDFeatureTransformer(), bgnbd_cols))
        if colunas_alta_card:
            transformers.append((
                "alta_card_oof",
                TargetEncoder(cv=te_cv, smooth="auto", target_type=target_type),
                colunas_alta_card,
            ))
        return ColumnTransformer(transformers, remainder="drop")

    raise ValueError(f"Modo desconhecido: {modo}")


def filtrar_outliers_iqr(
    X: pd.DataFrame,
    y: np.ndarray | pd.Series,
    colunas: List[str],
    fator: float = 1.5,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """Reproduz filtro IQR multivariado (apenas para ablacoes comparativas com o template anterior)."""
    q1 = X[colunas].quantile(0.25)
    q3 = X[colunas].quantile(0.75)
    iqr = q3 - q1
    lim_inf = q1 - fator * iqr
    lim_sup = q3 + fator * iqr
    dentro = X[colunas].isna() | (X[colunas].ge(lim_inf) & X[colunas].le(lim_sup))
    manter = dentro.all(axis=1).to_numpy()
    y_arr = np.asarray(y, dtype=int)
    return X.loc[manter].reset_index(drop=True), y_arr[manter]
