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

# Arquivo com todas as funcoes e codigos referentes a analise exploratoria

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


NOMES_CLASSES = {
    0: "0 - Ativo",
    1: "1 - Desacelerando",
    2: "2 - Dormente",
    3: "3 - Churn Efetivo",
}


def configurar_estilo_graficos() -> None:
    sns.set_theme(style="whitegrid", context="notebook")
    plt.rcParams["figure.dpi"] = 110
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.labelsize"] = 10


def tabela_resumo_atributos(train_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    """Retorna tabela com tipos, nulos, media, mediana e amplitude de treino e teste."""
    registros = []
    for col in train_df.columns:
        if col in ("CLI_CPF", "target"):
            continue
        s_tr = train_df[col]
        s_te = test_df[col]
        is_num = pd.api.types.is_numeric_dtype(s_tr)
        registros.append({
            "Atributo": col,
            "Tipo": "Numerico" if is_num else "Categorico",
            "Nulos Treino (%)": round(float(s_tr.isna().mean() * 100), 2),
            "Nulos Teste (%)": round(float(s_te.isna().mean() * 100), 2),
            "Media Treino": round(float(s_tr.mean()), 3) if is_num else np.nan,
            "Mediana Treino": round(float(s_tr.median()), 3) if is_num else np.nan,
            "Min": round(float(s_tr.min()), 3) if is_num else np.nan,
            "Max": round(float(s_tr.max()), 3) if is_num else np.nan,
            "Unicos Treino": int(s_tr.nunique(dropna=True)),
        })
    return pd.DataFrame(registros)


def analisar_distribuicao_classes(train_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Analisa desbalanceamento global e segmentado por regime (1 dia vs >=2 dias) e filial."""
    geral = train_df["target"].value_counts().sort_index().to_frame("Contagem")
    geral["Proporcao (%)"] = (geral["Contagem"] / len(train_df) * 100).round(2)
    geral.index = [NOMES_CLASSES[i] for i in geral.index]

    regime = pd.crosstab(
        np.where(train_df["n_dias_com_compra"] == 1, "1 dia de compra (53.1%)", "2+ dias de compra (46.9%)"),
        train_df["target"],
        normalize="index",
    ).mul(100).round(2)
    regime.columns = [NOMES_CLASSES[c] for c in regime.columns]

    filial = pd.crosstab(train_df["filial"], train_df["target"], normalize="index").mul(100).round(2)
    filial.columns = [NOMES_CLASSES[c] for c in filial.columns]

    return {"geral": geral, "por_regime": regime, "por_filial": filial}


def analisar_estrutura_ciclo_e_iqr(train_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """Mostra os limites estruturais do ciclo medio por classe e o vies da remocao IQR."""
    has_c = train_df[train_df["ciclo_medio_dias"].notna()].copy()
    has_c["rec_sobre_ciclo"] = has_c["recencia_dias"] / has_c["ciclo_medio_dias"]

    resumo_ciclo = (
        has_c.groupby("target")
        .agg(
            clientes=("CLI_CPF", "count"),
            ciclo_mediano=("ciclo_medio_dias", "median"),
            ciclo_max=("ciclo_medio_dias", "max"),
            rec_ciclo_mediano=("rec_sobre_ciclo", "median"),
            ticket_mediano=("ticket_medio", "median"),
        )
        .round(2)
    )
    resumo_ciclo.index = [NOMES_CLASSES[i] for i in resumo_ciclo.index]

    num_cols = [c for c in train_df.select_dtypes("number").columns if c != "target"]
    q1 = train_df[num_cols].quantile(0.25)
    q3 = train_df[num_cols].quantile(0.75)
    iqr = q3 - q1
    keep = (train_df[num_cols].isna() | (train_df[num_cols].ge(q1 - 1.5 * iqr) & train_df[num_cols].le(q3 + 1.5 * iqr))).all(axis=1)

    impacto_iqr = pd.DataFrame({
        "Total": train_df["target"].value_counts().sort_index(),
        "Mantidos": train_df.loc[keep, "target"].value_counts().sort_index(),
        "Removidos": train_df.loc[~keep, "target"].value_counts().sort_index(),
    })
    impacto_iqr["Removidos (%)"] = (impacto_iqr["Removidos"] / impacto_iqr["Total"] * 100).round(2)
    impacto_iqr.index = [NOMES_CLASSES[i] for i in impacto_iqr.index]

    return {"ciclo_por_classe": resumo_ciclo, "impacto_iqr": impacto_iqr}


def analisar_vendas_e_categorias_pre_cutoff(
    vendas_pre: pd.DataFrame,
    prod_limpo: pd.DataFrame,
    train_df: pd.DataFrame,
) -> Dict[str, pd.DataFrame]:
    """Analisa dinamica temporal mensal (jan-set/2024) e comportamento por categoria e filial."""
    v = vendas_pre[["CLI_CPF", "SKU", "dt", "FILIAL_LIMPA", "COD_ATEND", "fatur_limpo"]].merge(
        prod_limpo[["SKU", "CATEGORIA"]], on="SKU", how="left"
    )
    v["CATEGORIA"] = v["CATEGORIA"].fillna("DESCONHECIDA")
    v["mes"] = v["dt"].dt.to_period("M").astype(str)

    mensal = (
        v.groupby(["mes", "FILIAL_LIMPA"], sort=False)
        .agg(
            visitas=("COD_ATEND", "nunique"),
            clientes_unicos=("CLI_CPF", "nunique"),
            faturamento=("fatur_limpo", "sum"),
        )
        .reset_index()
        .sort_values(["mes", "FILIAL_LIMPA"])
    )

    # Categoria principal de cada cliente no treino (ordenacao rapida por inteiro unico)
    cat_cnt = v.groupby(["CLI_CPF", "CATEGORIA"], sort=False).size().reset_index(name="n")
    cat_top = (
        cat_cnt.sort_values("n", ascending=False, kind="mergesort")
        .drop_duplicates("CLI_CPF")[["CLI_CPF", "CATEGORIA"]]
    )
    m = train_df.merge(cat_top, on="CLI_CPF", how="inner")
    top_cats = m["CATEGORIA"].value_counts().head(10).index
    por_cat = pd.crosstab(
        m.loc[m["CATEGORIA"].isin(top_cats), "CATEGORIA"],
        m.loc[m["CATEGORIA"].isin(top_cats), "target"],
        normalize="index",
    ).mul(100).round(2)
    por_cat.columns = [NOMES_CLASSES[c] for c in por_cat.columns]
    por_cat["Clientes"] = m["CATEGORIA"].value_counts().loc[por_cat.index]
    por_cat = por_cat.sort_values("Clientes", ascending=False)

    return {"mensal_filial": mensal, "por_categoria": por_cat}


def gerar_graficos_eda(
    train_df: pd.DataFrame,
    vendas_pre: pd.DataFrame,
    prod_limpo: pd.DataFrame,
    figs_dir: str | Path = "figs",
    analise_tx: Optional[Dict[str, pd.DataFrame]] = None,
) -> list[Path]:
    """Gera e salva os principais graficos de EDA em figs/ (sem acentos nos nomes)."""
    configurar_estilo_graficos()
    out_dir = Path(figs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    salvos: list[Path] = []

    # 1. Distribuicao de classes e segmentacao por regime (1 dia vs >=2 dias)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    vc = train_df["target"].value_counts().sort_index()
    sns.barplot(x=[NOMES_CLASSES[i] for i in vc.index], y=vc.values, ax=axes[0], hue=[NOMES_CLASSES[i] for i in vc.index], palette="viridis", legend=False)
    axes[0].set_title("Distribuicao das Classes em train.csv (n=59.945)")
    axes[0].set_ylabel("Clientes")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].set_ylim(0, vc.max() * 1.18)
    for idx, val in enumerate(vc.values):
        axes[0].text(idx, val + 500, f"{val:,}\n({val/len(train_df)*100:.1f}%)", ha="center", va="bottom", fontsize=8)

    ct_reg = pd.crosstab(
        np.where(train_df["n_dias_com_compra"] == 1, "1 dia de compra", "2+ dias de compra"),
        train_df["target"],
        normalize="index",
    ).mul(100)
    ct_reg.columns = [NOMES_CLASSES[c] for c in ct_reg.columns]
    ct_reg.plot(kind="bar", stacked=True, ax=axes[1], colormap="viridis")
    axes[1].set_title("Proporcao das Classes por Regime de Compras")
    axes[1].set_xlabel("Regime de Compras")
    axes[1].set_ylabel("Proporcao (%)")
    axes[1].tick_params(axis="x", rotation=0)
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    p1 = out_dir / "eda_classes_regime.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    salvos.append(p1)

    # 2. Serie temporal mensal pre-cutoff por filial e ciclo_medio_dias por classe
    if analise_tx is None:
        analise_tx = analisar_vendas_e_categorias_pre_cutoff(vendas_pre, prod_limpo, train_df)
    mensal = analise_tx["mensal_filial"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    sns.lineplot(data=mensal, x="mes", y="faturamento", hue="FILIAL_LIMPA", marker="o", ax=axes[0])
    axes[0].set_title("Faturamento Mensal Pre-Cutoff por Filial (Jan-Set/2024)")
    axes[0].set_ylabel("Faturamento (R$)")
    axes[0].set_xlabel("Mes")
    axes[0].tick_params(axis="x", rotation=30)

    has_c = train_df[train_df["ciclo_medio_dias"].notna()].copy()
    has_c["Classe"] = has_c["target"].map(NOMES_CLASSES)
    ordem_classes = [NOMES_CLASSES[i] for i in range(4)]
    sns.boxplot(data=has_c, x="Classe", y="ciclo_medio_dias", order=ordem_classes, ax=axes[1], showfliers=False, hue="Classe", palette="Set2", legend=False)
    axes[1].axhline(46.0, color="red", linestyle="--", linewidth=1, label="Limite max Classe 2 (~45.3d)")
    axes[1].axhline(76.0, color="orange", linestyle=":", linewidth=1.2, label="Limite max Classe 1 (76.0d)")
    axes[1].set_title("Distribuicao do Ciclo Medio por Classe (2+ dias)")
    axes[1].set_ylabel("ciclo_medio_dias")
    axes[1].tick_params(axis="x", rotation=15)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    p2 = out_dir / "eda_temporal_ciclo.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    salvos.append(p2)

    # 3. Razao recencia / ciclo e impacto por categoria principal
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    has_c["rec_sobre_ciclo"] = np.clip(has_c["recencia_dias"] / has_c["ciclo_medio_dias"], 0, 5.0)
    sns.boxplot(data=has_c, x="Classe", y="rec_sobre_ciclo", order=ordem_classes, ax=axes[0], showfliers=False, hue="Classe", palette="Set2", legend=False)
    axes[0].axhline(1.2, color="blue", linestyle="--", linewidth=1, label="1.2x ciclo")
    axes[0].axhline(2.0, color="red", linestyle="--", linewidth=1, label="2.0x ciclo")
    axes[0].set_title("Razao Recencia / Ciclo Medio por Classe (clip 5.0)")
    axes[0].set_ylabel("recencia_dias / ciclo_medio_dias")
    axes[0].tick_params(axis="x", rotation=15)
    axes[0].legend(fontsize=8)

    por_cat = analise_tx["por_categoria"][[NOMES_CLASSES[0], NOMES_CLASSES[1], NOMES_CLASSES[2], NOMES_CLASSES[3]]]
    por_cat.plot(kind="barh", stacked=True, ax=axes[1], colormap="viridis")
    axes[1].set_title("Distribuicao de Target nas Top 10 Categorias Principais")
    axes[1].set_xlabel("Proporcao (%)")
    axes[1].legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    p3 = out_dir / "eda_razoes_categorias.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    salvos.append(p3)

    return salvos

