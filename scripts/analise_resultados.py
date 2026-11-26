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

# Arquivo com todas as funcoes e codigos referentes a analise dos resultados

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

from scripts.analise_exploratoria import NOMES_CLASSES, configurar_estilo_graficos
from scripts.preprocessamento import criar_preprocessador


CORES_CLASSES = ["#1b9e77", "#d95f02", "#7570b3", "#e7298a"]
ROTULOS_CURTOS = {0: "C0 Ativo", 1: "C1 Desacel.", 2: "C2 Dormente", 3: "C3 Churn"}


def formatar_tabela_media_desvio(df_resumo: pd.DataFrame) -> pd.DataFrame:
    """Formata tabela comparativa com F1-macro e AUC-OvR em formato media +- desvio padrao."""
    out = pd.DataFrame()
    out["Modelo"] = df_resumo["Modelo"]
    out["F1-macro (Argmax)"] = df_resumo["F1_macro_Argmax"].map(lambda x: f"{x:.4f}")
    out["F1-macro (PosProc)"] = [
        f"{m:.4f} ± {s:.4f}" for m, s in zip(df_resumo["F1_macro_PosProc"], df_resumo["F1_std_folds"])
    ]
    out["AUC-OvR macro"] = [
        f"{m:.4f} ± {s:.4f}" for m, s in zip(df_resumo["AUC_OvR_macro"], df_resumo["AUC_std_folds"])
    ]
    if "Log_Loss" in df_resumo.columns:
        out["Log-Loss"] = df_resumo["Log_Loss"].map(lambda x: f"{x:.4f}")
    out["Bal. Acc."] = df_resumo["Balanced_Acc"].map(lambda x: f"{x:.4f}")
    for c in range(4):
        out[f"F1 C{c}"] = df_resumo[f"F1_C{c}"].map(lambda x: f"{x:.4f}")
    return out


def executar_testes_estatisticos_folds(
    folds_f1_df: pd.DataFrame,
    modelo_referencia: Optional[str] = None,
) -> Dict[str, Any]:
    """Aplica teste global de Friedman e Wilcoxon pareado contra o melhor modelo nos 5 folds."""
    medias = folds_f1_df.mean(axis=0).sort_values(ascending=False)
    ref = modelo_referencia or str(medias.index[0])

    if folds_f1_df.shape[1] >= 3:
        stat_f, p_f = friedmanchisquare(*[folds_f1_df[col].to_numpy() for col in folds_f1_df.columns])
    else:
        stat_f, p_f = np.nan, np.nan

    linhas = []
    ref_vals = folds_f1_df[ref].to_numpy()
    for col in medias.index:
        vals = folds_f1_df[col].to_numpy()
        diff = float(np.mean(ref_vals - vals))
        if col == ref:
            p_w = np.nan
            sig = "Referencia"
        else:
            if np.allclose(ref_vals, vals):
                p_w = 1.0
            else:
                _, p_w = wilcoxon(ref_vals, vals, alternative="greater")
            sig = "p < 0.05" if (pd.notna(p_w) and p_w < 0.05) else "n.s. (n=5 folds)"
        linhas.append({
            "Modelo": col,
            "Media_F1_Folds": round(float(vals.mean()), 4),
            "Desvio_F1_Folds": round(float(vals.std(ddof=1)), 4),
            "Diferenca_vs_Ref": round(diff, 4),
            "Wilcoxon_p_unilateral": round(float(p_w), 4) if pd.notna(p_w) else np.nan,
            "Significancia": sig,
        })

    return {
        "referencia": ref,
        "friedman_stat": float(stat_f) if pd.notna(stat_f) else np.nan,
        "friedman_p": float(p_f) if pd.notna(p_f) else np.nan,
        "comparacao_pareada": pd.DataFrame(linhas),
    }


def gerar_matrizes_confusao_comparativas(
    y_true: np.ndarray,
    pred_antes: np.ndarray,
    pred_depois: np.ndarray,
    titulo_antes: str = "Argmax Padrao (Sem Pos-Processamento)",
    titulo_depois: str = "Ensemble Log-Odds + Calib. + Plug-in F1 (OOF)",
    figs_dir: str | Path = "figs",
) -> Path:
    """Plota matrizes de confusao normalizadas (%) lado a lado para evidenciar ganho nas minoritarias."""
    configurar_estilo_graficos()
    out_dir = Path(figs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = ["C0 Ativo", "C1 Desacel.", "C2 Dormente", "C3 Churn"]

    fig, axes = plt.subplots(1, 2, figsize=(12.6, 4.8))
    for ax, pred, titulo in [(axes[0], pred_antes, titulo_antes), (axes[1], pred_depois, titulo_depois)]:
        cm = confusion_matrix(y_true, pred, labels=[0, 1, 2, 3])
        cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100.0
        annot = np.empty_like(cm, dtype=object)
        for i in range(4):
            for j in range(4):
                annot[i, j] = f"{cm_norm[i, j]:.1f}%\n({cm[i, j]:,})"
        sns.heatmap(
            cm_norm,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=False,
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_title(titulo)
        ax.set_xlabel("Classe Predita")
        ax.set_ylabel("Classe Real")

    fig.tight_layout()
    caminho = out_dir / "res_matrizes_confusao.png"
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return caminho


def gerar_graficos_comparativos_modelos(
    df_modelos: pd.DataFrame,
    dist_classes: pd.DataFrame,
    figs_dir: str | Path = "figs",
    df_extras: Optional[pd.DataFrame] = None,
) -> list[Path]:
    """Gera graficos comparativos de F1-macro (antes/depois), F1 por classe e distribuicao de submissao."""
    configurar_estilo_graficos()
    out_dir = Path(figs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    salvos: list[Path] = []

    if df_extras is not None and not df_extras.empty:
        cols_comuns = [c for c in df_modelos.columns if c in df_extras.columns]
        df_comb = pd.concat([df_modelos[cols_comuns], df_extras[cols_comuns]], ignore_index=True)
        df_comb = df_comb.drop_duplicates(subset=["Modelo"], keep="last").reset_index(drop=True)
    else:
        df_comb = df_modelos.copy()

    # 1. F1-macro Argmax vs PosProc por modelo + F1 por classe nos Top 5
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6))
    df_plot = df_comb.sort_values("F1_macro_PosProc", ascending=True).reset_index(drop=True)
    y_pos = np.arange(len(df_plot))
    h = 0.38

    cores_pos = [
        "#1e8449" if "Ensemble" in str(m) else "#2980b9"
        for m in df_plot["Modelo"]
    ]
    axes[0].barh(y_pos - h / 2, df_plot["F1_macro_Argmax"], height=h, color="#95a5a6")
    axes[0].barh(y_pos + h / 2, df_plot["F1_macro_PosProc"], height=h, color=cores_pos)
    axes[0].axvline(0.4070, color="#c0392b", linestyle="--", linewidth=1.3)
    axes[0].set_yticks(y_pos)
    axes[0].set_yticklabels(df_plot["Modelo"], fontsize=8.5)
    axes[0].set_xlabel("F1-macro OOF (5-fold)")
    axes[0].set_title("Comparacao de Modelos: Argmax vs Pos-Processamento")
    axes[0].set_xlim(0, float(df_plot["F1_macro_PosProc"].max()) * 1.17)

    handles_left = [
        Patch(facecolor="#95a5a6", label="Argmax padrao"),
        Patch(facecolor="#2980b9", label="Calib. + Mascara + Plug-in Expected F1 (OOF)"),
        Patch(facecolor="#1e8449", label="Ensemble Log-Odds (16 modelos)"),
        plt.Line2D([0], [0], color="#c0392b", linestyle="--", linewidth=1.3, label="Baseline AM-master (0.4070)"),
    ]
    axes[0].legend(handles=handles_left, fontsize=7.8, loc="upper center", bbox_to_anchor=(0.48, -0.14), ncol=2, frameon=False)
    for idx, val in enumerate(df_plot["F1_macro_PosProc"]):
        axes[0].text(float(val) + 0.004, idx + h / 2, f"{val:.4f}", va="center", fontsize=8)

    top5 = df_comb.sort_values("F1_macro_PosProc", ascending=False).head(5).copy()
    top5["Modelo_Curto"] = (
        top5["Modelo"]
        .str.replace(" (3-Estagios, 3-Seeds)", " (3-Est.)", regex=False)
        .str.replace(" (Early Stopping)", " (ES)", regex=False)
        .str.replace(" (16 modelos)", "", regex=False)
    )
    top5_plot = top5.set_index("Modelo_Curto")[["F1_C0", "F1_C1", "F1_C2", "F1_C3"]]
    top5_plot.columns = ["C0 Ativo", "C1 Desacel.", "C2 Dormente", "C3 Churn"]
    top5_plot.plot(kind="bar", ax=axes[1], color=CORES_CLASSES, width=0.72)
    axes[1].set_ylim(0, 0.90)
    axes[1].set_ylabel("F1-score OOF")
    axes[1].set_xlabel("")
    axes[1].set_title("F1 por Classe nos Top 5 Modelos (PosProc Nested OOF)", pad=28)
    axes[1].tick_params(axis="x", rotation=18, labelsize=8.5)
    for lbl in axes[1].get_xticklabels():
        lbl.set_ha("right")
    axes[1].legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.11),
        ncol=4,
        fontsize=8,
        frameon=False,
    )

    fig.tight_layout()
    p1 = out_dir / "res_comparacao_modelos.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    salvos.append(p1)

    # 2. Distribuicao Real vs OOF vs Test
    fig, ax = plt.subplots(figsize=(9.2, 4.4))
    dist_plot = dist_classes[["Train_Real_%", "OOF_Ensemble_%", "Test_Ensemble_%"]].copy()
    dist_plot.index = [NOMES_CLASSES[i] for i in dist_plot.index]
    dist_plot.columns = ["Treino Real (%)", "OOF Ensemble (%)", "Teste Ensemble (%)"]
    dist_plot.plot(kind="bar", ax=ax, color=["#2c3e50", "#2980b9", "#27ae60"], width=0.72)
    ax.set_title("Simulacao de Targets: Treino Real vs OOF vs Submissao Teste (%)")
    ax.set_ylabel("Proporcao (%)")
    ax.set_ylim(0, 75)
    ax.tick_params(axis="x", rotation=10)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f%%", fontsize=7.5, padding=2)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    p2 = out_dir / "res_distribuicao_targets.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    salvos.append(p2)

    return salvos


def analisar_importancia_atributos(
    X: pd.DataFrame,
    y: np.ndarray,
    grupos: Dict[str, List[str]],
    modelo_clf,
    top_n: int = 20,
    figs_dir: str | Path = "figs",
) -> pd.DataFrame:
    """Ajusta o estimador no treino completo e extrai top-N atributos mais importantes."""
    configurar_estilo_graficos()
    out_dir = Path(figs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prep = criar_preprocessador(grupos["numericas"], grupos["categoricas"], grupos["alta_card"], modo="tree_native")
    pipe = Pipeline([("preprocessor", prep), ("classifier", modelo_clf)])
    pipe.fit(X, y)

    nomes_brutos = pipe.named_steps["preprocessor"].get_feature_names_out()
    nomes_feat = [str(n).split("__", 1)[-1] for n in nomes_brutos]

    imp = pipe.named_steps["classifier"].feature_importances_
    df_imp = (
        pd.DataFrame({"Atributo": nomes_feat, "Importancia": imp})
        .sort_values("Importancia", ascending=False)
        .reset_index(drop=True)
    )
    df_imp["Importancia_%"] = (df_imp["Importancia"] / df_imp["Importancia"].sum() * 100).round(2)

    top_df = df_imp.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8.8, 6.2))
    ax.barh(top_df["Atributo"], top_df["Importancia_%"], color="#2c7fb8")
    ax.set_xlabel("Importancia Relativa (% de splits/ganho)")
    ax.set_title(f"Top {top_n} Atributos Mais Importantes")
    fig.tight_layout()
    fig.savefig(out_dir / "res_importancia_atributos.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return df_imp.head(top_n)


def gerar_relatorio_por_classe(
    y_true: np.ndarray,
    proba_oof: np.ndarray,
    pred_argmax: np.ndarray,
    pred_posproc: np.ndarray,
) -> pd.DataFrame:
    """Relatorio detalhado por classe comparando Argmax vs PosProc + AUC-OvR, PR-AUC (AP) e Brier OvR."""
    y_true = np.asarray(y_true, dtype=int)
    p_arg, r_arg, f_arg, sup_real = precision_recall_fscore_support(y_true, pred_argmax, labels=[0, 1, 2, 3], zero_division=0)
    p_pos, r_pos, f_pos, _ = precision_recall_fscore_support(y_true, pred_posproc, labels=[0, 1, 2, 3], zero_division=0)
    sup_arg = np.bincount(np.asarray(pred_argmax, dtype=int), minlength=4)
    sup_pos = np.bincount(np.asarray(pred_posproc, dtype=int), minlength=4)

    linhas = []
    for c in range(4):
        y_bin = (y_true == c).astype(int)
        p_c = proba_oof[:, c]
        linhas.append({
            "Classe": NOMES_CLASSES[c],
            "Suporte_Real": int(sup_real[c]),
            "Suporte_Argmax": int(sup_arg[c]),
            "Suporte_PosProc": int(sup_pos[c]),
            "Precisao_Argmax": float(p_arg[c]),
            "Recall_Argmax": float(r_arg[c]),
            "F1_Argmax": float(f_arg[c]),
            "Precisao_PosProc": float(p_pos[c]),
            "Recall_PosProc": float(r_pos[c]),
            "F1_PosProc": float(f_pos[c]),
            "AUC_OvR": float(roc_auc_score(y_bin, p_c)),
            "PR_AUC_AP": float(average_precision_score(y_bin, p_c)),
            "Brier_OvR": float(brier_score_loss(y_bin, p_c)),
        })

    df_out = pd.DataFrame(linhas)
    macro = {
        "Classe": "Macro / Global",
        "Suporte_Real": int(len(y_true)),
        "Suporte_Argmax": int(len(y_true)),
        "Suporte_PosProc": int(len(y_true)),
        "Precisao_Argmax": float(df_out["Precisao_Argmax"].mean()),
        "Recall_Argmax": float(df_out["Recall_Argmax"].mean()),
        "F1_Argmax": float(df_out["F1_Argmax"].mean()),
        "Precisao_PosProc": float(df_out["Precisao_PosProc"].mean()),
        "Recall_PosProc": float(df_out["Recall_PosProc"].mean()),
        "F1_PosProc": float(df_out["F1_PosProc"].mean()),
        "AUC_OvR": float(df_out["AUC_OvR"].mean()),
        "PR_AUC_AP": float(df_out["PR_AUC_AP"].mean()),
        "Brier_OvR": float(df_out["Brier_OvR"].sum()),
    }
    return pd.concat([df_out, pd.DataFrame([macro])], ignore_index=True)


def gerar_curvas_roc_pr(
    y_true: np.ndarray,
    proba_oof: np.ndarray,
    figs_dir: str | Path = "figs",
) -> Path:
    """Plota curvas ROC-OvR e Precision-Recall OvR por classe + Macro-ROC."""
    configurar_estilo_graficos()
    out_dir = Path(figs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype=int)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    fpr_grid = np.linspace(0.0, 1.0, 250)
    mean_tpr = np.zeros_like(fpr_grid)
    aucs = []

    for c in range(4):
        y_bin = (y_true == c).astype(int)
        p_c = proba_oof[:, c]
        fpr, tpr, _ = roc_curve(y_bin, p_c)
        auc_c = roc_auc_score(y_bin, p_c)
        aucs.append(auc_c)
        mean_tpr += np.interp(fpr_grid, fpr, tpr) / 4.0
        axes[0].plot(fpr, tpr, color=CORES_CLASSES[c], lw=1.9, label=f"{ROTULOS_CURTOS[c]} (AUC = {auc_c:.4f})")

        prec, rec, _ = precision_recall_curve(y_bin, p_c)
        ap_c = average_precision_score(y_bin, p_c)
        prev_c = float(y_bin.mean())
        axes[1].plot(rec, prec, color=CORES_CLASSES[c], lw=1.9, label=f"{ROTULOS_CURTOS[c]} (AP = {ap_c:.4f}, prev = {prev_c:.1%})")
        axes[1].axhline(prev_c, color=CORES_CLASSES[c], linestyle=":", lw=1.0, alpha=0.5)

    axes[0].plot(fpr_grid, mean_tpr, color="#111111", linestyle="-.", lw=2.1, label=f"Macro-Average (AUC = {np.mean(aucs):.4f})")
    axes[0].plot([0, 1], [0, 1], color="#7f8c8d", linestyle="--", lw=1.1, label="Aleatorio (AUC = 0.5000)")
    axes[0].set_title("Curvas ROC One-vs-Rest (Ensemble OOF)")
    axes[0].set_xlabel("Taxa de Falsos Positivos (FPR)")
    axes[0].set_ylabel("Taxa de Verdadeiros Positivos (TPR / Recall)")
    axes[0].legend(loc="lower right", fontsize=8.5, frameon=True)

    axes[1].set_title("Curvas Precision-Recall One-vs-Rest (Ensemble OOF)")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precisao")
    axes[1].set_xlim(0, 1.0)
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(loc="upper right", fontsize=8.3, frameon=True)

    fig.tight_layout()
    caminho = out_dir / "res_curvas_roc_pr.png"
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return caminho


def _ece_binario(y_bin: np.ndarray, p_cls: np.ndarray, n_bins: int = 10) -> float:
    """Calcula Expected Calibration Error (ECE) em decis quantilicos de probabilidade."""
    if len(y_bin) == 0:
        return 0.0
    quantis = np.linspace(0.0, 1.0, n_bins + 1)
    edges = np.quantile(p_cls, quantis)
    edges[0], edges[-1] = -1e-9, 1.0 + 1e-9
    bins = np.searchsorted(edges[1:-1], p_cls, side="right")
    ece = 0.0
    n = len(y_bin)
    for b in range(n_bins):
        m = (bins == b)
        if np.any(m):
            ece += (np.sum(m) / n) * abs(float(y_bin[m].mean()) - float(p_cls[m].mean()))
    return float(ece)


def analisar_calibracao_probabilidades(
    y_true: np.ndarray,
    proba_bruta: np.ndarray,
    proba_calib: np.ndarray,
    figs_dir: str | Path = "figs",
) -> Dict[str, Any]:
    """Avalia ECE e Brier Score antes e depois da calibracao Logit + Diagrama de Confiabilidade e Risco."""
    configurar_estilo_graficos()
    out_dir = Path(figs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype=int)

    linhas = []
    for c in range(4):
        yb = (y_true == c).astype(int)
        linhas.append({
            "Classe": NOMES_CLASSES[c],
            "ECE_Antes": _ece_binario(yb, proba_bruta[:, c]),
            "ECE_Depois": _ece_binario(yb, proba_calib[:, c]),
            "Brier_Antes": float(brier_score_loss(yb, proba_bruta[:, c])),
            "Brier_Depois": float(brier_score_loss(yb, proba_calib[:, c])),
        })
    df_cal = pd.DataFrame(linhas)
    ll_bruto = float(log_loss(y_true, np.clip(proba_bruta, 1e-12, 1.0), labels=[0, 1, 2, 3]))
    ll_calib = float(log_loss(y_true, np.clip(proba_calib, 1e-12, 1.0), labels=[0, 1, 2, 3]))
    df_cal = pd.concat([
        df_cal,
        pd.DataFrame([{
            "Classe": "Global (Media ECE / Brier Multi)",
            "ECE_Antes": float(df_cal["ECE_Antes"].mean()),
            "ECE_Depois": float(df_cal["ECE_Depois"].mean()),
            "Brier_Antes": float(df_cal["Brier_Antes"].sum()),
            "Brier_Depois": float(df_cal["Brier_Depois"].sum()),
        }]),
    ], ignore_index=True)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))
    for c in range(4):
        yb = (y_true == c).astype(int)
        prob_true, prob_pred = calibration_curve(yb, proba_calib[:, c], n_bins=10, strategy="quantile")
        ece_c = df_cal.loc[c, "ECE_Depois"]
        axes[0].plot(prob_pred, prob_true, marker="o", markersize=4.5, lw=1.7, color=CORES_CLASSES[c], label=f"{ROTULOS_CURTOS[c]} (ECE={ece_c:.4f})")

    axes[0].plot([0, 1], [0, 1], "k--", lw=1.1, label="Calibracao Perfeita (y = x)")
    axes[0].set_title(f"Diagrama de Confiabilidade Pos-Calibracao (LL: {ll_bruto:.4f} -> {ll_calib:.4f})")
    axes[0].set_xlabel("Probabilidade Predita Media (por decil)")
    axes[0].set_ylabel("Frequencia Real Observada")
    axes[0].set_xlim(0, 1.0)
    axes[0].set_ylim(0, 1.0)
    axes[0].legend(loc="upper left", fontsize=8.3, frameon=True)

    df_box = pd.DataFrame({
        "Classe Real": pd.Categorical([ROTULOS_CURTOS[int(v)] for v in y_true], categories=[ROTULOS_CURTOS[i] for i in range(4)], ordered=True),
        "P_Churn": proba_calib[:, 3],
    })
    sns.boxplot(
        data=df_box,
        x="Classe Real",
        y="P_Churn",
        hue="Classe Real",
        palette=CORES_CLASSES,
        legend=False,
        showfliers=False,
        ax=axes[1],
    )
    medians = df_box.groupby("Classe Real", observed=False)["P_Churn"].median()
    for i, med in enumerate(medians):
        axes[1].text(
            i,
            float(med) + 0.015,
            f"med={med:.2f}",
            ha="center",
            va="bottom",
            fontsize=8.3,
            fontweight="bold",
            bbox=dict(facecolor="white", edgecolor="none", alpha=0.82, pad=1.2),
        )
    axes[1].set_title("Separacao de P(Churn Efetivo | x) por Classe Real")
    axes[1].set_xlabel("Classe Real de Engajamento")
    axes[1].set_ylabel("Probabilidade Calibrada P(y = 3 | x)")
    axes[1].set_ylim(0, 1.02)

    fig.tight_layout()
    caminho = out_dir / "res_calibracao_confianca.png"
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "tabela_calibracao": df_cal,
        "log_loss_bruto": ll_bruto,
        "log_loss_calib": ll_calib,
        "fig_path": caminho,
    }


def calcular_teto_bayes_monte_carlo(
    proba_calib: np.ndarray,
    y_true: np.ndarray,
    pred_oof: np.ndarray,
    n_sim: int = 20,
    random_state: int = 42,
) -> Dict[str, float]:
    """Estima teto teorico de F1-macro sob ruído irredutivel via simulacao Monte Carlo em P_cal."""
    from scripts.experimentos import ajustar_regra_plugin_expected_f1, predizer_regra_bayes

    proba_calib = np.asarray(proba_calib, dtype=np.float64)
    proba_norm = proba_calib / np.maximum(proba_calib.sum(axis=1, keepdims=True), 1e-12)
    a_ref, b_ref = ajustar_regra_plugin_expected_f1(proba_norm)
    pred_ref = predizer_regra_bayes(proba_norm, a_ref, b_ref)

    cum_p = np.cumsum(proba_norm, axis=1)
    rng = np.random.default_rng(random_state)
    f1_sims = []
    n_obs = len(y_true)
    for _ in range(n_sim):
        u = rng.random((n_obs, 1))
        y_sim = np.clip((u > cum_p).sum(axis=1), 0, 3)
        f1_sims.append(float(f1_score(y_sim, pred_ref, average="macro")))

    f1_atual = float(f1_score(y_true, pred_oof, average="macro"))
    teto_med = float(np.mean(f1_sims))
    return {
        "teto_f1_media": round(teto_med, 4),
        "teto_f1_std": round(float(np.std(f1_sims)), 4),
        "f1_oof_atual": round(f1_atual, 4),
        "gap_para_teto": round(teto_med - f1_atual, 4),
    }


def analisar_desempenho_por_subgrupos(
    train_df: pd.DataFrame,
    y_true: np.ndarray,
    proba_calib: np.ndarray,
    pred_oof: np.ndarray,
    figs_dir: str | Path = "figs",
) -> Dict[str, Any]:
    """Desempenho por regime (1-dia vs 2+dias), filial (RUA vs SHOPPING) e top confusoes OOF."""
    configurar_estilo_graficos()
    out_dir = Path(figs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    y_true = np.asarray(y_true, dtype=int)
    pred_oof = np.asarray(pred_oof, dtype=int)

    is_single = (train_df["n_dias_com_compra"].to_numpy() == 1)
    regimes = [("1 Dia de Compra", is_single), ("2+ Dias de Compra", ~is_single)]
    linhas_reg = []
    for nm, m in regimes:
        f1s = f1_score(y_true[m], pred_oof[m], average=None, labels=[0, 1, 2, 3], zero_division=0)
        linhas_reg.append({
            "Regime": nm,
            "Clientes": int(m.sum()),
            "Pct_Base": float(m.mean() * 100.0),
            "Log_Loss": float(log_loss(y_true[m], np.clip(proba_calib[m], 1e-12, 1.0), labels=[0, 1, 2, 3])),
            "Accuracy": float(accuracy_score(y_true[m], pred_oof[m])),
            "Balanced_Acc": float(balanced_accuracy_score(y_true[m], pred_oof[m])),
            "AUC_Churn": float(roc_auc_score((y_true[m] == 3).astype(int), proba_calib[m, 3])),
            "F1_C0": float(f1s[0]),
            "F1_C1": float(f1s[1]),
            "F1_C2": float(f1s[2]),
            "F1_C3": float(f1s[3]),
        })
    df_reg = pd.DataFrame(linhas_reg)

    filial_ser = train_df["filial"].astype(str).str.strip().str.upper()
    linhas_fil = []
    for fil in sorted(filial_ser.unique()):
        m = (filial_ser == fil).to_numpy()
        f1s = f1_score(y_true[m], pred_oof[m], average=None, labels=[0, 1, 2, 3], zero_division=0)
        linhas_fil.append({
            "Filial": fil,
            "Clientes": int(m.sum()),
            "Pct_Base": float(m.mean() * 100.0),
            "Log_Loss": float(log_loss(y_true[m], np.clip(proba_calib[m], 1e-12, 1.0), labels=[0, 1, 2, 3])),
            "F1_macro": float(f1_score(y_true[m], pred_oof[m], average="macro", labels=[0, 1, 2, 3], zero_division=0)),
            "AUC_OvR_macro": float(roc_auc_score(y_true[m], proba_calib[m], multi_class="ovr", average="macro")),
            "F1_C0": float(f1s[0]),
            "F1_C1": float(f1s[1]),
            "F1_C2": float(f1s[2]),
            "F1_C3": float(f1s[3]),
        })
    df_fil = pd.DataFrame(linhas_fil)

    mask_err = (y_true != pred_oof)
    n_erros = int(mask_err.sum())
    pares = []
    for r in range(4):
        for p in range(4):
            if r == p:
                continue
            m_pair = (y_true == r) & (pred_oof == p)
            qtd = int(m_pair.sum())
            if qtd > 0:
                pares.append({
                    "Transicao": f"{ROTULOS_CURTOS[r]} -> {ROTULOS_CURTOS[p]}",
                    "Qtd_Erros": qtd,
                    "Pct_dos_Erros": float(qtd / max(n_erros, 1) * 100.0),
                    "Pct_da_Base": float(qtd / len(y_true) * 100.0),
                    "P_Real_Media": float(proba_calib[m_pair, r].mean()),
                    "P_Predita_Media": float(proba_calib[m_pair, p].mean()),
                })
    df_erros = pd.DataFrame(pares).sort_values("Qtd_Erros", ascending=False).head(6).reset_index(drop=True)

    fig, axes = plt.subplots(1, 3, figsize=(16.8, 4.9))
    reg_plot = df_reg.set_index("Regime")[["F1_C0", "F1_C1", "F1_C2", "F1_C3"]]
    reg_plot.columns = ["C0", "C1", "C2", "C3"]
    reg_plot.T.plot(kind="bar", ax=axes[0], color=["#34495e", "#2980b9"], width=0.68)
    axes[0].set_title("F1 por Classe: 1-Dia vs 2+Dias")
    axes[0].set_ylabel("F1-score")
    axes[0].set_ylim(0, 1.04)
    axes[0].tick_params(axis="x", rotation=0)
    axes[0].legend(frameon=True, facecolor="white", framealpha=0.9, loc="upper left", fontsize=8.5)

    fil_plot = df_fil.set_index("Filial")[["F1_macro", "Log_Loss"]]
    fil_plot.plot(kind="bar", ax=axes[1], color=["#27ae60", "#e67e22"], width=0.55)
    axes[1].set_title("Estabilidade por Filial (RUA vs SHOPPING)")
    axes[1].set_ylabel("Valor da Metrica")
    axes[1].set_ylim(0, 0.92)
    axes[1].tick_params(axis="x", rotation=0)
    for cont in axes[1].containers:
        axes[1].bar_label(cont, fmt="%.3f", fontsize=8, padding=2)
    axes[1].legend(frameon=False, fontsize=8.5)

    err_plot = df_erros.iloc[::-1]
    axes[2].barh(err_plot["Transicao"], err_plot["Pct_dos_Erros"], color="#c0392b", alpha=0.85)
    axes[2].set_title("Top 6 Pares de Confusao (% dos Erros OOF)")
    axes[2].set_xlabel("% do Total de Erros")
    axes[2].set_xlim(0, float(err_plot["Pct_dos_Erros"].max()) * 1.25)
    for idx_e, val_e in enumerate(err_plot["Pct_dos_Erros"]):
        axes[2].text(float(val_e) + 0.5, idx_e, f"{val_e:.1f}%", va="center", fontsize=8.5)

    fig.tight_layout()
    caminho = out_dir / "res_analise_subgrupos_erros.png"
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)

    return {
        "por_regime": df_reg,
        "por_filial": df_fil,
        "top_erros": df_erros,
        "fig_path": caminho,
    }


def gerar_painel_ensemble_e_estabilidade(
    pesos_medios: Dict[str, float],
    folds_f1_df: pd.DataFrame,
    figs_dir: str | Path = "figs",
) -> Path:
    """Plota pesos aprendidos no Blending Log-Odds por familia e estabilidade nos 5 folds."""
    configurar_estilo_graficos()
    out_dir = Path(figs_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def _familia(nome: str) -> str:
        if "Hierarquico" in nome:
            return "Hierarquico 3-Est."
        if "Especialista" in nome:
            return "Especialista Regime"
        if "Logistic" in nome or "MLP" in nome or "Spline" in nome:
            return "Linear / MLP / GAM"
        if "Random Forest" in nome or "Extra Trees" in nome:
            return "Arvores Bagging"
        return "GBDT / CatBoost"

    cor_fam = {
        "Hierarquico 3-Est.": "#2980b9",
        "Linear / MLP / GAM": "#8e44ad",
        "GBDT / CatBoost": "#16a085",
        "Especialista Regime": "#d35400",
        "Arvores Bagging": "#7f8c8d",
    }

    w_ser = pd.Series(pesos_medios)
    w_ser = w_ser[w_ser > 0.001].sort_values(ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(15.2, 5.4))
    cores_w = [cor_fam[_familia(str(nm))] for nm in w_ser.index]
    axes[0].barh(w_ser.index, w_ser.values * 100.0, color=cores_w)
    axes[0].set_title("Pesos Aprendidos no Blending Log-Odds (SLSQP Nested OOF)")
    axes[0].set_xlabel("Peso Medio (%)")
    axes[0].set_xlim(0, float(w_ser.max()) * 125.0)
    for i, val in enumerate(w_ser.values):
        axes[0].text(val * 100.0 + 0.4, i, f"{val * 100.0:.2f}%", va="center", fontsize=8.3)
    handles_f = [Patch(facecolor=c, label=k) for k, c in cor_fam.items()]
    axes[0].legend(handles=handles_f, loc="lower right", fontsize=8, frameon=True)

    medias = folds_f1_df.mean(axis=0).sort_values(ascending=False)
    sel_cols = list(medias.head(8).index)
    if "Decision Tree" in folds_f1_df.columns and "Decision Tree" not in sel_cols:
        sel_cols.append("Decision Tree")
    sel_cols = list(reversed(sel_cols))

    df_melt = folds_f1_df[sel_cols].melt(var_name="Modelo", value_name="F1_macro")
    sns.boxplot(data=df_melt, y="Modelo", x="F1_macro", order=sel_cols, color="#d6eaf8", width=0.55, showfliers=False, ax=axes[1])
    sns.stripplot(data=df_melt, y="Modelo", x="F1_macro", order=sel_cols, color="#1b4f72", size=5, alpha=0.85, ax=axes[1])
    axes[1].axvline(0.4070, color="#c0392b", linestyle="--", lw=1.2, label="Baseline AM-master (0.4070)")
    axes[1].set_title("Distribuicao do F1-macro nos 5 Folds (Top 8 vs Arvore Baseline)")
    axes[1].set_xlabel("F1-macro por Fold")
    axes[1].set_ylabel("")
    axes[1].legend(loc="lower left", fontsize=8.3, frameon=True)

    fig.tight_layout()
    caminho = out_dir / "res_ensemble_estabilidade.png"
    fig.savefig(caminho, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return caminho


def executar_validacao_adversarial(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    num_cols: List[str],
    n_splits: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Executa validacao adversarial treino vs. teste em n_splits dobras estratificadas
    para medir se ha deslocamento de distribuicao entre train.csv e test.csv.
    """
    from lightgbm import LGBMClassifier
    from sklearn.model_selection import StratifiedKFold

    cols_validas = [c for c in num_cols if c in X_train.columns and c in X_test.columns]
    X_adv = np.vstack([
        X_train[cols_validas].to_numpy(dtype=np.float64),
        X_test[cols_validas].to_numpy(dtype=np.float64),
    ])
    y_adv = np.concatenate([
        np.zeros(len(X_train), dtype=int),
        np.ones(len(X_test), dtype=int),
    ])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    oof_adv = np.zeros(len(y_adv), dtype=np.float64)
    aucs_fold = []
    imp_list = []

    for tr_idx, va_idx in skf.split(X_adv, y_adv):
        clf = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.05,
            max_depth=4,
            num_leaves=15,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=4,
            verbose=-1,
        )
        clf.fit(X_adv[tr_idx], y_adv[tr_idx])
        p_va = clf.predict_proba(X_adv[va_idx])[:, 1]
        oof_adv[va_idx] = p_va
        aucs_fold.append(float(roc_auc_score(y_adv[va_idx], p_va)))
        imp_list.append(clf.booster_.feature_importance(importance_type="gain"))

    auc_global = float(roc_auc_score(y_adv, oof_adv))
    status = (
        "Sem shift relevante (treino ~ teste)"
        if abs(auc_global - 0.5) < 0.03
        else "Possivel shift treino-teste"
    )
    top_features = (
        pd.DataFrame({
            "Atributo": cols_validas,
            "Ganho_Medio": np.mean(imp_list, axis=0),
        })
        .sort_values("Ganho_Medio", ascending=False)
        .head(5)
        .reset_index(drop=True)
    )
    return {
        "auc_adversarial": auc_global,
        "auc_folds_std": float(np.std(aucs_fold, ddof=1)) if len(aucs_fold) > 1 else 0.0,
        "status": status,
        "top_features": top_features,
    }


def simular_bootstrap_leaderboard(
    y_true: np.ndarray,
    pred_oof: np.ndarray,
    n_public: int = 7707,
    n_private: int = 17984,
    n_resamples: int = 300,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Simula a variabilidade amostral do F1-macro em subamostras sem reposicao
    com os tamanhos do Public Leaderboard (30%) e Private Leaderboard (70%).
    """
    from scripts.experimentos import fast_macro_f1

    y_arr = np.asarray(y_true, dtype=int)
    p_arr = np.asarray(pred_oof, dtype=int)
    rng = np.random.default_rng(random_state)
    n_total = len(y_arr)

    linhas = []
    for particao, n_sub in [
        ("Public LB (30%, n=7.707)", n_public),
        ("Private LB (70%, n=17.984)", n_private),
    ]:
        tam = min(int(n_sub), n_total)
        f1s = np.empty(n_resamples, dtype=np.float64)
        for i in range(n_resamples):
            idx = rng.choice(n_total, size=tam, replace=False)
            f1s[i] = fast_macro_f1(y_arr[idx], p_arr[idx])
        linhas.append({
            "Particao": particao,
            "Clientes": int(tam),
            "F1_Medio": float(np.mean(f1s)),
            "Desvio_Padrao": float(np.std(f1s, ddof=1)) if n_resamples > 1 else 0.0,
            "IC90_Inf": float(np.quantile(f1s, 0.05)),
            "IC90_Sup": float(np.quantile(f1s, 0.95)),
        })

    return pd.DataFrame(linhas)


