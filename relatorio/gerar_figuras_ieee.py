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

"""Gera figuras vetoriais (.pdf) em escala tipografica nativa IEEE (1 coluna) para o relatorio."""

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FuncFormatter
from sklearn.metrics import confusion_matrix, f1_score, roc_auc_score, roc_curve


def fmt_pt(val: float, decimals: int = 1) -> str:
    """Formata numero com virgula decimal."""
    return f"{val:.{decimals}f}".replace(".", ",")


def configurar_estilo_ieee() -> None:
    """Configura Matplotlib para figuras vetoriais IEEE (1 coluna ~3,35 pol, serif, Type 42)."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 7.2,
        "axes.titlesize": 7.6,
        "axes.titleweight": "bold",
        "axes.labelsize": 7.0,
        "xtick.labelsize": 6.3,
        "ytick.labelsize": 6.3,
        "legend.fontsize": 6.0,
        "axes.linewidth": 0.60,
        "axes.edgecolor": "#333333",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.color": "#d0d0d0",
        "grid.linestyle": "--",
        "grid.linewidth": 0.42,
        "grid.alpha": 0.70,
    })


def gerar_fig1_eda(train_df: pd.DataFrame, vendas_pre: pd.DataFrame, out_path: Path) -> None:
    """Figura 1 (1 coluna): (a) Classes por regime, (b) Volume mensal 2024 e (c) RUA vs SHOPPING."""
    mask_1d = train_df["n_dias_com_compra"].to_numpy(dtype=int) == 1
    y_all = train_df["target"].to_numpy(dtype=int)
    pct_1d = np.array([(y_all[mask_1d] == c).mean() * 100.0 for c in range(4)])
    pct_2p = np.array([(y_all[~mask_1d] == c).mean() * 100.0 for c in range(4)])

    dt = pd.to_datetime(vendas_pre["dt"])
    dt_2024 = dt[dt.dt.year == 2024]
    mensal = dt_2024.dt.month.value_counts().reindex(range(1, 10), fill_value=0).to_numpy() / 1e3

    canais = train_df.groupby("filial").agg(
        Recorrentes=("n_dias_com_compra", lambda s: float((s >= 2).mean() * 100.0)),
        Churn=("target", lambda s: float((s == 3).mean() * 100.0)),
        Ticket=("ticket_medio", "mean"),
    ).loc[["RUA", "SHOPPING"]]

    fig = plt.figure(figsize=(3.35, 2.88))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.05, 1.0], hspace=0.56, wspace=0.35)
    fig.subplots_adjust(left=0.13, right=0.985, top=0.93, bottom=0.13)

    # (a) Distribuicao das classes por regime
    ax = fig.add_subplot(gs[0, :])
    x = np.arange(4)
    w = 0.36
    b1 = ax.bar(x - w / 2, pct_1d, width=w, color="#5d6d7e", label=f"1-Dia (n={int(mask_1d.sum()):,})".replace(",", "."))
    b2 = ax.bar(x + w / 2, pct_2p, width=w, color="#1b9e77", label=f"2+Dias (n={int((~mask_1d).sum()):,})".replace(",", "."))
    ax.set_title("(a) Distribuição das Classes por Regime", pad=4)
    ax.set_xticks(x)
    ax.set_xticklabels(["C0 (Ativo)", "C1 (Desac.)", "C2 (Dorm.)", "C3 (Churn)"])
    ax.set_ylabel("Proporção (%)")
    ax.set_ylim(0, 96)
    ax.grid(True, axis="y")
    for bar, val in zip(b1, pct_1d):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.4, fmt_pt(val, 1), ha="center", va="bottom", fontsize=5.7)
    for bar, val in zip(b2, pct_2p):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.4, fmt_pt(val, 1), ha="center", va="bottom", fontsize=5.7)
    ax.legend(loc="upper left", frameon=False, ncol=2, fontsize=5.9, columnspacing=1.0, handlelength=1.3)

    # (b) Volume mensal de vendas pre-corte (Jan-Set/2024)
    ax = fig.add_subplot(gs[1, 0])
    meses = np.arange(1, 10)
    letras = ["J", "F", "M", "A", "M", "J", "J", "A", "S"]
    cores_m = ["#3498db" if i != 7 else "#1f4e79" for i in range(9)]
    ax.bar(meses, mensal, width=0.66, color=cores_m)
    pico_v = float(mensal[7])
    ax.text(8, pico_v + 1.4, f"Ago: {fmt_pt(pico_v, 1)}k", ha="center", va="bottom", fontsize=5.6, fontweight="bold", color="#1f4e79")
    ax.set_title("(b) Volume Mensal (2024)", pad=4)
    ax.set_xticks(meses)
    ax.set_xticklabels(letras, fontsize=6.0)
    ax.set_xlabel("Mês (Jan–Set/2024)", labelpad=2)
    ax.set_ylabel("Vendas (mil)")
    ax.set_ylim(0, 62)
    ax.grid(True, axis="y")

    # (c) Filiais RUA vs SHOPPING
    ax = fig.add_subplot(gs[1, 1])
    xc = np.arange(2)
    wc = 0.34
    rec_v = canais["Recorrentes"].to_numpy()
    ch_v = canais["Churn"].to_numpy()
    br1 = ax.bar(xc - wc / 2, rec_v, width=wc, color="#2980b9", label="Recorr. 2+d")
    br2 = ax.bar(xc + wc / 2, ch_v, width=wc, color="#d95f02", label="Churn C3")
    tk_r, tk_s = float(canais.loc["RUA", "Ticket"]), float(canais.loc["SHOPPING", "Ticket"])
    ax.set_title("(c) Filiais RUA vs. SHOPPING", pad=4)
    ax.set_xticks(xc)
    ax.set_xticklabels([f"RUA\n(R$ {fmt_pt(tk_r, 2)})", f"SHOP.\n(R$ {fmt_pt(tk_s, 2)})"], fontsize=5.9)
    ax.set_xlabel("Filial (Tíquete Médio)", labelpad=2)
    ax.set_ylabel("Taxa (%)")
    ax.set_ylim(0, 96)
    ax.grid(True, axis="y")
    for bar, val in zip(br1, rec_v):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{fmt_pt(val, 1)}%", ha="center", va="bottom", fontsize=5.7)
    for bar, val in zip(br2, ch_v):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1.5, f"{fmt_pt(val, 1)}%", ha="center", va="bottom", fontsize=5.7)
    ax.legend(loc="upper center", frameon=False, ncol=2, fontsize=5.4, handlelength=1.1, columnspacing=0.6)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def gerar_fig2_matrizes(y_true: np.ndarray, proba_oof: np.ndarray, pred_plug: np.ndarray, out_path: Path) -> None:
    """Figura 2 (1 coluna): Matrizes de confusao normalizadas Argmax vs Plug-in F1."""
    pred_arg = proba_oof.argmax(axis=1)
    f1_arg = f1_score(y_true, pred_arg, average="macro")
    f1_plg = f1_score(y_true, pred_plug, average="macro")

    fig, axes = plt.subplots(1, 2, figsize=(3.35, 1.90))
    fig.subplots_adjust(left=0.17, right=0.985, top=0.84, bottom=0.31, wspace=0.12)

    paineis = [
        (axes[0], pred_arg, f"(a) Argmax (F1 = {fmt_pt(f1_arg, 4)})"),
        (axes[1], pred_plug, f"(b) Plug-in (F1 = {fmt_pt(f1_plg, 4)})"),
    ]
    im_ref = None
    y_nomes = ["C0 Ativo", "C1 Desac.", "C2 Dorm.", "C3 Churn"]
    x_nomes = ["C0", "C1", "C2", "C3"]
    for idx_ax, (ax, pred, tit) in enumerate(paineis):
        cm = confusion_matrix(y_true, pred, labels=[0, 1, 2, 3])
        cm_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1) * 100.0
        im = ax.imshow(cm_pct, cmap="Blues", vmin=0.0, vmax=100.0, aspect="auto")
        im_ref = im
        ax.set_title(tit, fontsize=7.0, pad=4)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(x_nomes, fontsize=6.3)
        if idx_ax == 0:
            ax.set_yticklabels(y_nomes, fontsize=6.1)
            ax.set_ylabel("Classe Real", fontsize=6.8, labelpad=2)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Classe Predita", fontsize=6.8, labelpad=2)
        for s in ax.spines.values():
            s.set_visible(True)
            s.set_linewidth(0.55)
        for i in range(4):
            for j in range(4):
                val = cm_pct[i, j]
                cor = "white" if val > 48.0 else "#111111"
                peso = "bold" if i == j else "normal"
                ax.text(j, i, f"{fmt_pt(val, 1)}%", ha="center", va="center", color=cor, fontsize=5.9, fontweight=peso)

    cax = fig.add_axes([0.24, 0.08, 0.66, 0.045])
    cb = fig.colorbar(im_ref, cax=cax, orientation="horizontal")
    cb.set_label("% da Classe Real", fontsize=6.5, labelpad=1.5)
    cb.ax.tick_params(labelsize=6.0, length=2.5)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def gerar_fig3_roc_e_atributos(
    y_true: np.ndarray,
    proba_calib: np.ndarray,
    imp_df: pd.DataFrame,
    out_path: Path,
) -> None:
    """Figura 3 (1 coluna): (a) Curvas ROC-OvR por classe e (b) Top 10 atributos por ganho."""
    fig = plt.figure(figsize=(3.35, 3.20))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.12], hspace=0.46)
    fig.subplots_adjust(left=0.28, right=0.97, top=0.93, bottom=0.10)

    # (a) Curvas ROC-OvR por Classe
    ax_roc = fig.add_subplot(gs[0, 0])
    cores = ["#1f4e79", "#d95f02", "#1b9e77", "#7b2cbf"]
    nomes_cls = ["C0 Ativo", "C1 Desac.", "C2 Dorm.", "C3 Churn"]
    aucs = []
    for c in range(4):
        y_bin = (y_true == c).astype(int)
        fpr, tpr, _ = roc_curve(y_bin, proba_calib[:, c])
        auc_c = roc_auc_score(y_bin, proba_calib[:, c])
        aucs.append(auc_c)
        if len(fpr) > 400:
            idx = np.linspace(0, len(fpr) - 1, 400, dtype=int)
            fpr, tpr = fpr[idx], tpr[idx]
        ax_roc.plot(fpr, tpr, color=cores[c], lw=1.30, label=f"{nomes_cls[c]} ({fmt_pt(auc_c, 3)})")
    ax_roc.plot([0, 1], [0, 1], "--", color="#888888", lw=0.85, label="Aleat. (0,500)")
    auc_macro = float(np.mean(aucs))
    ax_roc.set_title(f"(a) Curvas ROC-OvR por Classe (Macro = {fmt_pt(auc_macro, 3)})", pad=4)
    ax_roc.set_xlabel("Taxa de Falsos Positivos (FPR)", labelpad=2)
    ax_roc.set_ylabel("Taxa de Verd. Pos. (TPR)")
    ax_roc.set_xlim(0.0, 1.0)
    ax_roc.set_ylim(0.0, 1.03)
    ax_roc.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_roc.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax_roc.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_pt(v, 1)))
    ax_roc.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_pt(v, 1)))
    ax_roc.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_pt(v, 1)))
    ax_roc.yaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_pt(v, 1)))
    ax_roc.grid(True)
    ax_roc.legend(loc="lower right", frameon=True, framealpha=0.93, edgecolor="#cccccc", fontsize=5.6, handlelength=1.3)

    # (b) Top 10 Atributos por Ganho Medio (LightGBM)
    top10 = imp_df.head(10).iloc[::-1].reset_index(drop=True)
    gains_k = top10["Ganho_Medio"].to_numpy(dtype=np.float64) / 1e3
    labels_y = top10["Atributo"].tolist()

    ax_imp = fig.add_subplot(gs[1, 0])
    y_pos = np.arange(len(top10))
    colors_bar = ["#1f4e79" if g >= 12.0 else "#4a7ebb" for g in gains_k]
    bars = ax_imp.barh(y_pos, gains_k, height=0.68, color=colors_bar)
    ax_imp.set_title("(b) Top 10 Atributos por Ganho Médio (LightGBM)", pad=4)
    ax_imp.set_yticks(y_pos)
    ax_imp.set_yticklabels(labels_y, fontsize=5.8, family="monospace")
    ax_imp.set_xlabel("Ganho Médio por Divisão (mil)", labelpad=2)
    ax_imp.set_xlim(0, float(gains_k.max()) * 1.24)
    ax_imp.xaxis.set_major_formatter(FuncFormatter(lambda v, _: fmt_pt(v, 0)))
    ax_imp.grid(True, axis="x")
    for bar, val in zip(bars, gains_k):
        ax_imp.text(val + 0.8, bar.get_y() + bar.get_height() / 2, fmt_pt(val, 1), va="center", ha="left", fontsize=5.6)

    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def main() -> None:
    configurar_estilo_ieee()
    out_dir = Path("relatorio/figs")
    out_dir.mkdir(parents=True, exist_ok=True)
    for old_name in ("fig1_eda_negocio.pdf", "fig3_roi_negocio.pdf", "fig3_roc_e_roi.pdf"):
        old_path = out_dir / old_name
        if old_path.exists():
            old_path.unlink()
    for old in out_dir.glob("*.png"):
        old.unlink()

    base = joblib.load("cache/00_bases_limpas.joblib")
    train_df = base["train_df"]
    vendas_pre = base["vendas_pre"]
    y_true = train_df["target"].to_numpy(dtype=int)

    v4 = joblib.load("cache/v4_08_exp6b_10fold_ensemble.joblib")
    ens = v4["ensemble"]
    proba_oof = ens["blend_oof"]
    proba_calib = ens["posproc"]["proba_proc"]
    pred_plug = ens["posproc"]["pred_oof"]

    sel = joblib.load("cache/v3_03b_selecao_atributos.joblib")
    imp_df = sel["importancias_completas"]

    gerar_fig1_eda(train_df, vendas_pre, out_dir / "fig1_eda.pdf")
    gerar_fig2_matrizes(y_true, proba_oof, pred_plug, out_dir / "fig2_matrizes_confusao.pdf")
    gerar_fig3_roc_e_atributos(y_true, proba_calib, imp_df, out_dir / "fig3_roc_e_atributos.pdf")
    print("Figuras vetoriais IEEE (1 coluna) atualizadas:", sorted(p.name for p in out_dir.glob("*.pdf")))


if __name__ == "__main__":
    main()
