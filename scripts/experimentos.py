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

# Arquivo com todas as funcoes e codigos referentes aos experimentos

from __future__ import annotations

import re
import time
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from lightgbm import LGBMClassifier
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

try:
    from catboost import CatBoostClassifier, Pool
    HAS_CATBOOST = True
except ImportError:
    CatBoostClassifier = None
    Pool = None
    HAS_CATBOOST = False

from scripts.preprocessamento import (
    BGNBDFeatureTransformer,
    criar_preprocessador,
    filtrar_outliers_iqr,
)


N_SPLITS_DEFAULT = 5
RANDOM_STATE_DEFAULT = 42
N_JOBS_DEFAULT = 4


def carregar_ou_executar(
    nome_arquivo: str,
    fn: Callable[[], Any],
    cache_dir: str | Path = "cache",
    usar_cache: bool = True,
    forcar: bool = False,
) -> Any:
    """Executa fn() apenas se o artefato nao estiver em cache/, salvando em disco via joblib."""
    pasta = Path(cache_dir)
    pasta.mkdir(parents=True, exist_ok=True)
    caminho = pasta / nome_arquivo

    if usar_cache and not forcar and caminho.exists():
        t0 = time.time()
        res = joblib.load(caminho)
        print(f"[CACHE] Carregado {caminho} ({time.time() - t0:.2f}s)", flush=True)
        return res

    t0 = time.time()
    res = fn()
    joblib.dump(res, caminho, compress=3)
    print(f"[SALVO] {caminho} ({time.time() - t0:.1f}s)", flush=True)
    return res


def fast_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 4) -> float:
    """Calcula F1-macro exato de forma vetorizada via bincount."""
    cm = np.bincount(y_true * n_classes + y_pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    denom = 2 * tp + fp + fn
    f1 = np.where(denom > 0, 2.0 * tp / denom, 0.0)
    return float(f1.mean())


def calcular_metricas_completas(
    y_true: np.ndarray,
    proba: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Calcula F1-macro, AUC-OvR macro, Log-Loss, Balanced Accuracy, Accuracy e F1 por classe."""
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=np.float64)
    if y_pred is None:
        y_pred = np.argmax(proba, axis=1)
    else:
        y_pred = np.asarray(y_pred, dtype=int)

    somas = proba.sum(axis=1, keepdims=True)
    proba_norm = np.where(somas > 0, proba / np.maximum(somas, 1e-12), 0.25)
    proba_clip = np.clip(proba_norm, 1e-12, 1.0)
    proba_clip /= proba_clip.sum(axis=1, keepdims=True)

    f1_por_classe = f1_score(y_true, y_pred, labels=[0, 1, 2, 3], average=None, zero_division=0)
    try:
        auc_ovr = float(roc_auc_score(y_true, proba_clip, multi_class="ovr", average="macro"))
    except ValueError:
        auc_ovr = np.nan

    try:
        ll = float(log_loss(y_true, proba_clip, labels=[0, 1, 2, 3]))
    except ValueError:
        ll = np.nan

    return {
        "F1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "AUC_OvR_macro": auc_ovr,
        "Log_Loss": ll,
        "Balanced_Acc": float(balanced_accuracy_score(y_true, y_pred)),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "F1_C0": float(f1_por_classe[0]),
        "F1_C1": float(f1_por_classe[1]),
        "F1_C2": float(f1_por_classe[2]),
        "F1_C3": float(f1_por_classe[3]),
    }


def criar_mascara_estrutural(df: pd.DataFrame) -> np.ndarray:
    """Mascara booleana (N, 4) com classes estruturalmente possiveis pelo ciclo/dias de compra."""
    n = len(df)
    mask = np.ones((n, 4), dtype=bool)
    single_day = (df["n_dias_com_compra"].to_numpy() == 1)
    ciclo = df["ciclo_medio_dias"].to_numpy(dtype=np.float64)

    impossivel_c2 = single_day | (np.isfinite(ciclo) & (ciclo > 46.5))
    mask[impossivel_c2, 2] = False

    impossivel_c1 = (~single_day) & np.isfinite(ciclo) & (ciclo > 77.0)
    mask[impossivel_c1, 1] = False
    return mask


def aplicar_mascara_estrutural(proba: np.ndarray, df: pd.DataFrame) -> np.ndarray:
    """Zera probabilidades de classes estruturalmente impossiveis e renormaliza."""
    mask = criar_mascara_estrutural(df)
    p = np.where(mask, proba, 0.0)
    somas = p.sum(axis=1, keepdims=True)
    return np.where(somas > 0, p / np.maximum(somas, 1e-12), proba)


def ajustar_calibracao_logit(proba: np.ndarray, y_true: np.ndarray) -> Tuple[float, np.ndarray]:
    """
    Calibracao multiclasse por temperatura e vies nos logits:
    softmax(inv_T * log(p) + bias), otimizando log-loss com apenas 4 graus de liberdade.
    """
    y_true = np.asarray(y_true, dtype=int)
    logits = np.log(np.clip(proba, 1e-7, 1.0))
    idx_rows = np.arange(len(y_true))

    def loss_fn(params):
        inv_T = params[0]
        bias = np.array([params[1], params[2], params[3], 0.0], dtype=np.float64)
        z = logits * inv_T + bias[None, :]
        z -= z.max(axis=1, keepdims=True)
        lse = np.log(np.sum(np.exp(z), axis=1))
        nll = -np.mean(z[idx_rows, y_true] - lse)
        return float(nll) + 1e-4 * float(np.sum(bias ** 2) + (inv_T - 1.0) ** 2)

    res = minimize(
        loss_fn,
        x0=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        method="L-BFGS-B",
        bounds=[(0.2, 3.0), (-2.0, 2.0), (-2.0, 2.0), (-2.0, 2.0)],
    )
    inv_T = float(res.x[0])
    bias = np.array([res.x[1], res.x[2], res.x[3], 0.0], dtype=np.float64)
    return inv_T, bias


def aplicar_calibracao_logit(proba: np.ndarray, inv_T: float, bias: np.ndarray) -> np.ndarray:
    """Aplica temperatura e vies calibrados e retorna probabilidades normalizadas."""
    logits = np.log(np.clip(proba, 1e-7, 1.0))
    z = logits * inv_T + bias[None, :]
    z -= z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)


def ajustar_regra_bayes_f1(
    proba: np.ndarray,
    y_true: np.ndarray,
    n_iters: int = 80,
    damping: float = 0.4,
    usar_powell: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Regra otima de Bayes afim para maximizar F1-macro via ponto fixo vetorizado,
    com polimento opcional Powell diretamente no F1-macro do fold de treino.
    """
    y_true = np.asarray(y_true, dtype=int)
    proba = np.asarray(proba, dtype=np.float64)
    N = len(y_true)
    N_c = np.bincount(y_true, minlength=4).astype(np.float64)

    a_init = N / np.maximum(N_c, 1.0)
    b_init = np.zeros(4, dtype=np.float64)
    pred = np.argmax(proba * a_init[None, :], axis=1)
    best_f1 = fast_macro_f1(y_true, pred)
    best_a, best_b = a_init.copy(), b_init.copy()

    cm = np.bincount(y_true * 4 + pred, minlength=16).reshape(4, 4)
    tp = np.diag(cm).astype(np.float64)
    n_hat = cm.sum(axis=0).astype(np.float64)
    F = np.where(N_c + n_hat > 0, 2.0 * tp / (N_c + n_hat), 0.0)
    denom = np.maximum(N_c + n_hat, 1.0)

    for _ in range(n_iters):
        a = 2.0 / denom
        b = -F / denom
        pred_new = np.argmax(proba * a[None, :] + b[None, :], axis=1)
        f1_new = fast_macro_f1(y_true, pred_new)
        if f1_new > best_f1:
            best_f1 = f1_new
            best_a, best_b = a.copy(), b.copy()

        cm = np.bincount(y_true * 4 + pred_new, minlength=16).reshape(4, 4)
        tp_new = np.diag(cm).astype(np.float64)
        n_hat_new = cm.sum(axis=0).astype(np.float64)
        F_new = np.where(N_c + n_hat_new > 0, 2.0 * tp_new / (N_c + n_hat_new), 0.0)
        denom_new = np.maximum(N_c + n_hat_new, 1.0)
        F = (1.0 - damping) * F + damping * F_new
        denom = (1.0 - damping) * denom + damping * denom_new

    if usar_powell and best_a[3] > 0:
        scale = best_a[3]
        x0 = np.concatenate([best_a[:3] / scale, best_b / scale])

        def neg_f1(params):
            aa = np.array([params[0], params[1], params[2], 1.0], dtype=np.float64)
            bb = params[3:7]
            yp = np.argmax(proba * aa[None, :] + bb[None, :], axis=1)
            return -fast_macro_f1(y_true, yp)

        res_p = minimize(neg_f1, x0, method="Powell", options={"maxiter": 3, "ftol": 1e-4})
        aa_p = np.array([res_p.x[0], res_p.x[1], res_p.x[2], 1.0], dtype=np.float64)
        bb_p = res_p.x[3:7]
        f1_p = fast_macro_f1(y_true, np.argmax(proba * aa_p[None, :] + bb_p[None, :], axis=1))
        if f1_p > best_f1:
            best_a, best_b = aa_p, bb_p

    return best_a, best_b


def predizer_regra_bayes(proba: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Aplica a regra afim Bayes-otima (score = p * a + b) e retorna argmax."""
    return np.argmax(np.asarray(proba, dtype=np.float64) * a[None, :] + b[None, :], axis=1)


def ajustar_regra_plugin_expected_f1(
    proba: np.ndarray,
    n_iters: int = 80,
    damping: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Regra Plug-in de Expected Macro-F1 (Lipton et al., 2014): otimiza limiares afins
    exclusivamente sobre as probabilidades calibradas P(y|x), sem consultar y_true.
    """
    proba = np.asarray(proba, dtype=np.float64)
    N, n_cls = proba.shape
    idx_rows = np.arange(N)

    N_c_exp = proba.sum(axis=0)
    a_init = N / np.maximum(N_c_exp, 1.0)
    b_init = np.zeros(n_cls, dtype=np.float64)

    def _expected_stats(pred_vec: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
        p_sel = proba[idx_rows, pred_vec]
        tp_exp = np.bincount(pred_vec, weights=p_sel, minlength=n_cls).astype(np.float64)
        n_hat = np.bincount(pred_vec, minlength=n_cls).astype(np.float64)
        denom_c = np.maximum(N_c_exp + n_hat, 1.0)
        f_c = np.where(N_c_exp + n_hat > 0, 2.0 * tp_exp / denom_c, 0.0)
        return float(np.mean(f_c)), f_c, denom_c

    pred = np.argmax(proba * a_init[None, :], axis=1)
    best_exp_f1, F, denom = _expected_stats(pred)
    best_a, best_b = a_init.copy(), b_init.copy()

    for _ in range(n_iters):
        a = 2.0 / denom
        b = -F / denom
        pred_new = np.argmax(proba * a[None, :] + b[None, :], axis=1)
        exp_f1_new, F_new, denom_new = _expected_stats(pred_new)
        if exp_f1_new > best_exp_f1:
            best_exp_f1 = exp_f1_new
            best_a, best_b = a.copy(), b.copy()
        F = (1.0 - damping) * F + damping * F_new
        denom = (1.0 - damping) * denom + damping * denom_new

    return best_a, best_b


def avaliar_posproc_nested_oof(
    proba_oof: np.ndarray,
    y_true: np.ndarray,
    df_features: pd.DataFrame,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    usar_mascara: bool = True,
    por_regime: bool = False,
    calibrar: bool = True,
    usar_powell: bool = False,
    metodo_regra: str = "plugin",
) -> Dict[str, Any]:
    """
    Valida calibracao logit + mascara estrutural + regra de decisao ('plugin' expected F1,
    'bayes' ou 'bayes_powell') de forma estritamente Nested OOF (fold k ajustado em != k).
    """
    y_true = np.asarray(y_true, dtype=int)
    is_single = (df_features["n_dias_com_compra"].to_numpy() == 1)

    def _ajustar_regra(p_mat: np.ndarray, y_vec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if metodo_regra == "plugin" and not usar_powell:
            return ajustar_regra_plugin_expected_f1(p_mat)
        use_pw = usar_powell or (metodo_regra == "bayes_powell")
        return ajustar_regra_bayes_f1(p_mat, y_vec, usar_powell=use_pw)

    pred_oof = np.empty(len(y_true), dtype=int)
    proba_calib_oof = np.zeros_like(proba_oof, dtype=np.float64)
    metricas_folds = []

    for tr_idx, va_idx in folds:
        p_tr = proba_oof[tr_idx].copy()
        p_va = proba_oof[va_idx].copy()

        if calibrar:
            inv_t, bias = ajustar_calibracao_logit(p_tr, y_true[tr_idx])
            p_tr = aplicar_calibracao_logit(p_tr, inv_t, bias)
            p_va = aplicar_calibracao_logit(p_va, inv_t, bias)

        if usar_mascara:
            p_tr = aplicar_mascara_estrutural(p_tr, df_features.iloc[tr_idx])
            p_va = aplicar_mascara_estrutural(p_va, df_features.iloc[va_idx])

        proba_calib_oof[va_idx] = p_va

        if por_regime:
            a_s, b_s = _ajustar_regra(p_tr[is_single[tr_idx]], y_true[tr_idx][is_single[tr_idx]])
            a_m, b_m = _ajustar_regra(p_tr[~is_single[tr_idx]], y_true[tr_idx][~is_single[tr_idx]])
            pred_va = np.empty(len(va_idx), dtype=int)
            m_va = is_single[va_idx]
            if np.any(m_va):
                pred_va[m_va] = predizer_regra_bayes(p_va[m_va], a_s, b_s)
            if np.any(~m_va):
                pred_va[~m_va] = predizer_regra_bayes(p_va[~m_va], a_m, b_m)
        else:
            a, b = _ajustar_regra(p_tr, y_true[tr_idx])
            pred_va = predizer_regra_bayes(p_va, a, b)

        pred_oof[va_idx] = pred_va
        metricas_folds.append(calcular_metricas_completas(y_true[va_idx], p_va, pred_va))

    # Parametros finais ajustados sobre OOF completo para aplicacao no test.csv
    if calibrar:
        inv_T_fin, bias_fin = ajustar_calibracao_logit(proba_oof, y_true)
        proba_proc = aplicar_calibracao_logit(proba_oof, inv_T_fin, bias_fin)
    else:
        inv_T_fin, bias_fin = 1.0, np.zeros(4, dtype=np.float64)
        proba_proc = proba_oof.copy()

    if usar_mascara:
        proba_proc = aplicar_mascara_estrutural(proba_proc, df_features)

    a_final, b_final = _ajustar_regra(proba_proc, y_true)
    if por_regime:
        a_s_fin, b_s_fin = _ajustar_regra(proba_proc[is_single], y_true[is_single])
        a_m_fin, b_m_fin = _ajustar_regra(proba_proc[~is_single], y_true[~is_single])
    else:
        a_s_fin, b_s_fin = a_final, b_final
        a_m_fin, b_m_fin = a_final, b_final

    return {
        "pred_oof": pred_oof,
        "proba_proc": proba_calib_oof,
        "metricas_oof": calcular_metricas_completas(y_true, proba_calib_oof, pred_oof),
        "metricas_por_fold": pd.DataFrame(metricas_folds),
        "inv_T_final": inv_T_fin,
        "bias_final": bias_fin,
        "a_final": a_final,
        "b_final": b_final,
        "a_single_final": a_s_fin,
        "b_single_final": b_s_fin,
        "a_multi_final": a_m_fin,
        "b_multi_final": b_m_fin,
        "por_regime": por_regime,
        "calibrar": calibrar,
        "usar_mascara": usar_mascara,
        "metodo_regra": metodo_regra,
    }


def gerar_folds_estratificados(
    y: np.ndarray,
    n_splits: int = N_SPLITS_DEFAULT,
    random_state: int = RANDOM_STATE_DEFAULT,
) -> List[Tuple[np.ndarray, np.ndarray]]:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    return list(skf.split(np.zeros(len(y)), y))


def rodar_cv_oof(
    estimator_fn: Callable[[], Any],
    X: pd.DataFrame,
    y: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
    sample_weight_mode: Optional[str] = None,
    power_alpha: float = 0.5,
    iqr_cols: Optional[List[str]] = None,
    X_test: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Executa validacao cruzada OOF nos folds fixos sem vazamento entre treino e validacao."""
    y = np.asarray(y, dtype=int)
    oof_proba = np.zeros((len(X), 4), dtype=np.float64)
    test_proba = np.zeros((len(X_test), 4), dtype=np.float64) if X_test is not None else None
    metricas_folds = []

    for tr_idx, va_idx in folds:
        X_tr = X.iloc[tr_idx].reset_index(drop=True)
        y_tr = y[tr_idx]
        X_va = X.iloc[va_idx].reset_index(drop=True)
        y_va = y[va_idx]

        if iqr_cols is not None:
            X_tr, y_tr = filtrar_outliers_iqr(X_tr, y_tr, iqr_cols, fator=1.5)

        est = estimator_fn()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if sample_weight_mode == "balanced":
                sw = compute_sample_weight("balanced", y_tr)
                est.fit(X_tr, y_tr, classifier__sample_weight=sw)
            elif sample_weight_mode == "suave":
                counts = np.bincount(y_tr, minlength=4).astype(np.float64)
                pesos_c = (len(y_tr) / (4.0 * np.maximum(counts, 1.0))) ** power_alpha
                sw = pesos_c[y_tr]
                est.fit(X_tr, y_tr, classifier__sample_weight=sw)
            else:
                est.fit(X_tr, y_tr)

            proba_va = est.predict_proba(X_va)
            oof_proba[va_idx] = proba_va
            if X_test is not None:
                test_proba += est.predict_proba(X_test) / len(folds)

        metricas_folds.append(calcular_metricas_completas(y_va, proba_va))

    return {
        "oof_proba": oof_proba,
        "test_proba": test_proba,
        "metricas_argmax": calcular_metricas_completas(y, oof_proba),
        "metricas_folds_argmax": pd.DataFrame(metricas_folds),
    }


def rodar_cv_gbdt_early_stopping(
    familia: str,
    params: Dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    grupos: Dict[str, List[str]],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    X_test: Optional[pd.DataFrame] = None,
    early_stopping_rounds: int = 40,
    n_classes_sub: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """
    Treina GBDTs (LightGBM, XGBoost ou CatBoost) com Early Stopping por fold interno (88/12)
    ajustando BG/NBD e TargetEncoder exclusivamente no sub-treino interno (zero leakage!).
    """
    y = np.asarray(y, dtype=int)
    n_cls = len(n_classes_sub) if n_classes_sub is not None else 4
    oof_proba = np.zeros((len(X), n_cls), dtype=np.float64)
    test_proba = np.zeros((len(X_test), n_cls), dtype=np.float64) if X_test is not None else None
    importances_list = []
    feature_names_out = None
    best_iters = []
    metricas_folds = []

    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        sub_tr_idx, inner_va_idx = train_test_split(
            tr_idx, test_size=0.12, stratify=y[tr_idx], random_state=RANDOM_STATE_DEFAULT + fold_id
        )
        prep = criar_preprocessador(
            grupos["numericas"],
            grupos["categoricas"],
            grupos["alta_card"],
            modo="tree_native",
            target_type="binary" if n_cls == 2 else "auto",
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_sub = prep.fit_transform(X.iloc[sub_tr_idx], y[sub_tr_idx])
            X_inn = prep.transform(X.iloc[inner_va_idx])
            X_va_t = prep.transform(X.iloc[va_idx])
            X_te_t = prep.transform(X_test) if (X_test is not None and len(X_test) > 0) else None

        if feature_names_out is None:
            feature_names_out = [str(n).split("__", 1)[-1] for n in prep.get_feature_names_out()]

        y_sub, y_inn = y[sub_tr_idx], y[inner_va_idx]

        if familia == "lgbm":
            lgbm_params = dict(params)
            if n_cls == 2:
                lgbm_params["objective"] = "binary"
            clf = LGBMClassifier(**lgbm_params)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                clf.fit(
                    X_sub, y_sub,
                    eval_set=[(X_inn, y_inn)],
                    callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
                )
            best_iters.append(int(clf.best_iteration_ or lgbm_params.get("n_estimators", 300)))
            importances_list.append(clf.booster_.feature_importance(importance_type="gain"))

        elif familia == "xgb":
            xgb_params = dict(params)
            xgb_params["early_stopping_rounds"] = early_stopping_rounds
            if n_cls == 2:
                xgb_params["objective"] = "binary:logistic"
                xgb_params["eval_metric"] = "logloss"
            clf = XGBClassifier(**xgb_params)
            clf.fit(X_sub, y_sub, eval_set=[(X_inn, y_inn)], verbose=False)
            best_iters.append(int(getattr(clf, "best_iteration", xgb_params.get("n_estimators", 300)) or 100))
            importances_list.append(clf.feature_importances_)

        elif familia == "catboost":
            if not HAS_CATBOOST:
                raise RuntimeError("CatBoost nao instalado.")
            clf = CatBoostClassifier(**params)
            clf.fit(X_sub, y_sub, eval_set=(X_inn, y_inn), early_stopping_rounds=early_stopping_rounds, verbose=False)
            best_iters.append(int(clf.get_best_iteration() or params.get("iterations", 400)))
            importances_list.append(clf.get_feature_importance())
        else:
            raise ValueError(f"Familia desconhecida: {familia}")

        p_va = clf.predict_proba(X_va_t)
        oof_proba[va_idx] = p_va
        if X_test is not None and X_te_t is not None:
            test_proba += clf.predict_proba(X_te_t) / len(folds)

        if n_cls == 4:
            metricas_folds.append(calcular_metricas_completas(y[va_idx], p_va))

    mean_imp = np.mean(importances_list, axis=0) if importances_list else None
    imp_df = (
        pd.DataFrame({"Atributo": feature_names_out, "Ganho_Medio": mean_imp})
        .sort_values("Ganho_Medio", ascending=False)
        .reset_index(drop=True)
        if mean_imp is not None
        else None
    )

    return {
        "oof_proba": oof_proba,
        "test_proba": test_proba,
        "best_iters": best_iters,
        "mean_best_iter": int(np.mean(best_iters)) if best_iters else 0,
        "importancias": imp_df,
        "metricas_argmax": calcular_metricas_completas(y, oof_proba) if n_cls == 4 else {},
        "metricas_folds_argmax": pd.DataFrame(metricas_folds) if n_cls == 4 else pd.DataFrame(),
    }


def _construir_lgbm_padrao(random_state: int = RANDOM_STATE_DEFAULT, n_estimators: int = 350, **kwargs) -> LGBMClassifier:
    params = {
        "n_estimators": n_estimators,
        "learning_rate": 0.03,
        "max_depth": 5,
        "num_leaves": 20,
        "min_child_samples": 100,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.55,
        "reg_alpha": 2.0,
        "reg_lambda": 8.0,
        "random_state": random_state,
        "n_jobs": N_JOBS_DEFAULT,
        "verbose": -1,
    }
    params.update(kwargs)
    return LGBMClassifier(**params)


def _params_xgb_padrao(n_estimators: int = 1200) -> Dict[str, Any]:
    return {
        "n_estimators": n_estimators,
        "learning_rate": 0.03,
        "max_depth": 4,
        "min_child_weight": 40,
        "subsample": 0.8,
        "colsample_bytree": 0.55,
        "reg_alpha": 2.0,
        "reg_lambda": 10.0,
        "gamma": 0.2,
        "tree_method": "hist",
        "eval_metric": "mlogloss",
        "random_state": RANDOM_STATE_DEFAULT,
        "n_jobs": N_JOBS_DEFAULT,
    }


def _params_lgbm_padrao(n_estimators: int = 1200) -> Dict[str, Any]:
    return {
        "n_estimators": n_estimators,
        "learning_rate": 0.03,
        "max_depth": 5,
        "num_leaves": 20,
        "min_child_samples": 100,
        "subsample": 0.8,
        "subsample_freq": 1,
        "colsample_bytree": 0.55,
        "reg_alpha": 2.0,
        "reg_lambda": 8.0,
        "random_state": RANDOM_STATE_DEFAULT,
        "n_jobs": N_JOBS_DEFAULT,
        "verbose": -1,
    }


def executar_ablacao_preprocessamento(
    X_base: pd.DataFrame,
    y: np.ndarray,
    grupos: Dict[str, List[str]],
    folds: List[Tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Experimento 1: compara remocao IQR 1.5x (AM-master) vs manutencao integral e NaN nativo."""
    num_cols = grupos["numericas"]
    cat_cols = grupos["categoricas"]
    alta_card = grupos["alta_card"]

    cenarios = [
        ("A1: IQR 1.5x (AM-master) + Mediana + RF balanced", "tree_imputed", num_cols, "rf", "balanced"),
        ("A2: Sem IQR (100% dados) + Mediana + RF balanced", "tree_imputed", None, "rf", "balanced"),
        ("B1: IQR 1.5x + Mediana + LGBM balanced", "tree_imputed", num_cols, "lgbm", "balanced"),
        ("B2: Sem IQR (100% dados) + Mediana + LGBM balanced", "tree_imputed", None, "lgbm", "balanced"),
        ("B3: Sem IQR (100% dados) + NaN nativo + LGBM balanced", "tree_native", None, "lgbm", "balanced"),
        ("B4: Sem IQR + NaN nativo + LGBM + Regra Bayes F1 (OOF)", "tree_native", None, "lgbm", None),
    ]

    linhas = []
    total = len(cenarios)
    for idx, (nome, modo, iqr_c, modelo, sw_mode) in enumerate(cenarios, start=1):
        t0 = time.time()

        def make_pipe(m=modo, mod=modelo, swm=sw_mode):
            prep = criar_preprocessador(num_cols, cat_cols, alta_card, modo=m, usar_bgnbd=False)
            if mod == "rf":
                clf = RandomForestClassifier(
                    n_estimators=120, max_depth=18, min_samples_leaf=5,
                    class_weight="balanced_subsample" if swm == "balanced" else None,
                    random_state=RANDOM_STATE_DEFAULT, n_jobs=N_JOBS_DEFAULT,
                )
                return Pipeline([("preprocessor", prep), ("classifier", clf)])
            clf = _construir_lgbm_padrao(n_estimators=150)
            return Pipeline([("preprocessor", prep), ("classifier", clf)])

        res = rodar_cv_oof(
            make_pipe, X_base, y, folds,
            sample_weight_mode="balanced" if (modelo == "lgbm" and sw_mode == "balanced") else None,
            iqr_cols=iqr_c,
        )
        if "Bayes" in nome:
            pos = avaliar_posproc_nested_oof(
                res["oof_proba"], y, X_base, folds, usar_mascara=True, por_regime=False, calibrar=False, usar_powell=False, metodo_regra="bayes"
            )
            met = pos["metricas_oof"]
            folds_df = pos["metricas_por_fold"]
        else:
            met = res["metricas_argmax"]
            folds_df = res["metricas_folds_argmax"]

        print(f"  [{idx}/{total}] {nome} -> F1_macro={met['F1_macro']:.4f} | AUC={met['AUC_OvR_macro']:.4f} ({time.time() - t0:.1f}s)", flush=True)
        linhas.append({
            "Cenario": nome,
            "F1_macro": met["F1_macro"],
            "F1_macro_std": float(folds_df["F1_macro"].std()),
            "AUC_OvR_macro": met["AUC_OvR_macro"],
            "Balanced_Acc": met["Balanced_Acc"],
            "F1_C0": met["F1_C0"],
            "F1_C1": met["F1_C1"],
            "F1_C2": met["F1_C2"],
            "F1_C3": met["F1_C3"],
        })
    return pd.DataFrame(linhas)


def executar_ablacao_features(
    conjuntos: Dict[str, Tuple[pd.DataFrame, Dict[str, List[str]], bool]],
    y: np.ndarray,
    folds: List[Tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Experimento 2: compara Original vs +Grupo A vs +Grupo A+B+BG/NBD (Early Stopping) em 5 folds."""
    linhas = []
    total = len(conjuntos)
    for idx, (nome_nivel, (X_niv, grupos, com_bgnbd)) in enumerate(conjuntos.items(), start=1):
        t0 = time.time()
        if com_bgnbd:
            res_unw = rodar_cv_gbdt_early_stopping(
                "lgbm", _params_lgbm_padrao(1200), X_niv, y, grupos, folds
            )
            n_cols_efetivas = len(res_unw["importancias"]) if res_unw["importancias"] is not None else X_niv.shape[1]
        else:
            def make_pipe(g=grupos):
                prep = criar_preprocessador(g["numericas"], g["categoricas"], g["alta_card"], modo="tree_native", usar_bgnbd=False)
                return Pipeline([("preprocessor", prep), ("classifier", _construir_lgbm_padrao(n_estimators=220))])
            res_unw = rodar_cv_oof(make_pipe, X_niv, y, folds, sample_weight_mode=None)
            n_cols_efetivas = X_niv.shape[1]

        pos = avaliar_posproc_nested_oof(
            res_unw["oof_proba"], y, X_niv, folds, usar_mascara=True, por_regime=False, calibrar=True, metodo_regra="plugin"
        )
        met_arg = res_unw["metricas_argmax"]
        met_pos = pos["metricas_oof"]
        print(
            f"  [{idx}/{total}] {nome_nivel} ({n_cols_efetivas} feats) -> F1_PosProc={met_pos['F1_macro']:.4f} | "
            f"AUC={met_pos['AUC_OvR_macro']:.4f} | LL={met_pos['Log_Loss']:.4f} ({time.time() - t0:.1f}s)",
            flush=True,
        )
        linhas.append({
            "Conjunto_Features": nome_nivel,
            "N_Features_Efetivas": int(n_cols_efetivas),
            "F1_macro_Argmax": met_arg["F1_macro"],
            "F1_macro_PosProc": met_pos["F1_macro"],
            "F1_PosProc_std": float(pos["metricas_por_fold"]["F1_macro"].std()),
            "AUC_OvR_macro": met_pos["AUC_OvR_macro"],
            "Log_Loss": met_pos["Log_Loss"],
            "F1_C0": met_pos["F1_C0"],
            "F1_C1": met_pos["F1_C1"],
            "F1_C2": met_pos["F1_C2"],
            "F1_C3": met_pos["F1_C3"],
        })
    return pd.DataFrame(linhas)


def executar_selecao_atributos_oof(
    X_full: pd.DataFrame,
    y: np.ndarray,
    grupos: Dict[str, List[str]],
    folds: List[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Any]:
    """
    Experimento 2B: seleciona atributos por ganho medio nos 5 folds e compara Todas vs Top 80 vs Top 55.
    """
    t0 = time.time()
    print("  [1/3] Avaliando conjunto completo com Early Stopping + extracao de ganhos...", flush=True)
    res_all = rodar_cv_gbdt_early_stopping("lgbm", _params_lgbm_padrao(1200), X_full, y, grupos, folds)
    pos_all = avaliar_posproc_nested_oof(res_all["oof_proba"], y, X_full, folds, usar_mascara=True, calibrar=True, metodo_regra="plugin")
    imp_df = res_all["importancias"].copy()

    # Mapeia features derivadas para colunas de entrada em X_full
    obrigatorias = ["n_dias_com_compra", "antiguidade_dias", "recencia_dias", "ciclo_medio_dias", "variacao_ciclo", "visitas_90d", "filial"]
    colunas_ranked = []
    for feat in imp_df["Atributo"].tolist():
        # Remove sufixo _0.._3 de TargetEncoder ou prefixo filial_
        cand = feat
        if cand in X_full.columns:
            base_col = cand
        elif re.sub(r"_[0-3]$", "", cand) in X_full.columns:
            base_col = re.sub(r"_[0-3]$", "", cand)
        else:
            continue
        if base_col not in colunas_ranked:
            colunas_ranked.append(base_col)

    for c in X_full.columns:
        if c not in colunas_ranked:
            colunas_ranked.append(c)

    def _filtrar_grupos(k_cols: Optional[int]) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
        if k_cols is None:
            return X_full, grupos
        selecionadas = list(dict.fromkeys(obrigatorias + colunas_ranked[:k_cols]))
        sel_set = set(selecionadas)
        g_sub = {
            "numericas": [c for c in grupos["numericas"] if c in sel_set],
            "categoricas": [c for c in grupos["categoricas"] if c in sel_set],
            "alta_card": [c for c in grupos["alta_card"] if c in sel_set],
        }
        cols_ord = g_sub["numericas"] + g_sub["categoricas"] + g_sub["alta_card"]
        return X_full[cols_ord].copy(), g_sub

    tabela = [{
        "Subconjunto": f"Todas ({X_full.shape[1]} cols base)",
        "N_Cols_Base": X_full.shape[1],
        **pos_all["metricas_oof"],
        "F1_std": float(pos_all["metricas_por_fold"]["F1_macro"].std()),
    }]
    opcoes = {"Todas": (None, pos_all["metricas_oof"]["AUC_OvR_macro"], pos_all["metricas_oof"]["F1_macro"])}

    for idx, k in enumerate([80, 55], start=2):
        t_k = time.time()
        X_k, grp_k = _filtrar_grupos(k)
        res_k = rodar_cv_gbdt_early_stopping("lgbm", _params_lgbm_padrao(1200), X_k, y, grp_k, folds)
        pos_k = avaliar_posproc_nested_oof(res_k["oof_proba"], y, X_k, folds, usar_mascara=True, calibrar=True, metodo_regra="plugin")
        met_k = pos_k["metricas_oof"]
        print(f"  [{idx}/3] Top {k} cols -> F1_PosProc={met_k['F1_macro']:.4f} | AUC={met_k['AUC_OvR_macro']:.4f} | LL={met_k['Log_Loss']:.4f} ({time.time() - t_k:.1f}s)", flush=True)
        tabela.append({
            "Subconjunto": f"Top {X_k.shape[1]} cols base",
            "N_Cols_Base": X_k.shape[1],
            **met_k,
            "F1_std": float(pos_k["metricas_por_fold"]["F1_macro"].std()),
        })
        opcoes[f"Top {k}"] = (k, met_k["AUC_OvR_macro"], met_k["F1_macro"])

    # Escolhe o subconjunto com melhor combinacao F1_macro + 0.2*AUC (favorece menos ruido se houver ganho)
    melhor_chave = max(opcoes.keys(), key=lambda c: opcoes[c][2] + 0.1 * opcoes[c][1])
    k_escolhido = opcoes[melhor_chave][0]
    X_melhor, grp_melhor = _filtrar_grupos(k_escolhido)
    print(f"  -> Subconjunto selecionado: {melhor_chave} ({X_melhor.shape[1]} cols base) em {time.time() - t0:.1f}s", flush=True)

    return {
        "tabela": pd.DataFrame(tabela),
        "melhor_chave": melhor_chave,
        "colunas_selecionadas": list(X_melhor.columns),
        "grupos_selecionados": grp_melhor,
        "importancias_completas": imp_df,
    }


def executar_comparacao_desbalanceamento(
    X: pd.DataFrame,
    y: np.ndarray,
    grupos: Dict[str, List[str]],
    folds: List[Tuple[np.ndarray, np.ndarray]],
) -> pd.DataFrame:
    """Experimento 3: compara Sem balanceamento, class_weight, SMOTE, Bayes (+ Powell) e Plug-in Expected F1."""
    num_cols = grupos["numericas"]
    cat_cols = grupos["categoricas"]
    alta_card = grupos["alta_card"]

    def make_lgbm_native():
        return Pipeline([
            ("preprocessor", criar_preprocessador(num_cols, cat_cols, alta_card, modo="tree_native")),
            ("classifier", _construir_lgbm_padrao(n_estimators=260)),
        ])

    linhas = []

    t0 = time.time()
    res_none = rodar_cv_oof(make_lgbm_native, X, y, folds, sample_weight_mode=None)
    linhas.append({
        "Estrategia": "1. Sem balanceamento (argmax padrao)",
        **res_none["metricas_argmax"],
        "F1_macro_std": float(res_none["metricas_folds_argmax"]["F1_macro"].std()),
    })
    print(f"  [1/8] Sem balanceamento -> F1={res_none['metricas_argmax']['F1_macro']:.4f} ({time.time() - t0:.1f}s)", flush=True)

    t0 = time.time()
    res_bal = rodar_cv_oof(make_lgbm_native, X, y, folds, sample_weight_mode="balanced")
    linhas.append({
        "Estrategia": "2. class_weight='balanced' (argmax)",
        **res_bal["metricas_argmax"],
        "F1_macro_std": float(res_bal["metricas_folds_argmax"]["F1_macro"].std()),
    })
    print(f"  [2/8] class_weight='balanced' -> F1={res_bal['metricas_argmax']['F1_macro']:.4f} ({time.time() - t0:.1f}s)", flush=True)

    t0 = time.time()
    res_suave = rodar_cv_oof(make_lgbm_native, X, y, folds, sample_weight_mode="suave", power_alpha=0.5)
    linhas.append({
        "Estrategia": "3. Peso suave (alpha=0.5, argmax)",
        **res_suave["metricas_argmax"],
        "F1_macro_std": float(res_suave["metricas_folds_argmax"]["F1_macro"].std()),
    })
    print(f"  [3/8] Peso suave (0.5) -> F1={res_suave['metricas_argmax']['F1_macro']:.4f} ({time.time() - t0:.1f}s)", flush=True)

    t0 = time.time()
    def make_smote_lgbm():
        return ImbPipeline([
            ("preprocessor", criar_preprocessador(num_cols, cat_cols, alta_card, modo="tree_imputed")),
            ("smote", SMOTE(sampling_strategy={1: 10000, 2: 5000}, random_state=RANDOM_STATE_DEFAULT)),
            ("classifier", _construir_lgbm_padrao(n_estimators=180)),
        ])

    res_smote = rodar_cv_oof(make_smote_lgbm, X, y, folds, sample_weight_mode=None)
    linhas.append({
        "Estrategia": "4. SMOTE no fold de treino (argmax)",
        **res_smote["metricas_argmax"],
        "F1_macro_std": float(res_smote["metricas_folds_argmax"]["F1_macro"].std()),
    })
    print(f"  [4/8] SMOTE no fold -> F1={res_smote['metricas_argmax']['F1_macro']:.4f} ({time.time() - t0:.1f}s)", flush=True)

    t0 = time.time()
    pos_nomask = avaliar_posproc_nested_oof(
        res_none["oof_proba"], y, X, folds, usar_mascara=False, por_regime=False, calibrar=False, usar_powell=False, metodo_regra="bayes"
    )
    linhas.append({
        "Estrategia": "5. Regra Otima de Bayes Global (Nested OOF)",
        **pos_nomask["metricas_oof"],
        "F1_macro_std": float(pos_nomask["metricas_por_fold"]["F1_macro"].std()),
    })
    print(f"  [5/8] Bayes Global -> F1={pos_nomask['metricas_oof']['F1_macro']:.4f} ({time.time() - t0:.1f}s)", flush=True)

    t0 = time.time()
    pos_cal = avaliar_posproc_nested_oof(
        res_none["oof_proba"], y, X, folds, usar_mascara=True, por_regime=False, calibrar=True, usar_powell=False, metodo_regra="bayes"
    )
    linhas.append({
        "Estrategia": "6. Calibracao Logit + Mascara + Bayes Global (Nested OOF)",
        **pos_cal["metricas_oof"],
        "F1_macro_std": float(pos_cal["metricas_por_fold"]["F1_macro"].std()),
    })
    print(f"  [6/8] Calib + Mascara + Bayes -> F1={pos_cal['metricas_oof']['F1_macro']:.4f} ({time.time() - t0:.1f}s)", flush=True)

    t0 = time.time()
    pos_pow = avaliar_posproc_nested_oof(
        res_none["oof_proba"], y, X, folds, usar_mascara=True, por_regime=False, calibrar=True, usar_powell=True, metodo_regra="bayes_powell"
    )
    linhas.append({
        "Estrategia": "7. Calibracao Logit + Mascara + Bayes + Powell (Nested OOF)",
        **pos_pow["metricas_oof"],
        "F1_macro_std": float(pos_pow["metricas_por_fold"]["F1_macro"].std()),
    })
    print(f"  [7/8] Calib + Mascara + Bayes + Powell -> F1={pos_pow['metricas_oof']['F1_macro']:.4f} ({time.time() - t0:.1f}s)", flush=True)

    t0 = time.time()
    pos_plug = avaliar_posproc_nested_oof(
        res_none["oof_proba"], y, X, folds, usar_mascara=True, por_regime=False, calibrar=True, usar_powell=False, metodo_regra="plugin"
    )
    linhas.append({
        "Estrategia": "8. Calibracao Logit + Mascara + Plug-in Expected Macro-F1 (0 rotulos)",
        **pos_plug["metricas_oof"],
        "F1_macro_std": float(pos_plug["metricas_por_fold"]["F1_macro"].std()),
    })
    print(f"  [8/8] Plug-in Expected Macro-F1 -> F1={pos_plug['metricas_oof']['F1_macro']:.4f} ({time.time() - t0:.1f}s)", flush=True)

    return pd.DataFrame(linhas)


def executar_comparacao_modelos(
    X: pd.DataFrame,
    y: np.ndarray,
    grupos: Dict[str, List[str]],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    X_test: Optional[pd.DataFrame] = None,
    cache_dir: Optional[str | Path] = None,
    usar_cache: bool = True,
) -> Dict[str, Any]:
    """Experimento 4: compara ate 11 modelos (classicos, GAM-Spline, Early Stopping GBDTs e CatBoost)."""
    num_cols = grupos["numericas"]
    cat_cols = grupos["categoricas"]
    alta_card = grupos["alta_card"]

    definicoes: List[Tuple[str, str, Any]] = [
        (
            "Logistic Regression",
            "pipe:linear",
            lambda: LogisticRegression(max_iter=250, C=0.5, random_state=RANDOM_STATE_DEFAULT),
        ),
        (
            "Spline-LogReg (GAM)",
            "pipe:gam_spline",
            lambda: LogisticRegression(max_iter=250, C=0.3, random_state=RANDOM_STATE_DEFAULT),
        ),
        (
            "Decision Tree",
            "pipe:tree_imputed",
            lambda: DecisionTreeClassifier(max_depth=10, min_samples_leaf=50, random_state=RANDOM_STATE_DEFAULT),
        ),
        (
            "Linear SVM (SGD Calibrated)",
            "pipe:linear",
            lambda: CalibratedClassifierCV(
                SGDClassifier(loss="hinge", alpha=1e-3, max_iter=60, random_state=RANDOM_STATE_DEFAULT),
                cv=3,
                method="sigmoid",
            ),
        ),
        (
            "MLP (Neural Network)",
            "pipe:linear",
            lambda: MLPClassifier(
                hidden_layer_sizes=(64, 32),
                alpha=1e-2,
                early_stopping=True,
                max_iter=40,
                n_iter_no_change=5,
                random_state=RANDOM_STATE_DEFAULT,
            ),
        ),
        (
            "Random Forest",
            "pipe:tree_imputed",
            lambda: RandomForestClassifier(
                n_estimators=160, max_depth=16, min_samples_leaf=8, random_state=RANDOM_STATE_DEFAULT, n_jobs=N_JOBS_DEFAULT
            ),
        ),
        (
            "Extra Trees",
            "pipe:tree_imputed",
            lambda: ExtraTreesClassifier(
                n_estimators=160, max_depth=18, min_samples_leaf=8, random_state=RANDOM_STATE_DEFAULT, n_jobs=N_JOBS_DEFAULT
            ),
        ),
        (
            "HistGradientBoosting",
            "pipe:tree_native",
            lambda: HistGradientBoostingClassifier(
                max_iter=400, learning_rate=0.04, max_leaf_nodes=23, min_samples_leaf=80,
                l2_regularization=5.0, early_stopping=True, random_state=RANDOM_STATE_DEFAULT
            ),
        ),
        ("XGBoost (Early Stopping)", "es:xgb", _params_xgb_padrao(1200)),
        ("LightGBM (Early Stopping)", "es:lgbm", _params_lgbm_padrao(1200)),
    ]

    if HAS_CATBOOST:
        definicoes.append((
            "CatBoost (Bonus, Early Stopping)",
            "es:catboost",
            {
                "iterations": 1200,
                "learning_rate": 0.04,
                "depth": 5,
                "l2_leaf_reg": 10.0,
                "loss_function": "MultiClass",
                "eval_metric": "MultiClass",
                "random_seed": RANDOM_STATE_DEFAULT,
                "thread_count": N_JOBS_DEFAULT,
            },
        ))

    tabela_resumo = []
    oof_probas: Dict[str, np.ndarray] = {}
    test_probas: Dict[str, Optional[np.ndarray]] = {}
    posproc_detalhes: Dict[str, Dict[str, Any]] = {}
    folds_f1_posproc: Dict[str, np.ndarray] = {}

    total = len(definicoes)
    for idx, (nome, tipo, spec) in enumerate(definicoes, start=1):
        t0 = time.time()

        def _rodar_modelo(tp=tipo, sp=spec):
            if tp.startswith("pipe:"):
                modo_prep = tp.split(":", 1)[1]
                def make_pipe():
                    prep = criar_preprocessador(num_cols, cat_cols, alta_card, modo=modo_prep)
                    return Pipeline([("preprocessor", prep), ("classifier", sp())])
                res_m = rodar_cv_oof(make_pipe, X, y, folds, sample_weight_mode=None, X_test=X_test)
            else:
                fam = tp.split(":", 1)[1]
                res_m = rodar_cv_gbdt_early_stopping(fam, sp, X, y, grupos, folds, X_test=X_test)

            pos_m = avaliar_posproc_nested_oof(
                res_m["oof_proba"], y, X, folds, usar_mascara=True, por_regime=False, calibrar=True, metodo_regra="plugin"
            )
            return {"res": res_m, "pos": pos_m}

        if cache_dir is not None:
            slug = re.sub(r"[^a-z0-9]+", "_", nome.lower()).strip("_")
            pacote = carregar_ou_executar(f"v3_exp4_modelo_{slug}.joblib", _rodar_modelo, cache_dir=cache_dir, usar_cache=usar_cache)
        else:
            pacote = _rodar_modelo()

        res = pacote["res"]
        pos = pacote["pos"]
        oof_probas[nome] = res["oof_proba"]
        test_probas[nome] = res["test_proba"]
        posproc_detalhes[nome] = pos
        folds_f1_posproc[nome] = pos["metricas_por_fold"]["F1_macro"].to_numpy()

        met_arg = res["metricas_argmax"]
        met_pos = pos["metricas_oof"]
        print(
            f"  [{idx}/{total}] {nome:32s} -> F1_PosProc={met_pos['F1_macro']:.4f} | "
            f"AUC={met_pos['AUC_OvR_macro']:.4f} | LL={met_pos['Log_Loss']:.4f} ({time.time() - t0:.1f}s)",
            flush=True,
        )
        tabela_resumo.append({
            "Modelo": nome,
            "F1_macro_Argmax": met_arg["F1_macro"],
            "F1_macro_PosProc": met_pos["F1_macro"],
            "F1_std_folds": float(pos["metricas_por_fold"]["F1_macro"].std()),
            "AUC_OvR_macro": met_pos["AUC_OvR_macro"],
            "AUC_std_folds": float(pos["metricas_por_fold"]["AUC_OvR_macro"].std()),
            "Log_Loss": met_pos["Log_Loss"],
            "Balanced_Acc": met_pos["Balanced_Acc"],
            "Accuracy": met_pos["Accuracy"],
            "F1_C0": met_pos["F1_C0"],
            "F1_C1": met_pos["F1_C1"],
            "F1_C2": met_pos["F1_C2"],
            "F1_C3": met_pos["F1_C3"],
        })

    df_resumo = pd.DataFrame(tabela_resumo).sort_values("F1_macro_PosProc", ascending=False).reset_index(drop=True)
    return {
        "tabela": df_resumo,
        "oof_probas": oof_probas,
        "test_probas": test_probas,
        "posproc": posproc_detalhes,
        "folds_f1": pd.DataFrame(folds_f1_posproc),
    }


def _rodar_especialista_regime(
    familia: str,
    params: Dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    grupos: Dict[str, List[str]],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    X_test: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Treina modelos especialistas separados para Regime A (1 dia -> classes {0, 1, 3})
    e Regime B (2+ dias -> classes {0, 1, 2, 3}), usando Early Stopping e sem colunas 100% NaN no Regime A.
    """
    is_single = (X["n_dias_com_compra"].to_numpy() == 1)
    is_single_te = (X_test["n_dias_com_compra"].to_numpy() == 1) if X_test is not None else None

    # Remove colunas 100% NaN no subconjunto de 1 dia de compra
    num_validas_single = [c for c in grupos["numericas"] if X.loc[is_single, c].notna().any()]
    grp_single = {
        "numericas": num_validas_single,
        "categoricas": grupos["categoricas"],
        "alta_card": grupos["alta_card"],
    }

    idx_single = np.where(is_single)[0]
    idx_multi = np.where(~is_single)[0]

    # Mapear folds externos para cada regime mantendo exatamente a mesma divisao OOF
    folds_single = []
    folds_multi = []
    pos_map_s = {orig_i: k for k, orig_i in enumerate(idx_single)}
    pos_map_m = {orig_i: k for k, orig_i in enumerate(idx_multi)}
    for tr_idx, va_idx in folds:
        folds_single.append((
            np.array([pos_map_s[i] for i in tr_idx if i in pos_map_s], dtype=int),
            np.array([pos_map_s[i] for i in va_idx if i in pos_map_s], dtype=int),
        ))
        folds_multi.append((
            np.array([pos_map_m[i] for i in tr_idx if i in pos_map_m], dtype=int),
            np.array([pos_map_m[i] for i in va_idx if i in pos_map_m], dtype=int),
        ))

    # Regime A: mapear {0: 0, 1: 1, 3: 2}
    y_single_raw = y[idx_single]
    y_single_3 = np.where(y_single_raw == 3, 2, y_single_raw)
    X_single = X.iloc[idx_single].reset_index(drop=True)
    X_te_single = X_test.iloc[is_single_te].reset_index(drop=True) if X_test is not None else None

    res_s = rodar_cv_gbdt_early_stopping(
        familia, params, X_single, y_single_3, grp_single, folds_single, X_test=X_te_single, n_classes_sub=[0, 1, 2]
    )

    # Regime B: 4 classes completas
    y_multi = y[idx_multi]
    X_multi = X.iloc[idx_multi].reset_index(drop=True)
    X_te_multi = X_test.iloc[~is_single_te].reset_index(drop=True) if X_test is not None else None

    res_m = rodar_cv_gbdt_early_stopping(
        familia, params, X_multi, y_multi, grupos, folds_multi, X_test=X_te_multi, n_classes_sub=[0, 1, 2, 3]
    )

    # Recompor matrizes (N, 4)
    oof_proba = np.zeros((len(X), 4), dtype=np.float64)
    oof_proba[idx_single[:, None], [0, 1, 3]] = res_s["oof_proba"]
    oof_proba[idx_multi] = res_m["oof_proba"]

    test_proba = None
    if X_test is not None and is_single_te is not None:
        test_proba = np.zeros((len(X_test), 4), dtype=np.float64)
        idx_te_s = np.where(is_single_te)[0]
        idx_te_m = np.where(~is_single_te)[0]
        test_proba[idx_te_s[:, None], [0, 1, 3]] = res_s["test_proba"]
        test_proba[idx_te_m] = res_m["test_proba"]

    metricas_folds = [calcular_metricas_completas(y[va], oof_proba[va]) for _, va in folds]
    return {
        "oof_proba": oof_proba,
        "test_proba": test_proba,
        "metricas_argmax": calcular_metricas_completas(y, oof_proba),
        "metricas_folds_argmax": pd.DataFrame(metricas_folds),
    }


def _fit_predict_binary_stage(
    familia: str,
    params_stage: Dict[str, Any],
    X_st_tr: pd.DataFrame,
    y_st_tr: np.ndarray,
    grupos: Dict[str, List[str]],
    X_st_va: pd.DataFrame,
    X_st_te: Optional[pd.DataFrame],
    seed: int,
    early_stopping_rounds: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Treina um classificador binario com Early Stopping interno (88/12) e TargetEncoder/BG-NBD sem leakage."""
    y_st_tr = np.asarray(y_st_tr, dtype=int)
    sub_i, inn_i = train_test_split(
        np.arange(len(y_st_tr)), test_size=0.12, stratify=y_st_tr, random_state=seed
    )
    prep = criar_preprocessador(
        grupos["numericas"], grupos["categoricas"], grupos["alta_card"], modo="tree_native", target_type="binary"
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_sub = prep.fit_transform(X_st_tr.iloc[sub_i], y_st_tr[sub_i])
        X_inn = prep.transform(X_st_tr.iloc[inn_i])
        X_va_t = prep.transform(X_st_va) if len(X_st_va) > 0 else None
        X_te_t = prep.transform(X_st_te) if (X_st_te is not None and len(X_st_te) > 0) else None

    if familia == "lgbm":
        lgbm_p = dict(params_stage)
        lgbm_p["random_state"] = seed
        clf = LGBMClassifier(**lgbm_p)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clf.fit(
                X_sub, y_st_tr[sub_i],
                eval_set=[(X_inn, y_st_tr[inn_i])],
                callbacks=[lgb.early_stopping(early_stopping_rounds, verbose=False)],
            )
    elif familia == "xgb":
        xgb_p = dict(params_stage)
        xgb_p["random_state"] = seed
        xgb_p["early_stopping_rounds"] = early_stopping_rounds
        clf = XGBClassifier(**xgb_p)
        clf.fit(X_sub, y_st_tr[sub_i], eval_set=[(X_inn, y_st_tr[inn_i])], verbose=False)
    elif familia == "catboost":
        if not HAS_CATBOOST:
            raise RuntimeError("CatBoost nao instalado.")
        cb_p = dict(params_stage)
        cb_p["random_seed"] = seed
        cb_p["allow_writing_files"] = False
        clf = CatBoostClassifier(**cb_p)
        clf.fit(X_sub, y_st_tr[sub_i], eval_set=(X_inn, y_st_tr[inn_i]), early_stopping_rounds=early_stopping_rounds, verbose=False)
    else:
        raise ValueError(f"Familia nao suportada em estagio binario: {familia}")

    p_va = clf.predict_proba(X_va_t)[:, 1] if X_va_t is not None else np.empty(0, dtype=np.float64)
    p_te = clf.predict_proba(X_te_t)[:, 1] if X_te_t is not None else (np.empty(0, dtype=np.float64) if X_st_te is not None else None)
    return p_va, p_te


def _rodar_hierarquico_3_estagios(
    familia: str,
    params: Dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    grupos: Dict[str, List[str]],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    X_test: Optional[pd.DataFrame] = None,
    seeds: Tuple[int, ...] = (RANDOM_STATE_DEFAULT,),
) -> Dict[str, Any]:
    """
    Decomposicao Hierarquica em 3 estagios alinhada ao processo gerador:
      1) Recompra vs Churn: P(y != 3 | x) em 100% da base;
      2) Atraso vs No Prazo | Recompra: P(y in {1,2} | y != 3, x);
      3) Dormente vs Desacelerando | Atraso: P(y == 2 | y in {1,2}, n_dias>=2, ciclo<=46.5).
    Suporta Bagging Multi-Seed nos mesmos folds externos (zero leakage).
    """
    y = np.asarray(y, dtype=int)
    ciclo_tr = X["ciclo_medio_dias"].to_numpy(dtype=np.float64)
    elig_tr = (X["n_dias_com_compra"].to_numpy() >= 2) & np.isfinite(ciclo_tr) & (ciclo_tr <= 46.5)

    elig_te = None
    if X_test is not None:
        ciclo_te = X_test["ciclo_medio_dias"].to_numpy(dtype=np.float64)
        elig_te = (X_test["n_dias_com_compra"].to_numpy() >= 2) & np.isfinite(ciclo_te) & (ciclo_te <= 46.5)

    params_s1 = dict(params)
    params_s2 = dict(params)
    params_s3 = dict(params)
    if familia == "lgbm":
        params_s1["objective"] = "binary"
        params_s2["objective"] = "binary"
        params_s2["min_child_samples"] = min(int(params_s2.get("min_child_samples", 80)), 80)
        params_s3["objective"] = "binary"
        params_s3["min_child_samples"] = min(int(params_s3.get("min_child_samples", 40)), 40)
        params_s3["num_leaves"] = min(int(params_s3.get("num_leaves", 15)), 15)
    elif familia == "xgb":
        params_s1.update({"objective": "binary:logistic", "eval_metric": "logloss"})
        params_s2.update({"objective": "binary:logistic", "eval_metric": "logloss", "min_child_weight": min(int(params_s2.get("min_child_weight", 40)), 40)})
        params_s3.update({"objective": "binary:logistic", "eval_metric": "logloss", "min_child_weight": min(int(params_s3.get("min_child_weight", 15)), 15)})
    elif familia == "catboost":
        params_s1.update({"loss_function": "Logloss", "eval_metric": "Logloss", "allow_writing_files": False})
        params_s2.update({"loss_function": "Logloss", "eval_metric": "Logloss", "allow_writing_files": False})
        params_s3.update({
            "loss_function": "Logloss",
            "eval_metric": "Logloss",
            "allow_writing_files": False,
            "depth": min(int(params_s3.get("depth", 4)), 4),
            "l2_leaf_reg": max(float(params_s3.get("l2_leaf_reg", 12.0)), 12.0),
        })

    oof_proba = np.zeros((len(X), 4), dtype=np.float64)
    test_proba = np.zeros((len(X_test), 4), dtype=np.float64) if X_test is not None else None
    X_te_elig = X_test.iloc[elig_te] if (X_test is not None and elig_te is not None) else None

    for seed_val in seeds:
        oof_seed = np.zeros((len(X), 4), dtype=np.float64)
        test_seed = np.zeros((len(X_test), 4), dtype=np.float64) if X_test is not None else None

        for fold_id, (tr_idx, va_idx) in enumerate(folds):
            seed_base = int(seed_val) + fold_id * 10
            X_va_fold = X.iloc[va_idx]

            # Estagio 1: Recompra (y != 3) vs Churn (y == 3)
            p_rec_va, p_rec_te = _fit_predict_binary_stage(
                familia, params_s1, X.iloc[tr_idx], (y[tr_idx] != 3).astype(int), grupos, X_va_fold, X_test, seed_base + 1, 40
            )

            # Estagio 2: Atrasado ({1,2}) vs Ativo ({0}) entre quem recomprou
            tr_rec = tr_idx[y[tr_idx] != 3]
            p_del_va, p_del_te = _fit_predict_binary_stage(
                familia, params_s2, X.iloc[tr_rec], np.isin(y[tr_rec], [1, 2]).astype(int), grupos, X_va_fold, X_test, seed_base + 2, 40
            )

            # Estagio 3: Dormente (2) vs Desacelerando (1) na regiao estruturalmente elegivel
            tr_s3 = tr_idx[np.isin(y[tr_idx], [1, 2]) & elig_tr[tr_idx]]
            va_elig_mask = elig_tr[va_idx]
            p_dorm_elig_va, p_dorm_elig_te = _fit_predict_binary_stage(
                familia, params_s3, X.iloc[tr_s3], (y[tr_s3] == 2).astype(int), grupos, X_va_fold.iloc[va_elig_mask], X_te_elig, seed_base + 3, 30
            )

            p_dorm_va = np.zeros(len(va_idx), dtype=np.float64)
            p_dorm_va[va_elig_mask] = p_dorm_elig_va

            oof_seed[va_idx] = np.column_stack([
                p_rec_va * (1.0 - p_del_va),
                p_rec_va * p_del_va * (1.0 - p_dorm_va),
                p_rec_va * p_del_va * p_dorm_va,
                1.0 - p_rec_va,
            ])

            if test_seed is not None and X_test is not None and elig_te is not None:
                p_dorm_te = np.zeros(len(X_test), dtype=np.float64)
                if p_dorm_elig_te is not None and len(p_dorm_elig_te) > 0:
                    p_dorm_te[elig_te] = p_dorm_elig_te
                p_te_fold = np.column_stack([
                    p_rec_te * (1.0 - p_del_te),
                    p_rec_te * p_del_te * (1.0 - p_dorm_te),
                    p_rec_te * p_del_te * p_dorm_te,
                    1.0 - p_rec_te,
                ])
                test_seed += p_te_fold / len(folds)

        oof_proba += oof_seed / len(seeds)
        if test_proba is not None and test_seed is not None:
            test_proba += test_seed / len(seeds)

    metricas_folds = [calcular_metricas_completas(y[va], oof_proba[va]) for _, va in folds]
    return {
        "oof_proba": oof_proba,
        "test_proba": test_proba,
        "metricas_argmax": calcular_metricas_completas(y, oof_proba),
        "metricas_folds_argmax": pd.DataFrame(metricas_folds),
    }


def executar_tuning_e_especialistas(
    X: pd.DataFrame,
    y: np.ndarray,
    grupos: Dict[str, List[str]],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    X_test: Optional[pd.DataFrame] = None,
    n_iter: int = 6,
    cache_dir: Optional[str | Path] = None,
    usar_cache: bool = True,
) -> Dict[str, Any]:
    """
    Experimento 5: Tuning regularizado via RandomizedSearchCV estratificado + avaliacao 5-fold
    com Early Stopping + Especialistas por Regime (1-Dia vs 2+Dias) + Decomposicao Hierarquica 3-Estagios.
    """
    num_cols = grupos["numericas"]
    cat_cols = grupos["categoricas"]
    alta_card = grupos["alta_card"]
    cv_search = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE_DEFAULT)

    def _buscar_params():
        t0 = time.time()
        print(f"  [1/8] RandomizedSearchCV LightGBM regularizado (n_iter={n_iter}, cv=3)...", flush=True)
        pipe_lgbm = Pipeline([
            ("preprocessor", criar_preprocessador(num_cols, cat_cols, alta_card, modo="tree_native")),
            ("classifier", LGBMClassifier(n_estimators=400, random_state=RANDOM_STATE_DEFAULT, n_jobs=N_JOBS_DEFAULT, verbose=-1)),
        ])
        grid_lgbm = {
            "classifier__learning_rate": [0.025, 0.035],
            "classifier__max_depth": [4, 5, 6],
            "classifier__num_leaves": [15, 20, 31],
            "classifier__min_child_samples": [80, 120, 180],
            "classifier__subsample": [0.75, 0.85],
            "classifier__subsample_freq": [1],
            "classifier__colsample_bytree": [0.5, 0.65, 0.8],
            "classifier__reg_alpha": [1.0, 2.5, 5.0],
            "classifier__reg_lambda": [5.0, 10.0, 15.0],
        }
        s_lgbm = RandomizedSearchCV(
            pipe_lgbm, grid_lgbm, n_iter=n_iter, scoring="neg_log_loss", cv=cv_search,
            random_state=RANDOM_STATE_DEFAULT, n_jobs=1, refit=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s_lgbm.fit(X, y)
        best_l = {k.replace("classifier__", ""): v for k, v in s_lgbm.best_params_.items()}
        best_l.update({"n_estimators": 1400, "random_state": RANDOM_STATE_DEFAULT, "n_jobs": N_JOBS_DEFAULT, "verbose": -1})
        print(f"    -> Concluido ({time.time() - t0:.1f}s) | best LGBM={best_l}", flush=True)

        t0 = time.time()
        print(f"  [2/8] RandomizedSearchCV XGBoost regularizado (n_iter={n_iter}, cv=3)...", flush=True)
        pipe_xgb = Pipeline([
            ("preprocessor", criar_preprocessador(num_cols, cat_cols, alta_card, modo="tree_native")),
            ("classifier", XGBClassifier(n_estimators=400, tree_method="hist", eval_metric="mlogloss", random_state=RANDOM_STATE_DEFAULT, n_jobs=N_JOBS_DEFAULT)),
        ])
        grid_xgb = {
            "classifier__learning_rate": [0.025, 0.035],
            "classifier__max_depth": [4, 5],
            "classifier__min_child_weight": [30, 60, 100],
            "classifier__subsample": [0.75, 0.85],
            "classifier__colsample_bytree": [0.5, 0.65, 0.8],
            "classifier__reg_alpha": [1.0, 2.5, 5.0],
            "classifier__reg_lambda": [5.0, 10.0, 15.0],
            "classifier__gamma": [0.1, 0.3],
        }
        s_xgb = RandomizedSearchCV(
            pipe_xgb, grid_xgb, n_iter=n_iter, scoring="neg_log_loss", cv=cv_search,
            random_state=RANDOM_STATE_DEFAULT, n_jobs=1, refit=False,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            s_xgb.fit(X, y)
        best_x = {k.replace("classifier__", ""): v for k, v in s_xgb.best_params_.items()}
        best_x.update({"n_estimators": 1400, "tree_method": "hist", "eval_metric": "mlogloss", "random_state": RANDOM_STATE_DEFAULT, "n_jobs": N_JOBS_DEFAULT})
        print(f"    -> Concluido ({time.time() - t0:.1f}s) | best XGB={best_x}", flush=True)
        return {"best_lgbm": best_l, "best_xgb": best_x}

    if cache_dir is not None:
        best_p = carregar_ou_executar("v3_exp5_best_params.joblib", _buscar_params, cache_dir=cache_dir, usar_cache=usar_cache)
    else:
        best_p = _buscar_params()

    best_lgbm_params = best_p["best_lgbm"]
    best_xgb_params = best_p["best_xgb"]

    seeds_bagging = (RANDOM_STATE_DEFAULT, RANDOM_STATE_DEFAULT + 60, RANDOM_STATE_DEFAULT + 160)
    catboost_hier_params = {
        "iterations": 1200,
        "learning_rate": 0.04,
        "depth": 5,
        "l2_leaf_reg": 10.0,
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "random_seed": RANDOM_STATE_DEFAULT,
        "thread_count": N_JOBS_DEFAULT,
        "allow_writing_files": False,
    }

    candidatos = [
        ("LightGBM Tunado (ES)", "lgbm", best_lgbm_params, "global", (RANDOM_STATE_DEFAULT,)),
        ("XGBoost Tunado (ES)", "xgb", best_xgb_params, "global", (RANDOM_STATE_DEFAULT,)),
        ("Especialista LightGBM (1-Dia vs 2+Dias)", "lgbm", best_lgbm_params, "especialista", (RANDOM_STATE_DEFAULT,)),
        ("Especialista XGBoost (1-Dia vs 2+Dias)", "xgb", best_xgb_params, "especialista", (RANDOM_STATE_DEFAULT,)),
        ("Hierarquico LightGBM (3-Estagios, 3-Seeds)", "lgbm", best_lgbm_params, "hierarquico", seeds_bagging),
        ("Hierarquico XGBoost (3-Estagios, 3-Seeds)", "xgb", best_xgb_params, "hierarquico", seeds_bagging),
    ]
    if HAS_CATBOOST:
        candidatos.append((
            "Hierarquico CatBoost (3-Estagios, 3-Seeds)", "catboost", catboost_hier_params, "hierarquico", seeds_bagging
        ))

    total_etapas = 2 + len(candidatos)
    linhas = []
    resultados = {}
    for idx, (nome, fam, prm, modo_estr, sds) in enumerate(candidatos, start=3):
        t0 = time.time()

        def _run_cand(f=fam, p=prm, me=modo_estr, sd=sds):
            if me == "especialista":
                res_c = _rodar_especialista_regime(f, p, X, y, grupos, folds, X_test=X_test)
            elif me == "hierarquico":
                res_c = _rodar_hierarquico_3_estagios(f, p, X, y, grupos, folds, X_test=X_test, seeds=sd)
            else:
                res_c = rodar_cv_gbdt_early_stopping(f, p, X, y, grupos, folds, X_test=X_test)
            pos_c = avaliar_posproc_nested_oof(
                res_c["oof_proba"], y, X, folds, usar_mascara=True, por_regime=False, calibrar=True, metodo_regra="plugin"
            )
            return {"cv": res_c, "pos": pos_c}

        if cache_dir is not None:
            slug = re.sub(r"[^a-z0-9]+", "_", nome.lower()).strip("_")
            pac = carregar_ou_executar(f"v3_exp5_{slug}.joblib", _run_cand, cache_dir=cache_dir, usar_cache=usar_cache)
        else:
            pac = _run_cand()

        resultados[nome] = pac
        met = pac["pos"]["metricas_oof"]
        print(
            f"  [{idx}/{total_etapas}] {nome:44s} -> F1_PosProc={met['F1_macro']:.4f} | AUC={met['AUC_OvR_macro']:.4f} | "
            f"LL={met['Log_Loss']:.4f} ({time.time() - t0:.1f}s)",
            flush=True,
        )
        linhas.append({
            "Modelo": nome,
            "F1_macro_Argmax": pac["cv"]["metricas_argmax"]["F1_macro"],
            "F1_macro_PosProc": met["F1_macro"],
            "F1_std_folds": float(pac["pos"]["metricas_por_fold"]["F1_macro"].std()),
            "AUC_OvR_macro": met["AUC_OvR_macro"],
            "Log_Loss": met["Log_Loss"],
            "Balanced_Acc": met["Balanced_Acc"],
            "F1_C0": met["F1_C0"],
            "F1_C1": met["F1_C1"],
            "F1_C2": met["F1_C2"],
            "F1_C3": met["F1_C3"],
        })

    return {
        "best_lgbm_params": best_lgbm_params,
        "best_xgb_params": best_xgb_params,
        "tabela": pd.DataFrame(linhas).sort_values("F1_macro_PosProc", ascending=False).reset_index(drop=True),
        "resultados": resultados,
    }


def executar_blending_ensemble(
    probas_oof_dict: Dict[str, np.ndarray],
    probas_test_dict: Dict[str, Optional[np.ndarray]],
    y: np.ndarray,
    df_features: pd.DataFrame,
    folds: List[Tuple[np.ndarray, np.ndarray]],
) -> Dict[str, Any]:
    """
    Experimento 6 (Bonus): combina probabilidades de todos os modelos diversos comparando
    Blend Linear vs Blend Geometrico (Log-Odds) via SLSQP Nested OOF + Calibracao Logit + Plug-in Expected F1.
    """
    t0 = time.time()
    nomes = list(probas_oof_dict.keys())
    matrizes_oof = [np.asarray(probas_oof_dict[n], dtype=np.float64) for n in nomes]
    log_matrizes_oof = [np.log(np.clip(m, 1e-7, 1.0)) for m in matrizes_oof]
    n_models = len(nomes)
    y = np.asarray(y, dtype=int)

    def _softmax_logits(z: np.ndarray) -> np.ndarray:
        z_s = z - z.max(axis=1, keepdims=True)
        e = np.exp(z_s)
        return e / np.maximum(e.sum(axis=1, keepdims=True), 1e-12)

    blend_lin_oof = np.zeros_like(matrizes_oof[0])
    blend_geo_oof = np.zeros_like(matrizes_oof[0])
    pesos_lin_folds = []
    pesos_geo_folds = []

    bounds = [(0.0, 1.0)] * n_models
    cons = {"type": "eq", "fun": lambda w: float(np.sum(w) - 1.0)}
    w0 = np.ones(n_models, dtype=np.float64) / n_models

    for tr_idx, va_idx in folds:
        y_tr = y[tr_idx]
        idx_rows = np.arange(len(tr_idx))
        mats_lin_tr = [m[tr_idx] for m in matrizes_oof]
        mats_log_tr = [m[tr_idx] for m in log_matrizes_oof]

        def loss_lin(weights):
            w = np.maximum(weights, 0.0)
            s = w.sum()
            w = w / s if s > 0 else w0
            p = sum(w[i] * mats_lin_tr[i] for i in range(n_models))
            p = np.clip(p, 1e-12, 1.0)
            return float(-np.mean(np.log(p[idx_rows, y_tr])))

        def loss_geo(weights):
            w = np.maximum(weights, 0.0)
            s = w.sum()
            w = w / s if s > 0 else w0
            z = sum(w[i] * mats_log_tr[i] for i in range(n_models))
            z = z - z.max(axis=1, keepdims=True)
            lse = np.log(np.sum(np.exp(z), axis=1))
            return float(-np.mean(z[idx_rows, y_tr] - lse))

        res_l = minimize(loss_lin, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"ftol": 1e-9, "maxiter": 200})
        w_l = np.maximum(res_l.x, 0.0)
        w_l = w_l / w_l.sum() if w_l.sum() > 0 else w0
        pesos_lin_folds.append(w_l)
        blend_lin_oof[va_idx] = sum(w_l[i] * matrizes_oof[i][va_idx] for i in range(n_models))

        res_g = minimize(loss_geo, w0, method="SLSQP", bounds=bounds, constraints=cons, options={"ftol": 1e-9, "maxiter": 200})
        w_g = np.maximum(res_g.x, 0.0)
        w_g = w_g / w_g.sum() if w_g.sum() > 0 else w0
        pesos_geo_folds.append(w_g)
        z_va = sum(w_g[i] * log_matrizes_oof[i][va_idx] for i in range(n_models))
        blend_geo_oof[va_idx] = _softmax_logits(z_va)

    pesos_lin_med = np.mean(pesos_lin_folds, axis=0)
    pesos_lin_med /= pesos_lin_med.sum()
    pesos_geo_med = np.mean(pesos_geo_folds, axis=0)
    pesos_geo_med /= pesos_geo_med.sum()

    blend_lin_test = None
    blend_geo_test = None
    if all(probas_test_dict.get(n) is not None for n in nomes):
        blend_lin_test = sum(pesos_lin_med[i] * np.asarray(probas_test_dict[nomes[i]], dtype=np.float64) for i in range(n_models))
        z_te = sum(pesos_geo_med[i] * np.log(np.clip(probas_test_dict[nomes[i]], 1e-7, 1.0)) for i in range(n_models))
        blend_geo_test = _softmax_logits(z_te)

    pos_lin = avaliar_posproc_nested_oof(
        blend_lin_oof, y, df_features, folds, usar_mascara=True, por_regime=False, calibrar=True, metodo_regra="plugin"
    )
    pos_geo = avaliar_posproc_nested_oof(
        blend_geo_oof, y, df_features, folds, usar_mascara=True, por_regime=False, calibrar=True, metodo_regra="plugin"
    )

    ll_lin_antes = calcular_metricas_completas(y, blend_lin_oof)["Log_Loss"]
    ll_geo_antes = calcular_metricas_completas(y, blend_geo_oof)["Log_Loss"]

    linhas_bl = []
    opcoes = {
        "Linear": (blend_lin_oof, blend_lin_test, pesos_lin_med, pos_lin, ll_lin_antes),
        "Geometrico (Log-Odds)": (blend_geo_oof, blend_geo_test, pesos_geo_med, pos_geo, ll_geo_antes),
    }
    for tipo_b, (_, _, _, pos_b, ll_b) in opcoes.items():
        met_b = pos_b["metricas_oof"]
        linhas_bl.append({
            "Tipo_Blend": tipo_b,
            "N_Modelos": n_models,
            "Log_Loss_Bruto": ll_b,
            "Log_Loss_Calib": met_b["Log_Loss"],
            "AUC_OvR_macro": met_b["AUC_OvR_macro"],
            "F1_macro_PosProc": met_b["F1_macro"],
            "F1_std_folds": float(pos_b["metricas_por_fold"]["F1_macro"].std()),
            "F1_C0": met_b["F1_C0"],
            "F1_C1": met_b["F1_C1"],
            "F1_C2": met_b["F1_C2"],
            "F1_C3": met_b["F1_C3"],
        })
    tabela_blends = pd.DataFrame(linhas_bl).sort_values(["F1_macro_PosProc", "Log_Loss_Calib"], ascending=[False, True]).reset_index(drop=True)

    # Seleciona vencedor por maior F1-macro Nested OOF (desempate por menor Log-Loss)
    vencedor = max(
        opcoes.keys(),
        key=lambda k: (round(opcoes[k][3]["metricas_oof"]["F1_macro"], 6), -opcoes[k][3]["metricas_oof"]["Log_Loss"], 1 if k.startswith("Geometrico") else 0),
    )
    blend_oof, blend_test, pesos_medios, pos, ll_antes = opcoes[vencedor]
    ll_depois = pos["metricas_oof"]["Log_Loss"]

    print(
        f"  Blend Linear -> F1={pos_lin['metricas_oof']['F1_macro']:.4f} | LL={pos_lin['metricas_oof']['Log_Loss']:.4f} | "
        f"Blend Log-Odds -> F1={pos_geo['metricas_oof']['F1_macro']:.4f} | LL={pos_geo['metricas_oof']['Log_Loss']:.4f} "
        f"=> Selecionado: {vencedor} ({time.time() - t0:.1f}s)",
        flush=True,
    )
    return {
        "modelos": nomes,
        "tipo_blend_escolhido": vencedor,
        "tabela_blends": tabela_blends,
        "pesos_medios": dict(zip(nomes, np.round(pesos_medios, 4).tolist())),
        "blend_oof": blend_oof,
        "blend_test": blend_test,
        "log_loss_antes_calibracao": ll_antes,
        "log_loss_depois_calibracao": ll_depois,
        "posproc": pos,
        "posproc_linear": pos_lin,
        "posproc_logodds": pos_geo,
    }


def gerar_e_validar_submissoes(
    test_df: pd.DataFrame,
    sample_sub: pd.DataFrame,
    y_train: np.ndarray,
    proba_test_ens: np.ndarray,
    pos_ens: Dict[str, Any],
    proba_test_ind: np.ndarray,
    pos_ind: Dict[str, Any],
    X_test_feat: pd.DataFrame,
    out_dir: str | Path = "outputs",
) -> Dict[str, Any]:
    """Gera e valida arquivos finais de submissao aplicando calibracao, mascara e regra Bayes."""
    pasta = Path(out_dir)
    pasta.mkdir(parents=True, exist_ok=True)
    is_single_test = (X_test_feat["n_dias_com_compra"].to_numpy() == 1)

    def aplicar_e_salvar(proba_te, pos_dict, filename):
        p = proba_te.copy()
        if pos_dict.get("calibrar", False):
            p = aplicar_calibracao_logit(p, pos_dict["inv_T_final"], pos_dict["bias_final"])
        if pos_dict.get("usar_mascara", True):
            p = aplicar_mascara_estrutural(p, X_test_feat)

        if pos_dict.get("por_regime", False):
            pred_te = np.empty(len(p), dtype=int)
            if np.any(is_single_test):
                pred_te[is_single_test] = predizer_regra_bayes(
                    p[is_single_test], pos_dict["a_single_final"], pos_dict["b_single_final"]
                )
            if np.any(~is_single_test):
                pred_te[~is_single_test] = predizer_regra_bayes(
                    p[~is_single_test], pos_dict["a_multi_final"], pos_dict["b_multi_final"]
                )
        else:
            pred_te = predizer_regra_bayes(p, pos_dict["a_final"], pos_dict["b_final"])

        sub = pd.DataFrame({
            "CLI_CPF": test_df["CLI_CPF"].astype(str).values,
            "target": pred_te.astype(int),
        })
        assert len(sub) == 25691, f"Tamanho inesperado: {len(sub)}"
        assert (sub["CLI_CPF"].values == sample_sub["CLI_CPF"].astype(str).str.strip().values).all(), "Divergencia CLI_CPF"
        assert sub["target"].isna().sum() == 0, "NaNs na submissao"
        assert set(sub["target"].unique()).issubset({0, 1, 2, 3}), "Valores fora de {0,1,2,3}"
        caminho = pasta / filename
        sub.to_csv(caminho, index=False)
        return sub, caminho

    sub1, path1 = aplicar_e_salvar(proba_test_ens, pos_ens, "submissao_1_ensemble.csv")
    sub2, path2 = aplicar_e_salvar(proba_test_ind, pos_ind, "submissao_2_melhor_individual.csv")

    dist_comp = pd.DataFrame({
        "Train_Real_%": pd.Series(y_train).value_counts(normalize=True).sort_index().mul(100).round(2),
        "OOF_Ensemble_%": pd.Series(pos_ens["pred_oof"]).value_counts(normalize=True).sort_index().mul(100).round(2),
        "Test_Ensemble_%": sub1["target"].value_counts(normalize=True).sort_index().mul(100).round(2),
        "OOF_Individual_%": pd.Series(pos_ind["pred_oof"]).value_counts(normalize=True).sort_index().mul(100).round(2),
        "Test_Individual_%": sub2["target"].value_counts(normalize=True).sort_index().mul(100).round(2),
    }).fillna(0.0)
    dist_comp.index.name = "Classe"

    concordancia = float((sub1["target"].values == sub2["target"].values).mean() * 100)
    return {
        "path_submissao_1": str(path1),
        "path_submissao_2": str(path2),
        "distribuicao_classes": dist_comp,
        "concordancia_sub1_sub2_pct": round(concordancia, 2),
    }


def _preparar_df_catboost_nativo(
    X_sub: pd.DataFrame,
    X_inn: pd.DataFrame,
    X_va: Optional[pd.DataFrame],
    X_te: Optional[pd.DataFrame],
    grupos: Dict[str, List[str]],
) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[pd.DataFrame], Optional[pd.DataFrame], List[str]]:
    """
    Prepara DataFrames para CatBoost com cat_features nativas + 19 atributos BG/NBD
    ajustados exclusivamente no sub-treino interno (zero target, zero leakage).
    """
    num_cols = list(grupos["numericas"])
    cat_cols_all = list(grupos["categoricas"]) + list(grupos["alta_card"])
    bgnbd_cand = ["n_dias_com_compra", "antiguidade_dias", "recencia_dias", "ciclo_medio_dias", "variacao_ciclo", "visitas_90d"]
    bgnbd_cols = [c for c in bgnbd_cand if c in num_cols]

    bg_tx = None
    bg_names: List[str] = []
    if len(bgnbd_cols) >= 5:
        bg_tx = BGNBDFeatureTransformer()
        bg_tx.fit(X_sub[bgnbd_cols])
        bg_names = [str(c) for c in bg_tx.get_feature_names_out()]

    def _montar_bloco(df_in: Optional[pd.DataFrame]) -> Optional[pd.DataFrame]:
        if df_in is None or len(df_in) == 0:
            return None
        bloco_num = df_in[num_cols].astype(np.float64).reset_index(drop=True)
        partes = [bloco_num]
        if bg_tx is not None:
            bg_mat = bg_tx.transform(df_in[bgnbd_cols])
            partes.append(pd.DataFrame(bg_mat, columns=bg_names))
        if cat_cols_all:
            bloco_cat = pd.DataFrame({
                c: df_in[c].where(pd.notna(df_in[c]), "AUSENTE").astype(str).to_numpy()
                for c in cat_cols_all
            })
            partes.append(bloco_cat)
        return pd.concat(partes, axis=1)

    return (
        _montar_bloco(X_sub),
        _montar_bloco(X_inn),
        _montar_bloco(X_va),
        _montar_bloco(X_te),
        cat_cols_all,
    )


def _rodar_catboost_nativo_cv(
    params: Dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    grupos: Dict[str, List[str]],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    X_test: Optional[pd.DataFrame] = None,
    early_stopping_rounds: int = 40,
) -> Dict[str, Any]:
    """Treina CatBoost MultiClass com cat_features nativas, BG/NBD in-fold e Early Stopping (90/10)."""
    if not HAS_CATBOOST:
        raise RuntimeError("CatBoost nao instalado.")
    y = np.asarray(y, dtype=int)
    oof_proba = np.zeros((len(X), 4), dtype=np.float64)
    test_proba = np.zeros((len(X_test), 4), dtype=np.float64) if X_test is not None else None
    metricas_folds = []
    best_iters = []

    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        sub_tr_idx, inner_va_idx = train_test_split(
            tr_idx, test_size=0.10, stratify=y[tr_idx], random_state=RANDOM_STATE_DEFAULT + fold_id
        )
        df_sub, df_inn, df_va, df_te, cat_cols_all = _preparar_df_catboost_nativo(
            X.iloc[sub_tr_idx], X.iloc[inner_va_idx], X.iloc[va_idx], X_test, grupos
        )
        pool_sub = Pool(df_sub, y[sub_tr_idx], cat_features=cat_cols_all)
        pool_inn = Pool(df_inn, y[inner_va_idx], cat_features=cat_cols_all)

        cb_p = dict(params)
        cb_p["random_seed"] = int(cb_p.get("random_seed", RANDOM_STATE_DEFAULT)) + fold_id
        cb_p["allow_writing_files"] = False
        clf = CatBoostClassifier(**cb_p)
        clf.fit(pool_sub, eval_set=pool_inn, early_stopping_rounds=early_stopping_rounds, verbose=False)
        best_iters.append(int(clf.get_best_iteration() or cb_p.get("iterations", 400)))

        p_va = clf.predict_proba(Pool(df_va, cat_features=cat_cols_all))
        oof_proba[va_idx] = p_va
        if test_proba is not None and df_te is not None:
            test_proba += clf.predict_proba(Pool(df_te, cat_features=cat_cols_all)) / len(folds)
        metricas_folds.append(calcular_metricas_completas(y[va_idx], p_va))

    return {
        "oof_proba": oof_proba,
        "test_proba": test_proba,
        "best_iters": best_iters,
        "mean_best_iter": int(np.mean(best_iters)) if best_iters else 0,
        "metricas_argmax": calcular_metricas_completas(y, oof_proba),
        "metricas_folds_argmax": pd.DataFrame(metricas_folds),
    }


def _fit_predict_binary_catboost_nativo(
    params_stage: Dict[str, Any],
    X_st_tr: pd.DataFrame,
    y_st_tr: np.ndarray,
    grupos: Dict[str, List[str]],
    X_st_va: pd.DataFrame,
    X_st_te: Optional[pd.DataFrame],
    seed: int,
    early_stopping_rounds: int,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Estagio binario CatBoost com cat_features nativas e Early Stopping interno (90/10)."""
    y_st_tr = np.asarray(y_st_tr, dtype=int)
    sub_i, inn_i = train_test_split(
        np.arange(len(y_st_tr)), test_size=0.10, stratify=y_st_tr, random_state=seed
    )
    df_sub, df_inn, df_va, df_te, cat_cols_all = _preparar_df_catboost_nativo(
        X_st_tr.iloc[sub_i], X_st_tr.iloc[inn_i], X_st_va, X_st_te, grupos
    )
    cb_p = dict(params_stage)
    cb_p["random_seed"] = seed
    cb_p["allow_writing_files"] = False
    clf = CatBoostClassifier(**cb_p)
    clf.fit(
        Pool(df_sub, y_st_tr[sub_i], cat_features=cat_cols_all),
        eval_set=Pool(df_inn, y_st_tr[inn_i], cat_features=cat_cols_all),
        early_stopping_rounds=early_stopping_rounds,
        verbose=False,
    )
    p_va = clf.predict_proba(Pool(df_va, cat_features=cat_cols_all))[:, 1] if df_va is not None else np.empty(0, dtype=np.float64)
    p_te = clf.predict_proba(Pool(df_te, cat_features=cat_cols_all))[:, 1] if df_te is not None else (np.empty(0, dtype=np.float64) if X_st_te is not None else None)
    return p_va, p_te


def _rodar_hierarquico_catboost_nativo(
    params: Dict[str, Any],
    X: pd.DataFrame,
    y: np.ndarray,
    grupos: Dict[str, List[str]],
    folds: List[Tuple[np.ndarray, np.ndarray]],
    X_test: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """Decomposicao Hierarquica em 3 estagios utilizando CatBoost com cat_features nativas."""
    if not HAS_CATBOOST:
        raise RuntimeError("CatBoost nao instalado.")
    y = np.asarray(y, dtype=int)
    ciclo_tr = X["ciclo_medio_dias"].to_numpy(dtype=np.float64)
    elig_tr = (X["n_dias_com_compra"].to_numpy() >= 2) & np.isfinite(ciclo_tr) & (ciclo_tr <= 46.5)

    elig_te = None
    if X_test is not None:
        ciclo_te = X_test["ciclo_medio_dias"].to_numpy(dtype=np.float64)
        elig_te = (X_test["n_dias_com_compra"].to_numpy() >= 2) & np.isfinite(ciclo_te) & (ciclo_te <= 46.5)

    params_s1 = dict(params)
    params_s2 = dict(params)
    params_s3 = dict(params)
    params_s1.update({"loss_function": "Logloss", "eval_metric": "Logloss", "allow_writing_files": False})
    params_s2.update({"loss_function": "Logloss", "eval_metric": "Logloss", "allow_writing_files": False})
    params_s3.update({
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "allow_writing_files": False,
        "depth": min(int(params_s3.get("depth", 4)), 4),
        "l2_leaf_reg": max(float(params_s3.get("l2_leaf_reg", 12.0)), 12.0),
    })

    oof_proba = np.zeros((len(X), 4), dtype=np.float64)
    test_proba = np.zeros((len(X_test), 4), dtype=np.float64) if X_test is not None else None
    X_te_elig = X_test.iloc[elig_te] if (X_test is not None and elig_te is not None) else None

    for fold_id, (tr_idx, va_idx) in enumerate(folds):
        seed_base = RANDOM_STATE_DEFAULT + fold_id * 10
        X_va_fold = X.iloc[va_idx]

        p_rec_va, p_rec_te = _fit_predict_binary_catboost_nativo(
            params_s1, X.iloc[tr_idx], (y[tr_idx] != 3).astype(int), grupos, X_va_fold, X_test, seed_base + 1, 40
        )

        tr_rec = tr_idx[y[tr_idx] != 3]
        p_del_va, p_del_te = _fit_predict_binary_catboost_nativo(
            params_s2, X.iloc[tr_rec], np.isin(y[tr_rec], [1, 2]).astype(int), grupos, X_va_fold, X_test, seed_base + 2, 40
        )

        tr_s3 = tr_idx[np.isin(y[tr_idx], [1, 2]) & elig_tr[tr_idx]]
        va_elig_mask = elig_tr[va_idx]
        p_dorm_elig_va, p_dorm_elig_te = _fit_predict_binary_catboost_nativo(
            params_s3, X.iloc[tr_s3], (y[tr_s3] == 2).astype(int), grupos, X_va_fold.iloc[va_elig_mask], X_te_elig, seed_base + 3, 30
        )

        p_dorm_va = np.zeros(len(va_idx), dtype=np.float64)
        p_dorm_va[va_elig_mask] = p_dorm_elig_va
        oof_proba[va_idx] = np.column_stack([
            p_rec_va * (1.0 - p_del_va),
            p_rec_va * p_del_va * (1.0 - p_dorm_va),
            p_rec_va * p_del_va * p_dorm_va,
            1.0 - p_rec_va,
        ])

        if test_proba is not None and X_test is not None and elig_te is not None:
            p_dorm_te = np.zeros(len(X_test), dtype=np.float64)
            if p_dorm_elig_te is not None and len(p_dorm_elig_te) > 0:
                p_dorm_te[elig_te] = p_dorm_elig_te
            p_te_fold = np.column_stack([
                p_rec_te * (1.0 - p_del_te),
                p_rec_te * p_del_te * (1.0 - p_dorm_te),
                p_rec_te * p_del_te * p_dorm_te,
                1.0 - p_rec_te,
            ])
            test_proba += p_te_fold / len(folds)

    metricas_folds = [calcular_metricas_completas(y[va], oof_proba[va]) for _, va in folds]
    return {
        "oof_proba": oof_proba,
        "test_proba": test_proba,
        "metricas_argmax": calcular_metricas_completas(y, oof_proba),
        "metricas_folds_argmax": pd.DataFrame(metricas_folds),
    }


def executar_pipeline_10fold_170f(
    X_full: pd.DataFrame,
    y: np.ndarray,
    grp_full: Dict[str, List[str]],
    X_te_full: pd.DataFrame,
    best_lgbm_params: Dict[str, Any],
    best_xgb_params: Dict[str, Any],
    cache_dir: Optional[str | Path] = None,
    usar_cache: bool = True,
    X_sel: Optional[pd.DataFrame] = None,
    grp_sel: Optional[Dict[str, List[str]]] = None,
    X_te_sel: Optional[pd.DataFrame] = None,
) -> Dict[str, Any]:
    """
    Experimento 6B (Bonus Avancado): 10-Fold Stratified CV sobre TODAS as 170 colunas base (+19 BG/NBD),
    incluindo Hierarquico CatBoost Nativo (Ordered Target Statistics) e modelos complementares Top 82
    + Blending Linear vs Log-Odds.
    """
    y = np.asarray(y, dtype=int)
    folds_10 = gerar_folds_estratificados(y, n_splits=10, random_state=RANDOM_STATE_DEFAULT)
    num_cols = grp_full["numericas"]
    cat_cols = grp_full["categoricas"]
    alta_card = grp_full["alta_card"]

    X_s = X_sel if X_sel is not None else X_full
    g_s = grp_sel if grp_sel is not None else grp_full
    X_ts = X_te_sel if X_te_sel is not None else X_te_full

    cb_nat_bin = {
        "iterations": 1000,
        "learning_rate": 0.045,
        "depth": 5,
        "l2_leaf_reg": 10.0,
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "max_ctr_complexity": 2,
        "one_hot_max_size": 10,
        "random_seed": RANDOM_STATE_DEFAULT,
        "thread_count": N_JOBS_DEFAULT,
        "allow_writing_files": False,
    }

    especs: List[Tuple[str, Callable[[], Dict[str, Any]]]] = [
        (
            "Logistic Regression (10F, 170f)",
            lambda: rodar_cv_oof(
                lambda: Pipeline([
                    ("preprocessor", criar_preprocessador(num_cols, cat_cols, alta_card, modo="linear")),
                    ("classifier", LogisticRegression(max_iter=250, C=0.5, random_state=RANDOM_STATE_DEFAULT)),
                ]),
                X_full, y, folds_10, sample_weight_mode=None, X_test=X_te_full,
            ),
        ),
        (
            "MLP Neural Network (10F, 170f)",
            lambda: rodar_cv_oof(
                lambda: Pipeline([
                    ("preprocessor", criar_preprocessador(num_cols, cat_cols, alta_card, modo="linear")),
                    ("classifier", MLPClassifier(
                        hidden_layer_sizes=(64, 32), alpha=1e-2, early_stopping=True,
                        max_iter=40, n_iter_no_change=5, random_state=RANDOM_STATE_DEFAULT,
                    )),
                ]),
                X_full, y, folds_10, sample_weight_mode=None, X_test=X_te_full,
            ),
        ),
        (
            "LightGBM Tunado (10F, 170f)",
            lambda: rodar_cv_gbdt_early_stopping("lgbm", best_lgbm_params, X_full, y, grp_full, folds_10, X_test=X_te_full),
        ),
        (
            "XGBoost (ES, 10F, 170f)",
            lambda: rodar_cv_gbdt_early_stopping("xgb", _params_xgb_padrao(1200), X_full, y, grp_full, folds_10, X_test=X_te_full),
        ),
        (
            "Hierarquico LightGBM (3-Est, 10F, 170f)",
            lambda: _rodar_hierarquico_3_estagios("lgbm", best_lgbm_params, X_full, y, grp_full, folds_10, X_test=X_te_full),
        ),
        (
            "Hierarquico XGBoost (3-Est, 10F, 170f)",
            lambda: _rodar_hierarquico_3_estagios("xgb", best_xgb_params, X_full, y, grp_full, folds_10, X_test=X_te_full),
        ),
        (
            "Especialista LightGBM (1-Dia vs 2+Dias, 10F, 170f)",
            lambda: _rodar_especialista_regime("lgbm", best_lgbm_params, X_full, y, grp_full, folds_10, X_test=X_te_full),
        ),
        (
            "Logistic Regression (10F, Top 82)",
            lambda: rodar_cv_oof(
                lambda: Pipeline([
                    ("preprocessor", criar_preprocessador(g_s["numericas"], g_s["categoricas"], g_s["alta_card"], modo="linear")),
                    ("classifier", LogisticRegression(max_iter=250, C=0.5, random_state=RANDOM_STATE_DEFAULT)),
                ]),
                X_s, y, folds_10, sample_weight_mode=None, X_test=X_ts,
            ),
        ),
        (
            "MLP Neural Network (10F, Top 82)",
            lambda: rodar_cv_oof(
                lambda: Pipeline([
                    ("preprocessor", criar_preprocessador(g_s["numericas"], g_s["categoricas"], g_s["alta_card"], modo="linear")),
                    ("classifier", MLPClassifier(
                        hidden_layer_sizes=(64, 32), alpha=1e-2, early_stopping=True,
                        max_iter=40, n_iter_no_change=5, random_state=RANDOM_STATE_DEFAULT,
                    )),
                ]),
                X_s, y, folds_10, sample_weight_mode=None, X_test=X_ts,
            ),
        ),
    ]
    if HAS_CATBOOST:
        especs.append((
            "Hierarquico CatBoost Nativo (3-Est, 10F, 170f)",
            lambda: _rodar_hierarquico_catboost_nativo(cb_nat_bin, X_full, y, grp_full, folds_10, X_test=X_te_full),
        ))

    linhas = []
    resultados = {}
    oof_probas = {}
    test_probas = {}
    posproc_dict = {}
    total = len(especs)

    for idx, (nome, fn_cv) in enumerate(especs, start=1):
        t0 = time.time()

        def _exec(fn=fn_cv):
            res_c = fn()
            pos_c = avaliar_posproc_nested_oof(
                res_c["oof_proba"], y, X_full, folds_10, usar_mascara=True, por_regime=False, calibrar=True, metodo_regra="plugin"
            )
            return {"cv": res_c, "pos": pos_c}

        if cache_dir is not None:
            slug = re.sub(r"[^a-z0-9]+", "_", nome.lower()).strip("_")
            pac = carregar_ou_executar(f"v4_10f_{slug}.joblib", _exec, cache_dir=cache_dir, usar_cache=usar_cache)
        else:
            pac = _exec()

        resultados[nome] = pac
        oof_probas[nome] = pac["cv"]["oof_proba"]
        test_probas[nome] = pac["cv"]["test_proba"]
        posproc_dict[nome] = pac["pos"]

        met = pac["pos"]["metricas_oof"]
        print(
            f"  [{idx}/{total}] {nome:46s} -> F1_PosProc={met['F1_macro']:.4f} | AUC={met['AUC_OvR_macro']:.4f} | "
            f"LL={met['Log_Loss']:.4f} ({time.time() - t0:.1f}s)",
            flush=True,
        )
        linhas.append({
            "Modelo": nome,
            "F1_macro_Argmax": pac["cv"]["metricas_argmax"]["F1_macro"],
            "F1_macro_PosProc": met["F1_macro"],
            "F1_std_folds": float(pac["pos"]["metricas_por_fold"]["F1_macro"].std()),
            "AUC_OvR_macro": met["AUC_OvR_macro"],
            "Log_Loss": met["Log_Loss"],
            "Balanced_Acc": met["Balanced_Acc"],
            "F1_C0": met["F1_C0"],
            "F1_C1": met["F1_C1"],
            "F1_C2": met["F1_C2"],
            "F1_C3": met["F1_C3"],
        })

    res_ens_10f = executar_blending_ensemble(oof_probas, test_probas, y, X_full, folds_10)
    tabela_modelos = pd.DataFrame(linhas).sort_values("F1_macro_PosProc", ascending=False).reset_index(drop=True)
    return {
        "folds_10": folds_10,
        "tabela_modelos": tabela_modelos,
        "resultados": resultados,
        "oof_probas": oof_probas,
        "test_probas": test_probas,
        "posproc": posproc_dict,
        "ensemble": res_ens_10f,
    }
