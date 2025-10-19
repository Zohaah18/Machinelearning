#Question 2:
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, matthews_corrcoef

CSV_FILE = "wdbc_data.csv"    
TARGET_COL = "Diagnosis"      
MAP_TARGET = {'M': 1, 'B': 0}
RANDOM_STATE = 42
CV_FOLDS = 10
PARTITIONS = [(0.5, 0.5), (0.7, 0.3), (0.8, 0.2)]
C_range = 2.0 ** np.arange(-5, 16, 2)
gamma_range = 2.0 ** np.array([-15, -11, -7, -3, 0, 3])
param_grid = {'C': C_range, 'gamma': gamma_range, 'kernel': ['rbf']}


def compute_f_scores(X: pd.DataFrame, y: pd.Series):
    f_vals = {}
    for col in X.columns:
        xi = X[col].astype(float).values
        mean_all = np.mean(xi)
        pos = xi[y == 1]
        neg = xi[y == 0]
        if len(pos) == 0 or len(neg) == 0:
            f_vals[col] = 0.0
            continue
        mean_pos = np.mean(pos)
        mean_neg = np.mean(neg)
        var_pos = np.var(pos, ddof=0)
        var_neg = np.var(neg, ddof=0)
        numerator = (mean_pos - mean_all) ** 2 + (mean_neg - mean_all) ** 2
        denominator = var_pos + var_neg
        f_vals[col] = float(numerator / denominator) if denominator != 0 else 0.0
    return pd.Series(f_vals).sort_values(ascending=False)

def tune_and_evaluate(X_train, y_train, X_test, y_test):
    svc = SVC()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    grid = GridSearchCV(svc, param_grid, cv=cv, scoring='accuracy', n_jobs=1, verbose=0)
    grid.fit(X_train, y_train)
    best = grid.best_estimator_
    y_pred = best.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    return best, grid.best_params_, acc, y_pred

def safe_confusion_ravel(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    if cm.shape == (2,2):
        tn, fp, fn, tp = cm.ravel()
    else:
        full = np.zeros((2,2), dtype=int)
        for i, actual in enumerate([0,1]):
            for j, pred in enumerate([0,1]):
                try:
                    full[i,j] = cm[i,j]
                except Exception:
                    full[i,j] = 0
        tn, fp, fn, tp = full.ravel()
    return int(tn), int(fp), int(fn), int(tp)

def compute_metrics_from_confusion(y_true, y_pred):
    tn, fp, fn, tp = safe_confusion_ravel(y_true, y_pred)
    total = tp + tn + fp + fn
    accuracy = (tp + tn) / total if total else 0.0
    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    npv = tn / (tn + fn) if (tn + fn) else 0.0
    fdr = fp / (fp + tp) if (fp + tp) else 0.0
    forate = fn / (fn + tn) if (fn + tn) else 0.0
    mcc = matthews_corrcoef(y_true, y_pred) if len(np.unique(y_true)) > 1 else 0.0
    return {
        'accuracy': accuracy,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'ppv': ppv,
        'npv': npv,
        'fdr': fdr,
        'forate': forate,
        'mcc': mcc,
        'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
    }

def run_experiment():
    df = pd.read_csv(CSV_FILE)
    first_col = df.columns[0]
    if first_col.lower() in ('id', 'id_number', 'idnumber', 'index'):
        df = df.drop(columns=[first_col])
    if 'ID_number' in df.columns:
        df = df.drop(columns=['ID_number'])
    if 'id' in df.columns:
        df = df.drop(columns=['id'])

    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found in CSV. Columns: {list(df.columns)}")
    if df[TARGET_COL].dtype == object:
        df[TARGET_COL] = df[TARGET_COL].map(MAP_TARGET)


    X_all = df.drop(columns=[TARGET_COL])
    y_all = df[TARGET_COL].astype(int)

    table2 = {}
    table3 = {}
    table4 = {}
    table6 = {}
    table7 = {}

    for train_frac, test_frac in PARTITIONS:
        print(f"\nRunning partition {int(train_frac*100)}-{int(test_frac*100)} ...")
        X_train, X_test, y_train, y_test = train_test_split(
            X_all, y_all, train_size=train_frac, test_size=test_frac,
            stratify=y_all, random_state=RANDOM_STATE
        )

        fseries = compute_f_scores(X_train, y_train)
        table2[f"{int(train_frac*100)}-{int(test_frac*100)}"] = fseries

        ranked_features = list(fseries.index)
        models_features = []
        accuracies = []

        m = len(ranked_features)
        for k in range(1, m+1):
            selected = ranked_features[:k]
            models_features.append(selected)

            Xtr_sel = X_train[selected]
            Xte_sel = X_test[selected]

            best_est, best_params, acc, ypred = tune_and_evaluate(Xtr_sel, y_train, Xte_sel, y_test)
            accuracies.append(acc * 100)

            if k == 5:
                metrics = compute_metrics_from_confusion(y_test, ypred)
                table6[f"{int(train_frac*100)}-{int(test_frac*100)}"] = metrics
                table7[f"{int(train_frac*100)}-{int(test_frac*100)}"] = {
                    'confusion': (metrics['tn'], metrics['fp'], metrics['fn'], metrics['tp']),
                    'best_params': best_params
                }

            print(f"  Model#{k}: {k} features, test acc = {acc*100:.4f} %, best_params={best_params}")

        table3[f"{int(train_frac*100)}-{int(test_frac*100)}"] = models_features
        table4[f"{int(train_frac*100)}-{int(test_frac*100)}"] = accuracies

        print(f"Partition {int(train_frac*100)}-{int(test_frac*100)} top features:")
        print(fseries.head(10))

    for part, series in table2.items():
        out = series.rename_axis('Feature').reset_index(name='F-score')
        out.to_csv(f"table2_f_scores_{part}.csv", index=False)
        print(f"Saved Table2 for {part} -> table2_f_scores_{part}.csv")

    for part, models in table3.items():
        rows = []
        for i, featlist in enumerate(models, start=1):
            rows.append({'Model': f'Model#{i}', 'NumFeatures': len(featlist), 'Features': ';'.join(featlist)})
        pd.DataFrame(rows).to_csv(f"table3_models_{part}.csv", index=False)
        print(f"Saved Table3 for {part} -> table3_models_{part}.csv")

    for part, accs in table4.items():
        rows = [{'Model': f'Model#{i+1}', 'Accuracy(%)': round(accs[i], 4)} for i in range(len(accs))]
        pd.DataFrame(rows).to_csv(f"table4_accuracies_{part}.csv", index=False)
        print(f"Saved Table4 for {part} -> table4_accuracies_{part}.csv")

    for part, metrics in table6.items():
        mdf = pd.DataFrame([{
            'Partition': part,
            'Accuracy(%)': round(metrics['accuracy']*100, 4),
            'Sensitivity(%)': round(metrics['sensitivity']*100, 4),
            'Specificity(%)': round(metrics['specificity']*100, 4),
            'PPV(%)': round(metrics['ppv']*100, 4),
            'NPV(%)': round(metrics['npv']*100, 4),
            'FDR(%)': round(metrics['fdr']*100, 4),
            'FOR(%)': round(metrics['forate']*100, 4),
            'MCC': round(metrics['mcc'], 4)
        }])
        mdf.to_csv(f"table6_model5_{part}.csv", index=False)
        print(f"Saved Table6 for model#5 {part} -> table6_model5_{part}.csv")

    for part, info in table7.items():
        tn, fp, fn, tp = info['confusion']
        cm_df = pd.DataFrame([{'Partition': part, 'TN': tn, 'FP': fp, 'FN': fn, 'TP': tp}])
        cm_df.to_csv(f"table7_confusion_model5_{part}.csv", index=False)
        print(f"Saved Table7 (confusion) for model#5 {part} -> table7_confusion_model5_{part}.csv")
        print(f"Best params for model#5 on {part}: {info.get('best_params')}")

    print("\nAll experiments completed. CSVs saved in working directory.")

if __name__ == "__main__":
    run_experiment()
