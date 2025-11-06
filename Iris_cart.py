# Iris（あやめ）データで学ぶCART（決定木分類）教材アプリ
# O. Watanabe, Shinshu Univ.
# ---------------------------------------------------------------
# 使い方:
#   1) pip install -r requirements.txt
#   2) streamlit run app.py
# ---------------------------------------------------------------

import io
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# ページ設定
st.set_page_config(page_title="Irisで学ぶCART（決定木）", layout="wide")

# ============================================================
# ユーティリティ
# ============================================================

def build_meshgrid(X, h=0.02, pad=0.5):
    x_min, x_max = X[:, 0].min() - pad, X[:, 0].max() + pad
    y_min, y_max = X[:, 1].min() - pad, X[:, 1].max() + pad
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    return xx, yy


def plot_confusion_matrix(cm, classes, normalize=False):
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)
    fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=140)
    im = ax.imshow(cm, interpolation="nearest")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(len(classes)), yticks=np.arange(len(classes)))
    ax.set_xticklabels(classes, rotation=45, ha="right")
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            txt = f"{cm[i, j]:.2f}" if normalize else str(int(cm[i, j]))
            ax.text(j, i, txt, ha="center", va="center")
    fig.tight_layout()
    return fig


# ============================================================
# データ読み込み
# ============================================================
iris = load_iris(as_frame=True)
X_full = iris.data.copy()
y = iris.target.copy()
feature_names = list(X_full.columns)
class_names = list(iris.target_names)

JP_LABELS = {
    "sepal length (cm)": "がく片 長 (cm)",
    "sepal width (cm)": "がく片 幅 (cm)",
    "petal length (cm)": "花弁 長 (cm)",
    "petal width (cm)": "花弁 幅 (cm)",
}

# ============================================================
# サイドバー
# ============================================================
st.sidebar.header("📚 学習設定 / Learning Controls")

split_ratio = st.sidebar.slider("学習データの割合 / Train size", 0.5, 0.9, 0.7, 0.05)
random_state = st.sidebar.number_input("乱数シード / Random state", min_value=0, value=42, step=1)
stratify = st.sidebar.checkbox("層化サンプリング / Stratify", value=True)

# 特徴量選択
st.sidebar.subheader("🔧 特徴量選択 / Feature selection")
selected_features = st.sidebar.multiselect(
    "使う特徴量 / Features to use",
    feature_names,
    default=feature_names,
)
if len(selected_features) < 2:
    st.sidebar.warning("少なくとも2つの特徴量を選んでください。")

axis_options = selected_features if selected_features else feature_names
if len(axis_options) < 2:
    axis_options = feature_names
x_axis = st.sidebar.selectbox("X軸", axis_options, index=0)
y_options = [f for f in axis_options if f != x_axis]
if not y_options:
    y_options = [f for f in feature_names if f != x_axis]
y_axis = st.sidebar.selectbox("Y軸", y_options, index=0)

# 決定木ハイパーパラメータ
st.sidebar.subheader("🌲 決定木パラメータ / Decision Tree params")
criterion = st.sidebar.selectbox("不純度指標 / Criterion", ["gini", "entropy", "log_loss"], index=0)
max_depth = st.sidebar.slider("最大深さ / Max depth", 1, 10, 3)
min_samples_split = st.sidebar.slider("最小分割サンプル数 / min_samples_split", 2, 20, 2)
min_samples_leaf = st.sidebar.slider("最小葉ノードサンプル数 / min_samples_leaf", 1, 10, 1)
class_weight_opt = st.sidebar.selectbox("class_weight", ["None", "balanced"], index=0)
class_weight = None if class_weight_opt == "None" else class_weight_opt

cv_k = st.sidebar.slider("交差検証分割数 / CV folds", 2, 10, 5)

show_rules = st.sidebar.checkbox("ルール表示 / Show rules", value=True)
show_tree = st.sidebar.checkbox("決定木の図を表示 / Show tree", value=True)
show_boundary = st.sidebar.checkbox("決定境界を表示(2特征) / Decision boundary", value=True)
normalize_cm = st.sidebar.checkbox("混同行列を正規化 / Normalize confusion matrix", value=False)

# ============================================================
# 学習と評価
# ============================================================
X = X_full[selected_features].values if selected_features else X_full.values

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    train_size=split_ratio,
    random_state=random_state,
    stratify=y if stratify else None,
)

clf = DecisionTreeClassifier(
    criterion=criterion,
    max_depth=max_depth,
    min_samples_split=min_samples_split,
    min_samples_leaf=min_samples_leaf,
    class_weight=class_weight,
    random_state=random_state,
)
clf.fit(X_train, y_train)

cv_scores = cross_val_score(clf, X, y, cv=cv_k)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
report_text = classification_report(y_test, y_pred, target_names=class_names, zero_division=0)

# ============================================================
# レイアウト
# ============================================================
left, right = st.columns([1.1, 1])

with left:
    st.markdown("## 🌼 Iris × CART 学習 / Training")

    st.markdown("### 1) データセットの確認 / Preview first 15 rows")
    st.dataframe(X_full.head(15), use_container_width=True)
    st.caption("データ分布や外れ値を確認し、どの特徴量を使うかを考えます。")

    st.write("**特徴量 / Features:** ", ', '.join(selected_features))
    st.write(f"**学習サイズ / Train size:** {split_ratio:.2f}  |  **乱数 / Seed:** {random_state}")
    st.write(f"**精度(テスト) / Accuracy (test):** {acc:.3f}")
    st.write(f"**交差検証平均 / CV mean:** {cv_scores.mean():.3f}  (± {cv_scores.std():.3f})")

    st.markdown("### 2) 散布図 / Scatter (選択2軸)")
    st.caption("散布図でクラスの分離が良い2変数の組み合わせを選びます。")
    x_idx = feature_names.index(x_axis)
    y_idx = feature_names.index(y_axis)
    fig_sc, ax_sc = plt.subplots(figsize=(6, 4.5), dpi=140)
    for i, cname in enumerate(class_names):
        mask = (y == i)
        ax_sc.scatter(X_full.loc[mask, x_axis], X_full.loc[mask, y_axis], label=cname, s=35)
    ax_sc.set_xlabel(f"{x_axis} / {JP_LABELS.get(x_axis, x_axis)}")
    ax_sc.set_ylabel(f"{y_axis} / {JP_LABELS.get(y_axis, y_axis)}")
    ax_sc.legend(title="Class")
    ax_sc.grid(alpha=0.3)
    st.pyplot(fig_sc, use_container_width=True)

    if show_boundary:
        st.markdown("### 3) 決定境界 / Decision boundary")
        st.caption("サイドバーで gini/entropy/log_loss, 深度, 分割数, 最小ノード を調整して境界の変化を観察します。")
        X2 = X_full[[x_axis, y_axis]].values
        clf2 = DecisionTreeClassifier(
            criterion=criterion,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
        ).fit(X2, y)
        xx, yy = build_meshgrid(X2, h=0.02, pad=0.5)
        Z = clf2.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
        fig_bd, ax_bd = plt.subplots(figsize=(6, 4.5), dpi=140)
        ax_bd.contourf(xx, yy, Z, alpha=0.2)
        for i, cname in enumerate(class_names):
            mask = (y == i)
            ax_bd.scatter(X2[mask, 0], X2[mask, 1], label=cname, s=25)
        ax_bd.set_xlabel(f"{x_axis} / {JP_LABELS.get(x_axis, x_axis)}")
        ax_bd.set_ylabel(f"{y_axis} / {JP_LABELS.get(y_axis, y_axis)}")
        ax_bd.legend(title="Class")
        ax_bd.grid(alpha=0.3)
        st.pyplot(fig_bd, use_container_width=True)

with right:
    st.markdown("## 📊 評価 / Evaluation")
    st.markdown("### 混同行列 / Confusion matrix")
    fig_cm = plot_confusion_matrix(cm, class_names, normalize=normalize_cm)
    st.pyplot(fig_cm, use_container_width=True)

    st.markdown("### 分類レポート / Classification report")
    st.code(report_text)

    if show_tree:
        st.markdown("### 決定木の図 / Decision tree plot")
        fig_tr, ax_tr = plt.subplots(figsize=(12, 10), dpi=160)
        plot_tree(
            clf,
            feature_names=selected_features,
            class_names=class_names,
            filled=False,
            impurity=True,
            rounded=True,
            ax=ax_tr,
        )
        st.pyplot(fig_tr, use_container_width=True)

    if show_rules:
        st.markdown("### ルール（テキスト）/ Textual rules")
        rules = export_text(clf, feature_names=selected_features, show_weights=True)
        st.text(rules)

    st.markdown("### 特徴量の重要度 / Feature importances")
    importances = pd.Series(clf.feature_importances_, index=selected_features)
    st.dataframe(importances.sort_values(ascending=False).to_frame("importance"))

# ============================================================
# requirements.txt 表示
# ============================================================
REQ_TXT = """
streamlit>=1.37
scikit-learn>=1.4
pandas>=2.1
numpy>=1.26
matplotlib>=3.8
"""

with st.expander("📦 requirements.txt (コピー用)"):
    st.code(REQ_TXT.strip())

# ============================================================
# ジニ係数の説明（requirements.txt の下に表示）
# ============================================================
with st.expander("🧮 ジニ係数とは？（Gini Impurity）"):
    st.markdown(
        """
        **ジニ係数（Gini Impurity）** は、分類で使う「**混ざり具合**（不純度）」を表す指標です。  
        1つのグループ（ノード）の中に **複数のクラスがどれくらい混在しているか** を数値化します。

        - **ジニ係数 = 0** : そのグループが **1種類のクラスだけ**（完全に純粋）
        - **ジニ係数が大きい** : いろいろなクラスが **混ざっている**

        CART（決定木）では、**分割後のジニ係数が最も小さくなる** ような特徴量・しきい値を選びます。  
        つまり、**混ざり具合を一番減らせる分け方** を貪欲に選択していきます。
        """
    )

    st.markdown("#### 数式（イメージ）")
    st.latex(r"G = 1 - \sum_{k} p_k^2")
    st.markdown(
        r"""
        ここで \(p_k\) はグループ内で **クラス \(k\)** が占める割合です。
        - 例1：すべて同じクラス（\(p=1\)）なら  
          \( G = 1 - 1^2 = 0 \)
        - 例2：2クラスが半々（\(p_1=p_2=0.5\)）なら  
          \( G = 1 - (0.5^2 + 0.5^2) = 0.5 \)
        """
    )

    st.markdown(
        """
        **他の指標との違い（ざっくり）**  
        - **Entropy**（エントロピー）： \\(-\\sum p_k \\log_2 p_k\\)（理論的には似た動き）
        - **誤分類率**： \\(1-\\max(p_k)\\)（単純で鋭敏さに欠けることがある）

        実務では **Gini** か **Entropy** がよく使われます。Irisのようにクラスがはっきり分かれる課題では、
        どちらでも挙動は概ね似ています。
        """
    )

# ------------------------------------
#  最大深さの説明
# ------------------------------------
with st.expander("🌲 最大深さ（Max depth）とは？"):
    st.markdown(
        """
        **最大深さ（Max depth）** とは、決定木が **どれだけ深く分岐できるか** を制限するパラメータです。

        - 深さを **大きく** すると、データをより細かく分類できますが、訓練データに過度に適合してしまい、**過学習** が起こりやすくなります。
        - 深さを **小さく** すると、単純なモデルになり、汎用性は高まりますが、分類精度が下がる場合があります。

        💡 イメージとしては、「**木の深さ＝質問の段数**」です。
        たとえば、深さ3なら「はい/いいえ」の質問を3回行って分類する、ということです。

        適切な最大深さを設定することで、**モデルの複雑さと汎用性のバランス** をとることができます。
        """
    )
# ------------------------------------
#  最小分割サンプル数の説明
# ------------------------------------
with st.expander("🔧 最小分割サンプル数（min_samples_split）とは？"):
    st.markdown(
        """
        **最小分割サンプル数（min_samples_split）** は、**あるノードをさらに分割するために最低限必要なサンプル数**です。  
        この値よりサンプルが少ないノードは、それ以上**分割されません**。

        - 値を **大きく** すると：こまかい分割が減り、木は**浅く**・**単純**になりやすい（過学習を抑制）
        - 値を **小さく** すると：こまかい分割が増え、木は**深く**・**複雑**になりやすい（過学習のリスク）

        💡 直感：**「分岐させるには最低この人数は必要」** というルール。
        例）`min_samples_split=10` なら、そのノードに 10 未満のサンプルしか無いと**分岐不可**。

        **使いどころのヒント**
        - 学習データが少ない／ノイズが多いときは、やや**大きめ**に（例：10〜20）
        - データが十分に多く、細かい構造を拾いたいときは**小さめ**（例：2〜5）
        """
    )

# ------------------------------------
#  最小葉ノードの説明
# ------------------------------------
with st.expander("🍃 最小葉ノード（min_samples_leaf）とは？"):
    st.markdown(
        """
        **最小葉ノード（min_samples_leaf）** は、**葉ノードが保持すべき最少サンプル数**です。  
        これより少ないサンプルしか残らない分割は**許可されません**（＝不安定な“極小葉”を防ぐ）。

        - 値を **大きく** すると：葉がある程度**まとまったサイズ**になりやすく、**汎化**が安定（過学習を抑制）
        - 値を **小さく** すると：葉が**細かく**なりやすく、訓練データには強いが**汎化に弱く**なる可能性

        💡 直感：**「末端の箱は最低これだけの人数で」**というルール。  
        例）`min_samples_leaf=5` なら、どの葉にも最低 5 サンプルが残るように分割されます。

        **使いどころのヒント**
        - ノイズの多いデータやクラス不均衡では、**2〜10** 程度に設定して極端な葉を防ぐ
        - 連続値に弱い外れ値がある場合も、少し**大きめ**にして安定化
        """
    )

# 参考：3つのパラメータの違い（ざっくり早見表）
with st.expander("📋 早見表：max_depth / min_samples_split / min_samples_leaf の違い"):
    st.markdown(
        """
        | パラメータ | 役割 | 値を大きくすると | 値を小さくすると |
        |---|---|---|---|
        | **max_depth** | 木の深さの上限 | 木が浅く・単純に（過学習抑制） | 木が深く・複雑に（過学習リスク） |
        | **min_samples_split** | 分割に必要な最小サンプル数 | こまかい分岐が減る（抑制） | こまかい分岐が増える |
        | **min_samples_leaf** | 葉が保持すべき最小サンプル数 | 極小葉を防ぎ安定化 | 極小葉が増えやすい（過学習リスク） |
        """
    )