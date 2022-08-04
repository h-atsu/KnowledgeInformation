import numpy as np
import copy
from sklearn.datasets import load_wine
from sklearn.preprocessing import scale
from sklearn.semi_supervised import LabelSpreading
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
np.random.seed(42)


wine = load_wine()
X = scale(wine.data)
y = wine.target


X.shape


y.shape


from collections import Counter
Counter(y)


# 描画のためPCAで2次元に圧縮
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)

# クラス数 
n_class = len(np.unique(y)) # = 3

# 教師率
labeled_ratio = 0.1


def set_unlabeled_data(y_org, labeled_ratio):
    y1 = copy.copy(y_org)
    for i in range(len(y_org)):
        if(np.random.random()>labeled_ratio):
            y1[i] = -1
    return y1


y_semi = set_unlabeled_data(y, labeled_ratio)

ls = LabelSpreading(kernel='knn', n_neighbors=10, alpha=0.1)
ls.fit(X_pca, y_semi)

# プロットに使用する推定ラベルを保存
y_pred = ls.predict(X_pca)

# 全データに対するAccuracyの算出
acc = ls.score(X_pca, y)
print("Accuracy: get_ipython().run_line_magic(".4f"", " % acc)")


def plot_ssl(y_pred, y_semi):
    colors = (["lightgreen", "orange", "lightblue", "m", "b", "g", "c", "y", "w", "k"])
    markers = (["s", "o", "v", "^", "D", ">", "<", "d", "p", "H"])

    plt.figure(figsize=(30,8))
    plt.subplot(131)
    for c in range(-1,n_class):
        if(c == -1):
            cls_label = 'unlabeled'
        else:
            cls_label = 'class'+str(c+1)
        plt.scatter(X_pca[y_semi==c, 0],
                    X_pca[y_semi==c, 1],
                    s=50,
                    c=colors[c],
                    marker=markers[c],
                    label= cls_label)

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title("Original Data")
    plt.legend()
    plt.grid()

    plt.subplot(132)
    for c in range(0,n_class):
        plt.scatter(X_pca[y_pred==c, 0],
                    X_pca[y_pred==c, 1],
                    s=50,
                    c=colors[c],
                    marker=markers[c],
                    label="class " + str(c+1))

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title("Labels learned with Label Spreading")
    plt.legend()
    plt.grid()

    plt.subplot(133)
    for c in range(0,n_class):
        plt.scatter(X_pca[y==c, 0],
                    X_pca[y==c, 1],
                    s=50,
                    c=colors[c],
                    marker=markers[c],
                    label="class " + str(c+1))

    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.title("Ground Truth label")
    plt.legend()
    plt.grid()
    plt.show()


plot_ssl(y_pred, y_semi)


num_trial = 50
mean_accs = []
ratio_list = np.linspace(0.04,1)
for labeled_ratio in ratio_list:
    mean_acc = []
    for _ in range(num_trial):        
        y_semi = set_unlabeled_data(y, labeled_ratio)
        ls = LabelSpreading(kernel='knn', n_neighbors=10, alpha=0.1)
        ls.fit(X_pca, y_semi)
        y_pred = ls.predict(X_pca)
        acc = ls.score(X_pca, y)
        mean_acc.append(acc)
    mean_accs.append(np.array(mean_acc).mean())


plt.plot(ratio_list, mean_accs)
plt.xlabel('label ratio')
plt.title("accuracy transition with labeled data num")


print(ratio_list[-1])
print(mean_accs[-1])


num_trial = 100
mean_accs = []
clamping_factors = np.linspace(0.01,0.99,10)
mean_acc = np.zeros(10)
for _ in range(num_trial):        
    y_semi = set_unlabeled_data(y, 0.1)
    for i,cf in enumerate(clamping_factors):
        ls = LabelSpreading(kernel='knn', n_neighbors=10, alpha=cf)
        ls.fit(X_pca, y_semi)
        y_pred = ls.predict(X_pca)
        acc = ls.score(X_pca, y)
        mean_acc[i] += acc
mean_acc /= num_trial


plt.plot(clamping_factors, mean_acc)
plt.xlabel('Calmping Factor')
plt.title("accuracy transition with Clamping factor")


num_trial = 10
mean_accs = []
n_neighbors = np.arange(1,len(y))

mean_acc = np.zeros(len(n_neighbors))
for _ in range(num_trial):        
    y_semi = set_unlabeled_data(y, 0.1)
    for i,ng in enumerate(n_neighbors):
        ls = LabelSpreading(kernel='rbf', n_neighbors=ng, alpha=cf)
        ls.fit(X_pca, y_semi)
        y_pred = ls.predict(X_pca)
        acc = ls.score(X_pca, y)
        mean_acc[i] += acc
mean_acc /= num_trial


plt.plot(n_neighbors, mean_acc)


num_trial = 10
mean_accs = []
n_neighbors = np.arange(1,len(y))

pred_list = []
sm = []

mean_acc = np.zeros(len(n_neighbors))
for j in range(num_trial):        
    y_semi = set_unlabeled_data(y, 0.1)
    for i,ng in enumerate(n_neighbors):
        ls = LabelSpreading(kernel='knn', n_neighbors=ng, alpha=cf)
        ls.fit(X_pca, y_semi)
        y_pred = ls.predict(X_pca)
        acc = ls.score(X_pca, y)
        mean_acc[i] += acc
        if j == 0:
            pred_list.append(y_pred)
            sm = y_semi
mean_acc /= num_trial


plt.plot(n_neighbors, mean_acc)


len(pred_list)


from collections import Counter
Counter(y)


# 全部1と予測した際の正答率
max(Counter(y).values()) / len(y)


plot_ssl(pred_list[-1],sm)


plot_ssl(pred_list[14],sm)
