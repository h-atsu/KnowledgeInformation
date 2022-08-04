get_ipython().run_line_magic("matplotlib", " inline")
from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import precision_recall_fscore_support
import matplotlib.pyplot as plt


# テストデータの割合
test_proportion = 0.3
# Iris データセットをロード  
iris = datasets.load_iris()
# 特徴ベクトルを取得
X = iris.data
# クラスラベルを取得
y = iris.target


# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_proportion, random_state = 1)


# エントロピーを指標とする決定木のインスタンスを生成し，決定木のモデルに学習データを適合させる
tree= DecisionTreeClassifier(criterion='entropy', max_depth=10)
trained_model = tree.fit(X_train, y_train)


plt.figure(figsize=[8,8])
plot_tree(trained_model, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()


y_train_predicted = tree.predict(X_train)
y_test_predicted = tree.predict(X_test)


# テストデータの正解クラスと決定木による予測クラスを出力
print("Test Data")
print("True Label     ", y_test)
print("Predicted Label", y_test_predicted)


fscore_train = precision_recall_fscore_support(y_train, y_train_predicted)
fscore_test = precision_recall_fscore_support(y_test, y_test_predicted)


print('Training data')
print('Class 0 Precision: get_ipython().run_line_magic(".3f,", " Recall: %.3f, Fscore: %.3f' % ")
      (fscore_train[0][0], fscore_train[1][0], fscore_train[2][0]))
print('Class 1 Precision: get_ipython().run_line_magic(".3f,", " Recall: %.3f, Fscore: %.3f' % ")
      (fscore_train[0][1], fscore_train[1][1], fscore_train[2][1]))
print('Class 2 Precision: get_ipython().run_line_magic(".3f,", " Recall: %.3f, Fscore: %.3f' % ")
      (fscore_train[0][2], fscore_train[1][2], fscore_train[2][2]))
print('Average Precision: get_ipython().run_line_magic(".3f,", " Recall: %.3f, Fscore: %.3f' % ")
      (np.average(fscore_train[0]), np.average(fscore_train[1]), np.average(fscore_train[2])))

print('Test data')
print('Class 0 Precision: get_ipython().run_line_magic(".3f,", " Recall: %.3f, Fscore: %.3f' % ")
      (fscore_test[0][0], fscore_test[1][0], fscore_test[2][0]))
print('Class 1 Precision: get_ipython().run_line_magic(".3f,", " Recall: %.3f, Fscore: %.3f' % ")
      (fscore_test[0][1], fscore_test[1][1], fscore_test[2][1]))
print('Class 2 Precision: get_ipython().run_line_magic(".3f,", " Recall: %.3f, Fscore: %.3f' % ")
      (fscore_test[0][2], fscore_test[1][2], fscore_test[2][2]))
print('Average Precision: get_ipython().run_line_magic(".3f,", " Recall: %.3f, Fscore: %.3f' % ")
      (np.average(fscore_test[0]), np.average(fscore_test[1]), np.average(fscore_test[2])))


from sklearn.model_selection import cross_validate


depth = range(1,50)
train_scores = []
test_scores = []
for d in depth:
    tree= DecisionTreeClassifier(criterion='entropy', max_depth=d)
    scores = cross_validate(tree,X, y, cv = 10, return_train_score = True)
    train_score = scores['train_score']
    test_score = scores['test_score']
    train_scores.append(train_score.mean())
    test_scores.append(test_score.mean())


plt.plot(depth, train_scores, label = 'train')
plt.plot(depth, test_scores, label = 'test')
plt.legend()
plt.xlabel('depth of tree')
plt.ylabel('mean accuracy')


print("target 0 の個数 : ",(y == 0).sum())
print("target 1 の個数 : ",(y == 1).sum())
print("target 2 の個数 : ",(y == 2).sum())


from itertools import combinations
from matplotlib.colors import ListedColormap
def plot_decision_boundary(X, y,d):
    plt.figure(figsize = (28,12))
    for ith,axs in enumerate(combinations(range(4),2)):
        ith = ith + 1
        i,j = axs
        x1_min, x1_max = X[:, i].min() , X[:, i].max()
        x2_min, x2_max = X[:, j].min() , X[:, j].max() 
        xx1, xx2 = np.meshgrid(np.arange(x1_min-0.2, x1_max+0.2, 0.02),
                               np.arange(x2_min-0.2, x2_max+0.2, 0.02))
        
        X_std = X[:,:]
        X_std[:,i] = (X[:,i] - x1_min) / (x1_max - x1_min)
        X_std[:,j] = (X[:,j] - x2_min) / (x2_max - x2_min)
        X_std = X_std[:,[i,j]]
        
        tree= DecisionTreeClassifier(criterion='entropy', max_depth=d)
        model = tree.fit(X_std, y)
        Z = model.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
        Z = Z.reshape(xx1.shape)

        markers = ('s', 'x', 'o', '^', 'v')
        colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
        cmap = ListedColormap(colors[:len(np.unique(y))])

        plt.subplot(2,3,ith)

        plt.contourf(xx1, xx2, Z, alpha=0.5, cmap=cmap)
        plt.xlim(xx1.min(), xx1.max())
        plt.ylim(xx2.min(), xx2.max())

        for idx, cl in enumerate(np.unique(y_train)):
            plt.scatter(x=X_std[y == cl, 0], y=X_std[y == cl, 1],
                        alpha=0.8, c=colors[idx],
                        marker=markers[idx], label=iris.target_names[cl])

        fontsize = 20
        
        plt.xlabel("axis : {}".format(i), fontsize = fontsize)
        plt.ylabel("axis : {}".format(j), fontsize = fontsize)
        plt.legend()

        plt.title('projection of axis {} and {}'.format(i,j), fontsize = fontsize)


tree= DecisionTreeClassifier(criterion='entropy', max_depth=1)
model = tree.fit(X,y)
plt.figure(figsize=[8,8])
plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()


plot_decision_boundary(X, y, 1)


tree= DecisionTreeClassifier(criterion='entropy', max_depth=2)
model = tree.fit(X,y)
plt.figure(figsize=[8,8])
plot_tree(model, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()


plot_decision_boundary(X,y,2)


plot_decision_boundary(X,y,3)


plot_decision_boundary(X,y,4)


plot_decision_boundary(X,y,5)


plot_decision_boundary(X,y,6)


plot_decision_boundary(X,y,7)


plot_decision_boundary(X,y,8)


plot_decision_boundary(X,y,20)


plot_decision_boundary(X,y,30)


def plot_acc(X, y):
    plt.figure(figsize = (28,12))    
    for ith,axs in enumerate(combinations(range(4),2)):
        ith = ith + 1
        i,j = axs
        x1_min, x1_max = X[:, i].min() , X[:, i].max()
        x2_min, x2_max = X[:, j].min() , X[:, j].max() 
        xx1, xx2 = np.meshgrid(np.arange(x1_min-0.2, x1_max+0.2, 0.02),
                               np.arange(x2_min-0.2, x2_max+0.2, 0.02))
        
        X_std = X[:,:]
        X_std[:,i] = (X[:,i] - x1_min) / (x1_max - x1_min)
        X_std[:,j] = (X[:,j] - x2_min) / (x2_max - x2_min)
        X_std = X_std[:,[i,j]]
        
        depths = list(range(1,50))
        train_scores = []
        test_scores = []
        for d in depths:
            tree= DecisionTreeClassifier(criterion='entropy', max_depth=d)
            scores = cross_validate(tree,X_std, y, cv = 10, return_train_score = True)
            train_score = scores['train_score']
            test_score = scores['test_score']
            train_scores.append(train_score.mean())
            test_scores.append(test_score.mean())            
        plt.subplot(2,3,ith)

        fontsize = 20
        plt.ylim(-0.1,1.1)
        plt.xlabel("depth ", fontsize = fontsize)
        plt.ylabel("axis : {}".format(j), fontsize = fontsize)
        plt.legend()
        plt.plot(depths, train_scores, label = 'train')
        plt.plot(depths, test_scores, label = 'test')
        plt.legend()
        plt.xlabel('depth of tree')
        plt.ylabel('mean accuracy')

        plt.title('projection of axis {} and {}'.format(i,j), fontsize = fontsize)
        
plot_acc(X,y)        
