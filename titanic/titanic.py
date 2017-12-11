#coding=utf8
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import sklearn.preprocessing as preprocessing
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import linear_model
from sklearn.learning_curve import learning_curve
from sklearn import cross_validation, metrics
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
import pprint

def print_best_score(gsearch, param_test):
    # 输出best score
    print("Best score: %0.3f" % gsearch.best_score_)
    print("Best parameters set:")
    # 输出最佳的分类器到底使用了怎样的参数
    best_parameters = gsearch.best_estimator_.get_params()
    for param_name in sorted(param_test.keys()):
        print("\t%s: %r" % (param_name, best_parameters[param_name]))
# 用sklearn的learning_curve得到training_score和cv_score，使用matplotlib画出learning curve
def plot_learning_curve(estimator, title, X, y, ylim=None, cv=10, n_jobs=1,
                        train_sizes=np.linspace(.05, 1., 20), verbose=0, plot=True):
    """
    画出data在某模型上的learning curve.
    参数解释
    ----------
    estimator : 你用的分类器。
    title : 表格的标题。
    X : 输入的feature，numpy类型
    y : 输入的target vector
    ylim : tuple格式的(ymin, ymax), 设定图像中纵坐标的最低点和最高点
    cv : 做cross-validation的时候，数据分成的份数，其中一份作为cv集，其余n-1份作为training(默认为3份)
    n_jobs : 并行的的任务数(默认1)
    """

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, verbose=verbose)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u"训练样本数")
        plt.ylabel(u"得分")
        plt.gca().invert_yaxis()
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std,
                         alpha=0.1, color="b")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std,
                         alpha=0.1, color="r")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")

        plt.legend(loc="best")

        plt.draw()
        plt.gca().invert_yaxis()
        plt.show()

    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff

def feature_engine():
    #读取训练集和测试集合拼接进行数据预处理
    train_df = pd.read_csv("train.csv")
    test_df = pd.read_csv("test.csv")

    train_df["trainortest"]="train" #添加一列区分训练集合还是测试集合
    test_df["trainortest"]="test"

    y_train = train_df.pop("Survived")
    all_df = pd.concat((train_df, test_df))

    all_df.pop("Name") #暂时无用
    all_df.pop("Ticket")#暂时无用
    #类别型进行处理
    all_df = pd.get_dummies(all_df, columns=['Pclass'])
    all_df = pd.get_dummies(all_df, columns=['Sex'])
    all_df = pd.get_dummies(all_df, columns=['Embarked'],dummy_na=True) #将null 也进行处理
    #年龄缺失与否单独做了一个特征
    all_df.loc[all_df.Age.isnull() == True, "ageNull"] = 1;
    all_df.loc[all_df.Age.isnull() == False, "ageNull"] = 0;

    all_df.loc[(all_df.Age.isnull()), "Age"] = all_df.Age.mean()#这里采用均值填充
    all_df.loc[(all_df.Fare.isnull()), "Fare"] = all_df.Fare.mean()  # 这里采用均值填充

    all_df.loc[all_df.Cabin.isnull() == True, "Cabinnull"] = 1;
    all_df.loc[all_df.Cabin.isnull() == False, "Cabinnull"] = 0;
    all_df.pop("Cabin")

    #数值型归一化处理,简单的处理
    scaler = preprocessing.StandardScaler()
    age_scale_param = scaler.fit(all_df[['Age']])
    all_df['Age_scaled'] = scaler.fit_transform(all_df[['Age']], age_scale_param)

    age_scale_param = scaler.fit(all_df[['Fare']])
    all_df['Fare_scaled'] = scaler.fit_transform(all_df[['Fare']], age_scale_param)

    all_df.pop("Age")
    all_df.pop("Fare")

    x_train = all_df[all_df.trainortest=='train']
    x_test = all_df[all_df.trainortest=='test']
    x_train.pop("trainortest")
    x_test.pop("trainortest")

    return x_train ,x_test ,y_train
def model_select(x_train,y_train):
    print("start to train")
    X=x_train.values
    y=y_train.values

    parameters = {'penalty': ('l1', 'l2'),
                  'C': [0.01,0.1,1,10,20]
                  }
    estimator = linear_model.LogisticRegression(C=1.0, penalty='l1', tol=1e-6)
    gsearch = GridSearchCV(estimator, param_grid=parameters, scoring='roc_auc', cv=10)
    gsearch.fit(X=X,y=y )
    print(gsearch.grid_scores_, gsearch.best_params_, gsearch.best_score_)
    print_best_score(gsearch, parameters)

    return  linear_model.LogisticRegression(C=0.1, penalty='l1', tol=1e-6)


def model_select_xgb(x_train,y_train):
    print("start to train")
    X=x_train.values
    y=y_train.values
    parameters = {
        #'eta':[0.01,0.015,0.025,0.05,0.1],
        'gamma':[0.7],
        'max_depth': [9],
        'min_child_weight':[3],
        'subsample':[0.8],
        'colsample_bytree':[0.8],
        #'Lambda':[0.01,0.05,0.1,1.0]
    }
    gsearch1 = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=5,
                                                    min_child_weight=1, gamma=0, subsample=0.8, colsample_bytree=0.8,
                                                    objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                    seed=27),
                            param_grid=parameters, scoring='roc_auc', n_jobs=4, iid=False, cv=5)

    gsearch1.fit(X=X,y=y )
    print("-----------------------")
    print(gsearch1.grid_scores_)
    print("-----------------------")
    print(gsearch1.best_params_)
    print("-----------------------")
    print(gsearch1.best_score_)

    print_best_score(gsearch1, parameters)


if __name__ == "__main__":
    print("start to go ")
    feature_engine()
    x_train,x_test,y_train=feature_engine()
    x_train.pop('PassengerId')
    x_test_PassengerId=x_test.pop('PassengerId')

    print(type(x_test_PassengerId))

    print(x_train.head(3))
    print("====================")
    print(y_train.head(3))
    print("====================")
    print(x_test.head(3))
    print("====================")
    print(x_test_PassengerId.head(3))
    #model_select_xgb(x_train=x_train,y_train=y_train)

    clf=XGBClassifier(learning_rate=0.1, n_estimators=140, max_depth=9,
                                                    min_child_weight=25, gamma=0.7, subsample=0.8, colsample_bytree=0.8,
                                                    objective='binary:logistic', nthread=4, scale_pos_weight=1,
                                                    seed=27)

    plot_learning_curve(estimator=clf, X=x_train.values, y=y_train.values, title="lr learning")

    clf.fit(X=x_train.values, y=y_train.values)
    predictions = clf.predict(x_test.values)

    result = pd.DataFrame(
        {'PassengerId': x_test_PassengerId, 'Survived': predictions.astype(np.int32)})
    result.to_csv("logistic_regression_predictions2.csv", index=False)


    '''
    clf=model_select(x_train=x_train,y_train=y_train)
    plot_learning_curve(estimator=clf,X=x_train.values,y=y_train.values,title="lr learning")
    clf.fit(X=x_train.values,y=y_train.values)
    predictions = clf.predict(x_test.values)

    result = pd.DataFrame(
        {'PassengerId': x_test_PassengerId, 'Survived': predictions.astype(np.int32)})
    result.to_csv("logistic_regression_predictions2.csv", index=False)
    '''













