#coding=utf8
import pandas as pd
from sklearn import preprocessing
def feature_engine():
    #读取训练集和测试集合拼接进行数据预处理
    train_df = pd.read_csv("./input_data/train.csv")
    test_df = pd.read_csv("./input_data/test.csv")

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

if __name__ == "__main__":
    print("feature engine start ")


