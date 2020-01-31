from sklearn import svm
import pandas as pd
import joblib

#Chargement du nom des colonnes et suppression des espaces
index_column = pd.read_csv('../../features.txt', header=None, sep="\n")
index_column = index_column.loc[:,0].str.strip()
# Chargement des données de train
trainX_df = pd.read_csv('../../Train/X_train.txt', sep=" ", header=None)
trainY_df = pd.read_csv('../../Train/y_train.txt', names={"id_label"}, header=None)

# Chargement des données de test
testX_df = pd.read_csv('../../Test/X_test.txt', sep=" ", header=None)
testY_df = pd.read_csv('../../Test/y_test.txt', names={"id_label"}, header=None)

# Insertion des noms des colonnes
trainX_df.columns = index_column.values
testX_df.columns = index_column.values

# Chargement des labels des activités
activity_label_df = pd.read_csv('../../activity_labels.txt', sep="\n", names={"activity"}, header=None)
activity_label_df['id_label'] = activity_label_df['activity'].str[0:2].astype('int64')
activity_label_df['activity'] = activity_label_df['activity'].str[2:]
activity_label_df['activity'] = activity_label_df['activity'].str.strip().astype('category')

# Merge
trainY_df = pd.merge(trainY_df, activity_label_df, on='id_label', how='left', sort=False)
testY_df = pd.merge(testY_df, activity_label_df, on='id_label', how='left', sort=False)

# Create your models here.
model = svm.SVC(C=16, kernel='linear')
model = model.fit(trainX_df, trainY_df['id_label'])

joblib.dump(model, "./model/SVC.joblib", compress=True)