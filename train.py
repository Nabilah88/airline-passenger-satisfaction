
import pickle

import pandas as pd
import xgboost as xgb
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score


# In[2]:


train_path = 'data/train.csv'
test_path = 'data/test.csv'
output_path = 'model.bin'




def load_and_clean(path):
    df = pd.read_csv(path)
    df = df.drop(columns=['Unnamed: 0', 'id'])
    df.columns = df.columns.str.lower().str.replace(' ', '_').str.replace('/', '_').str.replace('-', '_')

    for col in df.select_dtypes(include=['object', 'string']).columns:
        df[col] = df[col].str.lower().str.replace(' ', '_')

    df['arrival_delay_in_minutes'] = df['arrival_delay_in_minutes'].fillna(df['departure_delay_in_minutes'])
    df['satisfaction'] = (df['satisfaction'] == 'satisfied').astype(int)
    return df



df_train = load_and_clean(train_path)
y_train = df_train['satisfaction'].values
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(df_train.drop(columns=['satisfaction']).to_dict(orient='records'))



model = xgb.XGBClassifier(n_estimators=100, max_depth=7, learning_rate=0.3, random_state=1, n_jobs=-1)
model.fit(X_train, y_train)




df_test = load_and_clean(test_path)
y_test = df_test['satisfaction'].values
X_test = dv.transform(df_test.drop(columns=['satisfaction']).to_dict(orient='records'))
print('test AUC:', roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]))




with open(output_path, 'wb') as f_out:
    pickle.dump((dv, model), f_out)

print(f'model saved to {output_path}')





