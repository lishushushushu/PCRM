import pandas as pd
df1 = pd.read_csv('./original_data/train_log.csv')
df2 = pd.read_csv('./original_data/test_log.csv')
df3 = pd.read_csv('./original_data/test_truth.csv')
df4 = pd.read_csv('./original_data/train_truth.csv')
left = pd.concat([df1,df2])
right = pd.concat([df3,df4])
df5 = pd.merge(left,right,on=['enroll_id'],how='left')

df5.to_csv('./row_data/all_original_data.csv',index=0,header=True)

#print(df1)