import pandas as pd
#读取数据
data = pd.read_csv('./row_data/all_original_data.csv')

#set是对元素进行去重并按照从小到大进行排序，list是将其转化为列表形式
data_enroll = list(set(list(data['enroll_id'])))

video_action = ['seek_video','play_video','pause_video','stop_video','load_video']
problem_action = ['problem_get','problem_check','problem_save','reset_problem','problem_check_correct', 'problem_check_incorrect']
forum_action = ['click_forum','create_thread','create_comment','delete_thread','delete_comment','close_forum']
courseware_action = ['click_courseware','close_courseware']

#根据enroll_id统计该学生活动的总次数，并命名该列为all#count
all_num = data.groupby('enroll_id').count()[['action']]
all_num.columns = ['all#count']

#对每个活动进行遍历并统计次数
for a in video_action + problem_action + forum_action + courseware_action:
    action_ = (data['action'] == a).astype(int)
    data[a+'#num'] = action_
    action_num = data.groupby('enroll_id').sum()[[a+'#num']]
    all_num = pd.merge(all_num, action_num, left_index=True, right_index=True)

enroll_info = data[['username','course_id','enroll_id','truth']].drop_duplicates()

enroll_info.index = enroll_info['enroll_id']
del enroll_info['enroll_id']

all_num = pd.merge(all_num, enroll_info, left_index=True, right_index=True)

all_num['complete'] = all_num['truth'].map(lambda x: 1 if x == 0 else 0)
del all_num['truth']

all_num.loc[data_enroll].to_csv('row_data/all_features.csv')






