

import pandas as pd

root = '/Users/sungheui/Downloads/test_answer_sample.csv'

submission = pd.read_csv(root, encoding = 'utf-8', index_col = 0)
submission['answer value'] = 1
submission.to_csv('./submission_VIT_small'+'_'+str(1)+'.csv', index = False)
