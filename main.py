import os

import videoDetect
import pandas as pd

folder_path = 'D:/hackaton_video/after'

def get_files_in_folder(folder_path):
    file_list = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_list.append(os.path.join(file))
    return file_list

files = get_files_in_folder(folder_path)

num = 0

drunk_df = []
for file in files:
    input_file = folder_path + '/' + file
    drunk_df.append(videoDetect.detect_to_csv(input_file,'drunk'))
    print(input_file+' --- finished')

df = pd.DataFrame(drunk_df)
df.to_csv('drunk.csv',index=False)