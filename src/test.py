import os
import shutil
train_dir = "/home/weizi/workspace/misbehavior_prediction/src/output/epoch_old"
record_path = "/home/weizi/workspace/misbehavior_prediction/src/output/epoch_old/record.txt"


cnt = 0
with open(record_path, "r") as fp:
   for line in fp.readlines():
      line = line.strip('\n')
      data = line.split(" ")
      folder, label = data[0], data[1]
      folder_path = os.path.join(train_dir, folder)
      if label == "2":
         cnt += 1
         # files = os.listdir(folder_path)
         # files.sort()
         # for file in files[:-16]:
         #    file_path = os.path.join(folder_path, file)
         #    os.remove(file_path)
      # files = os.listdir(folder_path)
      # files.sort()
      # if label == "1":
      #    cnt += 1
      #    shutil.rmtree(folder_path)
print(cnt)


# files = os.listdir(train_dir)
# folders = []
# for folder in files:
#     if os.path.isdir(os.path.join(train_dir, folder)):
#         folders.append(folder)

# for folder in folders:
#     folder_path = os.path.join(train_dir, folder)
#     files = os.listdir(folder_path)
#     for file in files:
#         new_file = file.split('_')[0] + ".png"
#         os.rename(os.path.join(folder_path, file), os.path.join(folder_path, new_file))
