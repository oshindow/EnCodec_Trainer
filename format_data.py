import os
# prosody
data = {}
prosody_root = '/data2/junchuan/VALLE-X/prosody_vec/'
for file in os.listdir(prosody_root):
    if '.npy' in file:
        # 4297_13009_000046_000001.npy 
        uid = file[:-4]
        # print(uid)
        filepath = os.path.join(prosody_root, file)
        data[uid] = [filepath]


prosody_root = '/data2/junchuan/VALLE-X/timbre_vec/'
for file in os.listdir(prosody_root):
    if '.npy' in file:
        # 4297_13009_000046_000001.npy 
        uid = file[:-4]
        # print(uid)
        filepath = os.path.join(prosody_root, file)
        data[uid].append(filepath)

target_root = '/data2/xintong/LibriTTS_encodec_continuous/train-clean-100'
# 911_128684_000020_000004.npy
for root, dirs, files in os.walk(target_root):
    for file in files:
        if '.npy' in file:
            filepath = os.path.join(root, file)
            uid = file[:-4]
            data[uid].append(filepath)

with open('train.txt', 'w', encoding='utf8') as output:
    for key, item in data.items():
        output.write(key + '|' + item[0] + '|' + item[1] + '|' + item[2] + '\n')

