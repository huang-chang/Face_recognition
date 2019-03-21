import os

face_number = []

for directory, folders, files in os.walk('/data1/facenet-master/face_565_8_24_selected'):
    if len(folders) == 0:
        face_number.append([directory.split('/')[-1], len(files)])
face_number.sort()

with open('face_number.txt', 'w') as f:
	for index, item in enumerate(face_number):
		f.write('{}:{}\n'.format(item[0],item[1]))
