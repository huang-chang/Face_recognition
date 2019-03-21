import os

face_name = []

for directory, folders, files in os.walk('/data1/facenet-master/face_565_8_24_selected'):
    face_name.extend(folders)
    face_name.sort()
    break

with open('face_name.txt', 'w') as f:
	for index, item in enumerate(face_name):
		f.write('{}:{}\n'.format(index,item))
