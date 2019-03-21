import os

face_name = []

def change_name(name_list):
    new_name_list = []
    for name in name_list:
        temp_name = 'a'
        for letter in name:
            if letter != '_':
                temp_name = temp_name + letter
            else:
                temp_name = temp_name + ' '
        new_name_list.append(temp_name[1:])
    return new_name_list

with open('face_name.txt', 'r') as f:
	for i in f.readlines():
		face_name.append(i.strip().split(':')[1])

new_face_name = change_name(face_name)  

with open('face_name_on_line.txt', 'w') as f:
	for index, item in enumerate(new_face_name):
		f.write('{}:{}\n'.format(index, item))
