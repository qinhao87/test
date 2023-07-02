import json

# fp=open("json_test.json",encoding="utf-8")
# data=json.load(fp)
# #if data['shapes'] is not None:
# print(data[0])
filename='test.json'

with open(filename) as f_obj:
    number=json.load(f_obj)
    print(len(number))