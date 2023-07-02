import os
from txt_to_xml import to_xml
txt_path='E:\data/test/new_txt'
xml_path='E:\data/test/test_xml'
# print(os.listdir(path))
for i in os.listdir(txt_path):
    txt_dir=txt_path+'/'+i
    print(txt_dir)
    a=i.split('.')[0]+'.png'
    print(a)
    xml_dir=xml_path+'/'+i.split('.')[0]
    print(xml_dir)
    to_xml(txt_dir,xml_dir,a)
