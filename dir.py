# import txt_to_xml,builtins
# print(dir(txt_to_xml.__builtins__))
# class resever:
#     def __init__(self,data):
#         self.data=data
#         self.index=-1
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         if self.index==len(self.data)-1:
#             raise StopIteration
#
#         self.index=self.index+1
#         return self.data[self.index]
#
# revs=resever('abcd')
#
# for rev in revs:
#     print(rev)
def reverse(data):
    for i in range(len(data)-1,-1,-1):
        print(range(len(data)-1,-1,-1),i)
        yield data[i]

for char in reverse('data'):
    print(char)
