

filename='information.txt'
i=0
with open(filename) as f_obj:
    for line in f_obj:
        i += 1
        if 'pred_bbox' in line:
            print(type(line))


        if i>15:
            break