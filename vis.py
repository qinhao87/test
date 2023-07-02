pic_dir='/home/hao/下载/VOCdevkit/VOC2007/JPEGImages'
        if len(message[0].split('.')[0]) == 1 :
            pic='00000'+message[0].split('.')[0]+'.jpg'
        elif len(message[0].split('.')[0]) == 2 :
            pic='0000'+message[0].split('.')[0]+'.jpg'
        elif len(message[0].split('.')[0]) == 3 :
            pic = '000' + message[0].split('.')[0] + '.jpg'
        elif len(message[0].split('.')[0]) == 4 :
            pic = '00' + message[0].split('.')[0] + '.jpg'
        gt_class_dict['pic_txt']=os.path.join(pic_dir,pic)


        for i in range(gt_pre_iou.shape[1]):
            if gt_pre_iou[0][i]>0.5:
                rec_dict[pre_cla[i]].append(float(pre_score[i]))

                if pre_cla[i] == gt_class :
                    gt_class_dict[gt_class].append(pre_boxes[i])

        if '3.txt' in message:

            print(gt_class_dict)
            v = vis(gt_class_dict)
            v.visualization()
            break



class vis:


    def __init__(self,rect_dict):


        self.rect=rect_dict[list(rect_dict.keys())[0]]
        self.dir=rect_dict[list(rect_dict.keys())[1]]



    def visualization(self):


        img = cv.imread(self.dir)
        for rec in self.rect:
            lefttop = (int(rec[1]),int(rec[0]))
            rightdown = (int(rec[3]),int(rec[2]))
            bbox_color = (0,255,0)
            bbox_thick = 1
            cv.rectangle(img,lefttop,rightdown,bbox_color,bbox_thick)
        cv.imshow('img',img)
        cv.waitKey(0)
        cv.destroyAllWindows()
