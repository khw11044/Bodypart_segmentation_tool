import cv2
import numpy as np
import os
import copy
import argparse

# python drow.py --img_dir=img/side --seg_dir=seg/side --save_dir=save --folder_name=bad_side_swing1232    
# python drow.py --img_dir=img/front --seg_dir=seg/front --save_dir=save/front --folder_name=bad_front_swing1404

def get_arguments():
    parser = argparse.ArgumentParser(description="segmentationtool")

    # default = 이미지 dirdad
    parser.add_argument("--img_dir", type=str, default='./img', help="path of input image folder.")
    #폴더이름
    parser.add_argument("--folder_name", type=str, default='bad_front_swing1418', help="")
    # default = 세그 dir
    parser.add_argument("--seg_dir", type=str, default='./seg', help="path of output segmentation folder.")
    # default = 저장시킬 dir
    parser.add_argument("--save_dir", type=str, default='./result', help="path of save segmentation folder.")

    return parser.parse_args()

body_part = [[127,127,127,127,'Head'],
             [76,0,0,255,'Torso'],
             [226,0,255,255,'Upper_left_arm'],
             [105,255,0,255,'Lower_left_arm'],
             [179, 255,255,0,'Left_hand'],
             [151, 0,127,255,'Upper_right_arm'],
             [67,255,0,127,'Lower_right_arm'],
             [113,0,127,127,'Right_hand '],
             [150,0,255,0,'Upper_left_leg'],
             [202,127,255,127,'Lower_left_leg'],
             [240,127,255,255,'Left_foot'],
             [29,255,0,0,'Upper_right_leg'],
             [104,255,127,0,'Lower_right_leg'],
             [165,127,127,255,'Right_foot'],
             [255,255,255,255,'Club']]

class StackUnDo:
    def __init__(self, max_length=10):
        self.stack = []
        self.max_length = max_length
        self.zero = []

    def __len__(self):
        return len(self.stack)

    def __getitem__(self, idx):
        return self.stack[idx]

    def pop(self):
        if len(self.stack) ==1:
            self.zero = self.stack[0]

        if len(self.stack) ==0:
            x = self.zero
        else:
            x = self.stack.pop(-1)

        return x

    def append(self, item):
        if len(self.stack) >= self.max_length:
            self.stack.pop(0)
        self.stack.append(item)

    def __str__(self):
        return self.stack.__str__()

def img_mash(seg,imag,i):
    global bb

    gray = cv2.cvtColor(seg, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(gray, body_part[i][0], body_part[i][0])

    human_t = np.where(gray>0,True,False)
    human_t = np.expand_dims(human_t, axis=2)
    human_t = np.concatenate((human_t, human_t, human_t), axis=2)

    human = np.where(mask>0,True,False)
    human = np.expand_dims(human,axis=2)
    human = np.concatenate((human,human,human),axis=2)

    back_t = np.where(gray==0,True,False)
    back_t = np.expand_dims(back_t, axis=2)
    back_t = np.concatenate((back_t, back_t, back_t), axis=2)

    back = np.where(mask==0,True,False)
    back = np.expand_dims(back,axis=2)
    back = np.concatenate((back,back,back),axis=2)

    f = seg*human_t
    e = imag*back_t

    a = imag*back
    b = seg*human
    if bb == True:
        c = cv2.addWeighted(imag, 0.8, seg, 0.2, 0)
    else:
        c = cv2.addWeighted(imag, 0.5, seg, 0.5, 0)
    d = seg*back
    bler = cv2.addWeighted(imag, 0.05, b, 0.95, 0)*human +a

    return c , mask, bler , b , d

# 중심점
def center(mask):
    global H,W
    contro ,_ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    try:
        cnt = contro[0]
        (cen_x,cen_y) ,_ = cv2.minEnclosingCircle(cnt)

        cen_x = int(cen_x)
        cen_y = int(cen_y)
        if cen_y + 250 > H:
            cen_y = H - 250
        if cen_y - 250 < 0:
            cen_y = 250

        if cen_x + 350 > W:
            cen_x = W-350
        if cen_x - 350 < 0:
            cen_x = 350
        cen_x = int(cen_x)
        cen_y = int(cen_y)
    except:
        cen_x = True
        cen_y = True
    return cen_x, cen_y

# Mouse Callback함수
class mous():
    def __init__(self, cen_x, cen_y, body_part,i,mask):
        self.cen_x = cen_x
        self.cen_y = cen_y
        self.body_part = body_part[i]
        self.mask = mask

    def draw_circle(self,event, x,y, flags, param):
        global ix,iy, drawing, modem ,undo, pre_draw, size, poly_point, pre_poly ,xx2, yy2, shit
        xx = x
        yy = y
        x -= 280
        if self.cen_y == True:
            x -= 420
        else:
            x += self.cen_x - 350
            y += self.cen_y - 250

        if pol == True and pol_e == True:

            if pre_draw != drawing:
                pre_draw = not pre_draw
                if drawing == True:
                    undo.append(copy.deepcopy(seg_save[j]))
                    undo_num.append(copy.deepcopy(j))
            if event == cv2.EVENT_LBUTTONDOWN: #마우스를 누른 상태
                drawing = True
                ix, iy = x,y
            elif event == cv2.EVENT_MOUSEMOVE: # 마우스 이동
                if drawing == True:            # 마우스를 누른 상태 일경우
                    if mode == True:
                        cv2.circle(seg_save[j],(x,y),draw_size(size),(self.body_part[1],self.body_part[2],self.body_part[3]),-1)
                    else:
                        if size == True:
                            if shit == True:
                                cv2.rectangle(e,(x-int(draw_size(size)/2),y-int(draw_size(size)/2)),(x+int(draw_size(size)/2),y+int(draw_size(size)/2)),(0,0,0) ,-1)
                                seg_save[j] = cv2.add(e,f)
                            else:
                                cv2.rectangle(seg_save[j], (x - int(draw_size(size) / 2), y - int(draw_size(size) / 2)),
                                              (x + int(draw_size(size) / 2), y + int(draw_size(size) / 2)), (0, 0, 0),
                                              -1)


                        else:
                            if shit == True:
                                cv2.circle(e,(x,y),draw_size(size),(0,0,0) ,-1)
                                seg_save[j] = cv2.add(e, f)
                            else:
                                cv2.circle(seg_save[j], (x, y), draw_size(size), (0, 0, 0), -1)

            elif event == cv2.EVENT_LBUTTONUP:
                drawing = False             # 마우스를 때면 상태 변경
                if mode == True:
                    cv2.circle(seg_save[j],(x,y),draw_size(size),(self.body_part[1],self.body_part[2],self.body_part[3]),-1)
                else:
                    if size == True:
                        if shit == True:
                            cv2.rectangle(e, (x - int(draw_size(size) / 2), y - int(draw_size(size) / 2)),
                                          (x + int(draw_size(size) / 2), y + int(draw_size(size) / 2)), (0, 0, 0), -1)
                            seg_save[j] = cv2.add(e, f)
                        else:
                            cv2.rectangle(seg_save[j], (x - int(draw_size(size) / 2), y - int(draw_size(size) / 2)),
                                          (x + int(draw_size(size) / 2), y + int(draw_size(size) / 2)), (0, 0, 0),
                                          -1)
                    else:
                        if shit == True:
                            cv2.circle(e, (x, y), draw_size(size), (0, 0, 0), -1)
                            seg_save[j] = cv2.add(e, f)
                        else:
                            cv2.circle(seg_save[j], (x, y), draw_size(size), (0, 0, 0), -1)

        elif pol == False:
            if event == cv2.EVENT_LBUTTONDOWN: #마우스를 누른 상태
                ix, iy = x,y
                cv2.circle(mask_0, (xx, yy), 1,
                           (225, 225, 127), -1)
                poly_point.append([x,y])

                if len(poly_point) == 1:
                    undo.append(copy.deepcopy(seg_save[j]))
                    undo_num.append(copy.deepcopy(j))
                else:
                    cv2.line(mask_0,(xx,yy),(xx2,yy2),(self.body_part[1],self.body_part[2],self.body_part[3]),1)
                    cv2.line(mask_1, (xx, yy), (xx2, yy2), (self.body_part[1], self.body_part[2], self.body_part[3]), 1)
                xx2 = xx
                yy2 = yy
        else:
            if event == cv2.EVENT_LBUTTONDOWN: #마우스를 누른 상태
                ix, iy = x,y
                cv2.circle(mask_0, (xx, yy), 1,
                           (225, 225, 127), -1)
                poly_point.append([x,y])

                if len(poly_point) == 1:
                    undo.append(copy.deepcopy(seg_save[j]))
                    undo_num.append(copy.deepcopy(j))
                else:
                    cv2.line(mask_0,(xx,yy),(xx2,yy2),(0,0,0),1)
                    cv2.line(mask_1, (xx, yy), (xx2, yy2), (0, 0, 0), 1)
                xx2 = xx
                yy2 = yy




def draw_size(size):
    if size == True:
        a = 2
    else:
        a =6
    return a




drawing = False #Mouse가 클릭된 상태 확인용
pre_draw = False
mode = True # True이면 그리기, false면 지우기
shit = True # True이면 불투명, false면 투명
bb = True
ix,iy = -1,-1
H, W = 0, 0
size = True
pol = False
pol_e = True
pre_poly = pol
poly_point = []
poly_save = []
xx2 = -1
yy2 = -1
# 폴더 이름 1개, SEG 경로 , img 경로
if __name__ == '__main__':
    args = get_arguments()
    folder_name = args.folder_name
    input_img_dir = os.path.join(args.img_dir, folder_name)
    input_seg_dir = os.path.join(args.seg_dir, folder_name)

    input_list=sorted(os.listdir(input_img_dir))
    len_1 = len(input_list)

    output_dir = os.path.join(args.save_dir, folder_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    j=0
    pre_1 =0
    imag = cv2.imread(os.path.join(input_img_dir, input_list[j]))
    H,W,_ = imag.shape

    seg_save = []
    for p in range(len_1):
        if os.path.exists(os.path.join(output_dir, input_list[p])):
            seg_path = os.path.join(output_dir, input_list[p])
        else:
            seg_path = os.path.join(input_seg_dir, input_list[p])

        seg_save.append(cv2.imread(seg_path))

    undo = StackUnDo(max_length=20)
    undo_num = StackUnDo(max_length=20)
    undo.append(copy.deepcopy(seg_save[j]))
    undo_num.append(copy.deepcopy(j))
    i = 0
    pre = 0

    c , mask,b , e, f = img_mash(seg_save[j],imag,i)
    name = 'seg'
    cv2.namedWindow ( name, cv2.WINDOW_NORMAL)

    cen_x , cen_y = center(mask)
    mous_1 = mous(cen_x, cen_y, body_part,i,mask)
    cv2.setMouseCallback(name,mous_1.draw_circle)


    while True:
        if len(poly_point) == 0:
            if pre_1 != j:
                imag = cv2.imread(os.path.join(input_img_dir, input_list[j]))
                c, mask ,b ,e,f= img_mash(seg_save[j],imag,i)
                cen_x , cen_y = center(mask)

            pre_1 = j

            if pre != i :
                c, mask,b,e,f = img_mash(seg_save[j],imag,i)
                cen_x , cen_y = center(mask)

            pre = i
            c, mask, b , e, f = img_mash(seg_save[j], imag, i)
            if bb == True:
                mask_0 = cv2.addWeighted(imag, 0.7, b, 0.3, 0)
            else:
                mask_0 = b

            mask_1 = c
            mask_2 = mask_0
            mask_3 = mask_1


            if cen_y != True:
                mask_0 = mask_0 [cen_y - 250:cen_y + 250, cen_x - 350:cen_x + 350]
                mask_1 = mask_1[cen_y - 250:cen_y + 250, cen_x - 350:cen_x + 350]
                empty_box = np.zeros((100, 280, 3), np.uint8)
                empty_box = cv2.rectangle(empty_box, (0, 0), (500, 100), (255, 255, 255), -1)
                if shit == True:
                    in_all_mask = cv2.resize(mask_3, (280, 400))
                else:
                    in_all_mask = cv2.resize(mask_2, (280, 400))
                left_img = cv2.vconcat([empty_box, in_all_mask])
                mask_0 = cv2.hconcat([left_img, mask_0])
                mask_1 = cv2.hconcat([left_img, mask_1])


            else :
                empty_box = np.zeros((100, 700, 3), np.uint8)
                empty_box = cv2.rectangle(empty_box, (0, 0), (H, 100), (255, 255, 255), -1)
                if shit == True:
                    in_all_mask = cv2.resize(mask_3, (700, H-100))
                else:
                    in_all_mask = cv2.resize(mask_2, (700, H - 100))
                left_img = cv2.vconcat([empty_box, in_all_mask])
                mask_0 = cv2.hconcat([left_img, mask_0])
                mask_1 = cv2.hconcat([left_img, mask_1])


        #왼쪽 text
        cv2.putText(mask_0, str(j + 1), (10, 30), 2, 1, (0, 0, 0), 2)
        cv2.putText(mask_0, body_part[i][4], (10, 60), 2, 1, (0, 0, 0), 2)
        if pol == True and pol_e == True:
            if mode == True and size == True:
                cv2.putText(mask_0, 'Small Pen', (10, 90), 2, 1, (0, 0, 0), 2)
            elif mode == True and size == False:
                cv2.putText(mask_0, 'Big Pen', (10, 90), 2, 1, (0, 0, 0), 2)
            elif mode == False and size == False:
                cv2.putText(mask_0, 'Big Erase', (10, 90), 2, 1, (0, 0, 0), 2)
            elif mode == False and size == True:
                cv2.putText(mask_0, 'Small Erase', (10, 90), 2, 1, (0, 0, 0), 2)
        elif pol == False :
            cv2.putText(mask_0, 'poly', (10, 90), 2, 1, (0, 0, 0), 2)
        else:
            cv2.putText(mask_0, 'poly Erase', (10, 90), 2, 1, (0, 0, 0), 2)

        cv2.putText(mask_1, str(j + 1), (10, 30), 2, 1, (0, 0, 0), 2)
        cv2.putText(mask_1, body_part[i][4], (10, 60), 2, 1, (0, 0, 0), 2)
        if pol == True and pol_e == True:
            if mode == True and size == True:
                cv2.putText(mask_1, 'Small Pen', (10, 90), 2, 1, (0, 0, 0), 2)
            elif mode == True and size == False:
                cv2.putText(mask_1, 'Big Pen', (10, 90), 2, 1, (0, 0, 0), 2)
            elif mode == False and size == False:
                cv2.putText(mask_1, 'Big Erase', (10, 90), 2, 1, (0, 0, 0), 2)
            elif mode == False and size == True:
                cv2.putText(mask_1, 'Small Erase', (10, 90), 2, 1, (0, 0, 0), 2)
        elif pol == False:
            cv2.putText(mask_1, 'poly', (10, 90), 2, 1, (0, 0, 0), 2)
        else:
            cv2.putText(mask_1, 'poly Erase', (10, 90), 2, 1, (0, 0, 0), 2)



        k = cv2.waitKey(1) & 0xFF
        if k == ord('3'):    # 펜/지우개
            if pol == False:
                if len(poly_point) > 1:
                    cv2.fillPoly(seg_save[j], [np.array(poly_point, np.int32)],
                                 (body_part[i][1], body_part[i][2], body_part[i][3]))

                poly_point = []
            elif pol_e == False:
                if len(poly_point) > 1:
                    if shit == True:
                        cv2.fillPoly(e, [np.array(poly_point, np.int32)],
                                     (0, 0, 0))
                        seg_save[j] = cv2.add(e, f)
                    else:
                        cv2.fillPoly(seg_save[j], [np.array(poly_point, np.int32)],
                                     (0, 0, 0))
                poly_point = []
            mode = True
            pol = True
            pol_e = True
            size = not size
        elif k == 27:        # esc를 누르면 종료
            cv2.destroyAllWindows()
            break


        elif k == ord('d') :      # d / 다음 파트
            if pol == False:
                if len(poly_point) > 1:
                    cv2.fillPoly(seg_save[j], [np.array(poly_point, np.int32)],
                                 (body_part[i][1], body_part[i][2], body_part[i][3]))

                poly_point = []
            elif pol_e == False:
                if len(poly_point) > 1:
                    if shit == True:
                        cv2.fillPoly(e, [np.array(poly_point, np.int32)],
                                     (0, 0, 0))
                        seg_save[j] = cv2.add(e, f)
                    else:
                        cv2.fillPoly(seg_save[j], [np.array(poly_point, np.int32)],
                                     (0, 0, 0))
                poly_point = []
           
            if i == 14:
                i-=15
            i+=1
        elif k == ord('a') :     # a / 이전 파트
            if pol == False:
                if len(poly_point) > 1:
                    cv2.fillPoly(seg_save[j], [np.array(poly_point, np.int32)],
                                 (body_part[i][1], body_part[i][2], body_part[i][3]))

                poly_point = []
            elif pol_e == False:
                if len(poly_point) > 1:
                    if shit == True:
                        cv2.fillPoly(e, [np.array(poly_point, np.int32)],
                                     (0, 0, 0))
                        seg_save[j] = cv2.add(e, f)
                    else:
                        cv2.fillPoly(seg_save[j], [np.array(poly_point, np.int32)],
                                     (0, 0, 0))
                poly_point = []
           
            if i == 0:
                i+=15
            i-=1
        elif k == ord('s'):      # s / 이전 이미지
            if pol == False:
                if len(poly_point) > 1:
                    cv2.fillPoly(seg_save[j], [np.array(poly_point, np.int32)],
                                 (body_part[i][1], body_part[i][2], body_part[i][3]))

                poly_point = []
            elif pol_e == False:
                if len(poly_point) > 1:
                    if shit == True:
                        cv2.fillPoly(e, [np.array(poly_point, np.int32)],
                                     (0, 0, 0))
                        seg_save[j] = cv2.add(e, f)
                    else:
                        cv2.fillPoly(seg_save[j], [np.array(poly_point, np.int32)],
                                     (0, 0, 0))
                poly_point = []
            
            if j == 0:
                j+=len_1
            j-=1
        elif k ==ord('w'):       # w / 다음 이미지
            if pol == False:
                if len(poly_point) > 1:
                    cv2.fillPoly(seg_save[j], [np.array(poly_point, np.int32)],
                                 (body_part[i][1], body_part[i][2], body_part[i][3]))

                poly_point = []
            elif pol_e == False:
                if len(poly_point) > 1:
                    if shit == True:
                        cv2.fillPoly(e, [np.array(poly_point, np.int32)],
                                     (0, 0, 0))
                        seg_save[j] = cv2.add(e, f)
                    else:
                        cv2.fillPoly(seg_save[j], [np.array(poly_point, np.int32)],
                                     (0, 0, 0))
                poly_point = []
           
            if j == len_1-1:
                j-=len_1
            j+=1
        elif k == ord('4'):
            if pol == False:
                if len(poly_point) > 1:
                    cv2.fillPoly(seg_save[j], [np.array(poly_point, np.int32)],
                                 (body_part[i][1], body_part[i][2], body_part[i][3]))

                poly_point = []
            elif pol_e == False:
                if len(poly_point) > 1:
                    if shit == True:
                        cv2.fillPoly(e, [np.array(poly_point, np.int32)],
                                     (0, 0, 0))
                        seg_save[j] = cv2.add(e, f)
                    else:
                        cv2.fillPoly(seg_save[j], [np.array(poly_point, np.int32)],
                                     (0, 0, 0))
                poly_point = []
            mode = False
            pol = True
            pol_e = True
            size = not size
        elif k == ord('1'):
            if pol_e == False:
                if len(poly_point) > 1:
                    if shit == True:
                        cv2.fillPoly(e, [np.array(poly_point, np.int32)],
                                     (0, 0, 0))
                        seg_save[j] = cv2.add(e, f)
                    else:
                        cv2.fillPoly(seg_save[j], [np.array(poly_point, np.int32)],
                                     (0, 0, 0))
                poly_point = []
            pol = False
            pol_e = True
        elif k == 32:
            if pol == False:
                if len(poly_point) > 1:
                    cv2.fillPoly(seg_save[j], [np.array(poly_point, np.int32)],
                                 (body_part[i][1], body_part[i][2], body_part[i][3]))
                poly_point = []
            elif pol_e == False:
                if len(poly_point) > 1:
                    if shit == True:
                        cv2.fillPoly(e, [np.array(poly_point, np.int32)],
                                     (0, 0, 0))
                        seg_save[j] = cv2.add(e, f)
                    else:
                        cv2.fillPoly(seg_save[j], [np.array(poly_point, np.int32)],
                                     (0, 0, 0))
                poly_point = []
            else:
                pass
        elif k == ord('z'):      # z / 실행 취소
            j = undo_num.pop()
            seg_save[j] = undo.pop()
        elif k == ord('2'):
            if pol == False:
                if len(poly_point) > 1:
                    cv2.fillPoly(seg_save[j], [np.array(poly_point, np.int32)],
                                 (body_part[i][1], body_part[i][2], body_part[i][3]))
                poly_point = []
            pol_e = False
            pol = True
        elif k == ord('r'):
            undo.append(copy.deepcopy(seg_save[j]))
            undo_num.append(copy.deepcopy(j))
            e = np.zeros_like(e)
            seg_save[j] = cv2.add(e, f)
        elif k == ord('v'):
            bb = not bb
        elif k == ord('c'):
            shit = not shit
        pre_poly = pol


        if shit == True:
            mous_1 = mous(cen_x, cen_y, body_part, i, mask)
            cv2.namedWindow(name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(name, mous_1.draw_circle)
            cv2.imshow(name, mask_0)
        else:
            mous_1 = mous(cen_x, cen_y, body_part,i,mask)
            cv2.namedWindow ( name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(name,mous_1.draw_circle)
            cv2.imshow(name, mask_1)


    for u in range(len_1):
        path = os.path.join(output_dir,input_list[u])
        cv2.imwrite(path,seg_save[u])


