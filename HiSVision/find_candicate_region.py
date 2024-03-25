import cv2
from PIL import Image
import numpy as np
import os
import time
import torch
from torch import nn
import torchvision.transforms as T
from main import get_args_parser as get_main_args_parser
from models import build_model
import argparse

torch.set_grad_enabled(False)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("[INFO] {}".format(device))

transform = T.Compose([
    T.Resize(800),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


# plot box by opencv
def plot_result(pil_img, prob, boxes, save_name=None, imshow=False, imwrite=False):
    opencvImage = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    LABEL = ['N/A', 'SV']
    for p, (xmin, ymin, xmax, ymax) in zip(prob, boxes):
        cl = p.argmax()
        label_text = '{}: {}%'.format(LABEL[cl], round(p[cl] * 100, 2))


        cv2.rectangle(opencvImage, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (255, 255, 0), 2)
        cv2.putText(opencvImage, label_text, (int(xmin) + 10, int(ymin) + 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 0), 2)

    if imshow:
        cv2.imshow('detect', opencvImage)
        cv2.waitKey(0)

    if imwrite:
        if not os.path.exists(img_save_path):
            os.makedirs(img_save_path)
        cv2.imwrite(f'{img_save_path}/{save_name}', opencvImage)



def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)


def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b.cpu().numpy()
    b = b * np.array([img_w, img_h, img_w, img_h], dtype=np.float32)
    return b


def load_model(model_path, args):
    model, _, _ = build_model(args)
    model.cuda()
    model.eval()
    state_dict = torch.load(model_path, map_location='cuda:0') 
    model.load_state_dict(state_dict["model"])
    model.to(device)
    print("load model sucess")
    return model



def detect(im, model, transform, prob_threshold):
    # mean-std normalize the input image (batch-size: 1)
    img = transform(im).unsqueeze(0)

    # propagate through the model
    img = img.to(device)
    start = time.time()
    outputs = model(img)

    probas = outputs['pred_logits'].softmax(-1)[0, :, :-1]
    keep = probas.max(-1).values > prob_threshold

    probas = probas.cpu().detach().numpy()
    keep = keep.cpu().detach().numpy()

    # ------------------------------------------------
    keep_max = np.sort(probas.max(-1))[-3] 
    if keep_max > prob_threshold:
        keep = probas.max(-1) >= keep_max
    else:
        pass

    # convert boxes from [0; 1] to image scales
    bboxes_scaled = rescale_bboxes(outputs['pred_boxes'][0, keep], im.size)
    end = time.time()
    return probas[keep], bboxes_scaled, end - start


def box2pos(file,boxes,resolution,w,h):
    for i in range(len(boxes)):
        xmin, ymin, xmax, ymax = boxes[i][0], boxes[i][1], boxes[i][2], boxes[i][3]
        if file.split('_')[1][0].isalpha():
            row = int(file.split('.')[0].split('_')[2])
            col = int(file.split('.')[0].split('_')[3])

            start1 = int((200 * 0.8) * (row - 1) + (ymin / (h/200))) * resolution
            end1 = int((200 * 0.8) * (row - 1) + (ymax / (h/200))) * resolution
            start2 = int((200 * 0.8) * (col - 1) + (xmin / (w/200))) * resolution
            end2 = int((200 * 0.8) * (col - 1) + (xmax / (w/200))) * resolution
            with open(pos_save_path + "\\candicate_inter.txt", 'a+') as f:
                f.write(f"{file.split('_')[0]} {start1} {end1} {file.split('_')[1]} {start2} {end2} \n")
        else:
            row = int(file.split('.')[0].split('_')[1])
            col = int(file.split('.')[0].split('_')[2])

            start1 = int((160 * 0.8) * (row - 1) + (ymin / (h/160))) * resolution
            end1 = int((160 * 0.8) * (row - 1) + (ymax / (h/160))) * resolution
            start2 = int((160 * 0.8) * (col - 1) + (xmin / (w/160))) * resolution
            end2 = int((160 * 0.8) * (col - 1) + (xmax / (w/160))) * resolution
            with open(pos_save_path + "\\candicate_intra.txt", 'a+') as f:
                f.write(f"{file.split('_')[0]} {start1} {end1} {start2} {end2} \n")



if __name__ == "__main__":


    main_args = get_main_args_parser().parse_args()
    model_path = main_args.model
    img_path = main_args.img
    output_path = main_args.output

    dfdetr = load_model(model_path, main_args)
    resolution = 50000
    img_load_path = img_path
    files = os.listdir(img_load_path)
    img_save_path = output_path
    pos_save_path = os.path.join(output_path,"candicate_SV_region")
    if not os.path.exists(pos_save_path):
        os.makedirs(pos_save_path)
    threshold = 0.85

    for file in files:
        img_path = os.path.join(img_load_path, file) 
        im = Image.open(img_path)
        w,h = im.size
        if w < 20 or h < 20:
            continue
        print(file)
        scores, boxes, waste_time = detect(im, dfdetr, transform,threshold)
        if scores.any() > threshold:
            box2pos(file,boxes,resolution,w,h)
            plot_result(im, scores, boxes, save_name=file, imshow=False, imwrite=False)
        print(" [INFO] {} time: {} done.".format(file, waste_time))
