import os
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

def get_args():
	parser = argparse.ArgumentParser()
	parser.add_argument('--model', default=None, help='the model saved path')
	parser.add_argument('--candicate', default=None, help='the candicate region saved path')
	parser.add_argument('--output', default=None, help='the output path')

	return parser.parse_args()


class LSTMClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(LSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.fc(h_n.squeeze(0))
        out = self.softmax(out)
        return out

input_size = 20
hidden_size = 50
num_classes = 9


def min_max_normalization(matrix):
    max_value = matrix.max()
    min_value =  matrix.min()
    matrix = ( matrix- min_value) / (max_value - min_value)

    return matrix

model = LSTMClassifier(input_size, hidden_size, num_classes)

args = get_args()
model_saved_path = args.model
candicate_saved_path = args.candicate
output_path = args.output


model.load_state_dict(torch.load(model_saved_path))
model.eval()
torch.set_printoptions(precision=5,sci_mode=False)


candicate_save_path = candicate_saved_path


filenames = []
all_data =[]
prediction = []
sv_list=[]


def cal_counts(mat,type):
    if type == 1:
        return np.sum(mat[:10,:10])
    elif type == 2:
        return np.sum(mat[:10,10:])
    elif type == 3:
        return np.sum(mat[10:,:10])
    elif type == 4:
        return np.sum(mat[10:,10:])



for filename in os.listdir(candicate_save_path):
    txt_path = os.path.join(candicate_save_path,filename)
    filenames.append(filename)
    matrix = np.loadtxt(txt_path)
    if matrix.shape[0] != 20 or matrix.shape[1] != 20:
        filenames = filenames[:-1]
        continue
    all_data.append(min_max_normalization(matrix))

for i in range(len(all_data)):
    new_matrix_tensor = torch.tensor(all_data[i], dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        output = model(new_matrix_tensor)

    prob, predicted_class = torch.max(output, 1)
    predicted_class = predicted_class.item()

    if predicted_class in [1,2,3,4,7,8] and prob > 0.9999 or predicted_class in [5,6] and prob > 0.99:

        if filenames[i].split('.')[0].split('_')[1][0].isalpha():

            chr1 = filenames[i].split('.')[0].split('_')[0]
            chr2 = filenames[i].split('.')[0].split('_')[1]
            pos1 = int(filenames[i].split('.')[0].split('_')[2])
            pos2 = int(filenames[i].split('.')[0].split('_')[3])

            flag = True
            for j in range(len(sv_list)):
                if chr1 == sv_list[j][0] and chr2 == sv_list[j][1] and np.abs(sv_list[j][2] - pos1) < 10 and np.abs(sv_list[j][3] - pos2) < 10:
                    flag = False
                if chr1 == sv_list[j][0] and chr2 == sv_list[j][1] and np.abs(sv_list[j][3] - pos1) < 3 and np.abs(sv_list[j][2] - pos2) < 3:
                    flag = False
            if flag:
                sv_list.append([chr1,pos1,chr2,pos2,predicted_class])

        else:
            chr = filenames[i].split('.')[0].split('_')[0]
            pos1 = int(filenames[i].split('.')[0].split('_')[1])
            pos2 = int(filenames[i].split('.')[0].split('_')[2])

            if pos1 > pos2:
                tmp = pos1
                pos1 = pos2
                pos2 = tmp

            flag = True

            for j in range(len(sv_list)):
                if chr == sv_list[j][0] and np.abs(sv_list[j][1] - pos1) < 10 and np.abs(sv_list[j][2] - pos2) < 10:
                    flag = False
                    continue
                if chr == sv_list[j][0] and np.abs(sv_list[j][2] - pos1) < 3 and np.abs(sv_list[j][1] - pos2) < 3:
                    flag = False
                    continue
                if chr == sv_list[j][0] and np.abs(sv_list[j][1] - pos1) < 5 and np.abs(sv_list[j][2] - pos2) <= 15:
                    mat1 = all_data[i]
                    mat2 = np.loadtxt(sv_list[j][4])
                    mat1_counts = cal_counts(mat1,predicted_class)
                    mat2_counts = cal_counts(mat2,predicted_class)
                    if mat1_counts > mat2_counts:
                        sv_list[j][1] = pos1
                        sv_list[j][2] = pos2
                    else:
                        flag = False
            if flag:
                sv_list.append([chr, pos1, pos2,predicted_class,os.path.join(txt_save_path,filenames[i])])


if len(sv_list) == 0:
    with open(os.join(output_path , 'SV_list.txt'), 'w') as f:
        for item in sv_list:
            f.write("%s\n" % item)
else:
    if sv_list[0][2][0].isalpha():
        with open(os.path.join(output_path, 'Inter_SV_list.txt'), 'w') as f:
            f.write('chrA' + "\t" + 'breakpoint_A' + "\t" + 'chrB' + "\t" + 'breakpoint_B' + "\t" + "SV_type" + "\n")
            for item in sv_list:
                f.write(item[0] + '\t' + str(int(item[1]) * 50000) + '\t' + item[2] + '\t' + str(int(item[3]) * 50000) + '\t' + str(item[4]))

    else:
        with open(os.path.join(output_path, 'Intra_SV_list.txt'), 'w') as f:
            f.write('chr' + "\t" + 'breakpoint_A' + "\t" + 'breakpoint_B' + "\t" + "SV_type" + "\n")
            for item in sv_list:
                f.write(item[0] + '\t' + str(int(item[1]) * 50000) + '\t' + str(int(item[2]) * 50000) + '\t' + str(item[3]))

