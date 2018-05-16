import numpy as np
from scipy import stats
import face_iqa
from skimage import io
import os

mode = 'ft'

if mode == 'ft12':
    model_path = '/data/junrui/github/transIQA/model/ft12/live_mse_405_0.9725_0.9674.pth.tar'
elif mode == 'ft2':
    model_path = '/data/junrui/github/transIQA/model/ft2/live_mae_398_0.9104_0.9255.pth.tar'
elif mode == 'ft':
    model_path = '/data/junrui/github/transIQA/model/ft/live_mae_283_0.9776_0.9692.pth.tar'
else:
    print('error mode')
    exit(0)

tp = mode + '_each'

fi = face_iqa.FaceIQA(enable_cuda=True, model_path=model_path, mode=mode)

res_dir = './log/results/live/'

srocc_file = open(res_dir + tp + '_srocc' + '.txt', "w")
lcc_file = open(res_dir + tp + '_lcc' + '.txt', "w")

test_file = './data/live_generator/' + 'ft_live_test.txt'
filename = [line.rstrip('\n') for line in open(test_file)]


for tp in ['gblur', 'wn', 'jpeg', 'jp2k', 'fastfading']:

    roidb = []
    scores = []
    for i in filename:
        if os.path.split(os.path.split(i.split()[0])[0])[1] == tp:
            roidb.append(i.split()[0])
            scores.append(float(i.split()[1]))
    scores = np.asarray(scores)

    Num_Image = len(scores)
    # feat = np.zeros([Num_Image,Num_Patch])
    pre = np.zeros(Num_Image)
    med = np.zeros(Num_Image)

    for i in range(Num_Image):
        directory = roidb[i]
        # im = np.asarray(cv2.imread(directory))
        im = io.imread(directory)
        # for j in range(Num_Patch):
        #    x =  im.shape[0]
        #    y = im.shape[1]
        #    x_p = np.random.randint(x-224,size=1)[0]
        #    y_p = np.random.randint(y-224,size=1)[0]
        #    temp = im[x_p:x_p+224,y_p:y_p+224,:].transpose([2,0,1])

        #    out = net.forward_all(data=np.asarray([temp]))
        #    feat[i,j] = out[ft][0]
        #    pre[i] += out[ft][0]
        # pre[i] /= Num_Patch
        # med [i] = np.median(feat[i,:])
        pre[i] = fi.get_score(im)
    srocc = stats.spearmanr(pre, scores)[0]
    lcc = stats.pearsonr(pre, scores)[0]
    print(str(tp) + '%   LCC of mean: {}'.format(lcc))
    print(str(tp) + '% SROCC of mean: {}'.format(srocc))

    srocc_file.write('%s: %6.3f\n' % (tp, srocc))
    lcc_file.write('%s: %6.3f\n' % (tp, lcc))

    # srocc = stats.spearmanr(med,scores)[0]
    # lcc = stats.pearsonr(med,scores)[0]
    # print str(tp) +'%   LCC of median: {}'.format(lcc)
    # print str(tp) + '% SROCC of median: {}'.format(srocc)

srocc_file.close()
lcc_file.close()
