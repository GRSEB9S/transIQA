import numpy as np
import os.path as osp

input_txt = './data/image_live_2.txt'
output_txt = './data/image_score_generated.txt'


def read_txt_file(input_txt):
    images = [line.rstrip('\n').split()[0] for line in open(input_txt)]
    scores = [line.rstrip('\n').split()[1] for line in open(input_txt)]
    scores = np.array(scores, dtype=np.float32)

    return images, scores

def write_txt_file(lines, output_txt):
    with open(output_txt, 'w') as f:
        for line in lines:
            if osp.isfile(line.split()[0]):
                f.write(line + '\n')
            else:
                print('wrong with image path:\n%s'%line.split()[0])
                exit(1)
    print('OK!')


def generate_level_score(input_txt, output_txt):
    distortion = ['GB', 'GN', 'JP2K', 'JPEG']
    suffix = ['.bmp', '.bmp', '.jp2', '.jpg']
    level = 5

    images, scores = read_txt_file(input_txt)
    lines = []
    for i in range(len(images)):
        lines.append(images[i] + ' ' + str(scores[i])[:9])
        #debug
        debug=0
        if debug:
            path, file_name = osp.split(images[i])
            image_name, suffix = osp.splitext(file_name)
            path_root = osp.split(osp.split(images[i])[0])[0]
            print(path_root, image_name, suffix)
            exit(0)
        path_root = osp.split(osp.split(images[i])[0])[0]
        image_name, _ = osp.splitext(osp.split(images[i])[1])
        for d in range(len(distortion)):
            for l in range(level):
                score = scores[i] + (l + 1) * 10
                file_path = osp.join(path_root, distortion[d],
                                     distortion[d] + str(l + 1),
                                     image_name + suffix[d])
                lines.append(file_path + ' ' + str(score)[:9])

    write_txt_file(lines, output_txt)




generate_level_score(input_txt, output_txt)