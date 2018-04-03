# used for generating dataset
# from file:
# image_live_1.txt

import os.path as osp
import os
import shutil

input_txt = './data/image_live_1.txt'
output_root = '../dataset/transIQA'
output_txt = output_root + '/image_live_2.txt'

images = [line.rstrip('\n').split()[0] for line in open(input_txt)]
scores = [line.rstrip('\n').split()[1] for line in open(input_txt)]
images_dst = []

def copyfile(src, dst_root):
    """
    Copy file from source file directory to destination root
    :param src: string type
    :param dest_root: do not have '/'
    :return: dst file name
    """
    assert osp.isfile(src), 'src file fault'
    assert osp.isdir(dst_root), 'dst root fault'

    src_root, src_name = osp.split(src)
    dst = dst_root + '/' + src_name

    shutil.copyfile(src, dst)
    print('Copy %s -> %s'%(src, dst))

    return dst


# copy file and generate new file name
for i in range(len(images)):
    dst = copyfile(images[i], output_root + '/pristine_images')
    images_dst.append(dst)

# generate new txt file
with open(output_txt, 'w') as f:
    for i in range(len(images_dst)):
        f.write(images_dst[i] + ' ' + scores[i] + '\n')
print('generate file -> %s'%(output_txt))

print('Done!')


