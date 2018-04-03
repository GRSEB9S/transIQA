# contain the image with scores less than 30
# this is done when live ft score is established
# in print_live_filted_full.txt
# live_ft is okay to distinguish roughly
# output file is
# image_live_1.txt

input_txt = './data/print_live_filted_full.txt'
output_txt = './data/image_live_1.txt'

images = [line.rstrip('\n').split()[0] for line in open(input_txt)]
scores = [line.rstrip('\n').split()[1] for line in open(input_txt)]

images_txt = []
scores_txt = []

for i in range(len(images)):
    print(scores[i])
    if float(scores[i]) < 30. :
        images_txt.append(images[i])
        scores_txt.append(scores[i])

with open(output_txt, 'w') as f:
    for i in range(len(images_txt)):
        f.write(images_txt[i] + ' ' + scores_txt[i] + '\n')