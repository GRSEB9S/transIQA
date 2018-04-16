from scipy import stats
import numpy as np

label = np.array([1., 2., 3.])
hypo = np.array([1., 1., 2.])

srocc = stats.spearmanr(hypo, label)[0]
lcc = stats.pearsonr(hypo, label)[0]

print('srocc:{:.4f}'.format(srocc))
print('lcc:{:.4f}'.format(lcc))
