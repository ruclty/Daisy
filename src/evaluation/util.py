import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import scipy.stats
from scipy import linalg
from mpl_toolkits.mplot3d import Axes3D
from PMF import PMF


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


def group_draw_histogram(pmf1, pmf2, concat=True, label=False, width=12.0, height=6.0, savepath=None):
    if pmf1.continuous == True and pmf2.continuous == True and pmf1.interval == pmf2.interval:
        minimum = min(pmf1.minimum, pmf2.minimum)
        maximum = max(pmf1.maximum, pmf2.maximum)
        pmf1.interval_align(minimum=minimum, maximum=maximum)
        pmf2.interval_align(minimum=minimum, maximum=maximum)
    elif pmf1.continuous == False and pmf2.continuous == False:
        pmf1.interval_align(keys=pmf2.keys)
        pmf2.interval_align(keys=pmf1.keys)
    else:
        print("ERROR")
        return
    labels = pmf1.keys
    data1 = pmf1.probability
    data2 = pmf2.probability
    if concat == True:
        plt.rcParams['figure.figsize'] = (width, height)
        x = np.arange(len(labels))  # the label locations
        width = 0.35  # the width of the bars
        fig, ax = plt.subplots()
        rects1 = ax.bar(x - width/2, data1, width, label='True Data')
        rects2 = ax.bar(x + width/2, data2, width, label='Generated Data')
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax.set_title("Probability Distribution")
        ax.set_xlabel('Keys')
        ax.set_ylabel('Probability')
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.legend()
        if label == True:
            autolabel(rects1, ax)
            autolabel(rects2, ax)
        fig.tight_layout()
    else:
        plt.rcParams['figure.figsize'] = (width, height*1.5)
        fig, ax=plt.subplots(2)
        # Add some text for labels, title and custom x-axis tick labels, etc.
        ax[0].set_title("Probability Distribution")
        ax[0].set_xlabel('Keys')
        ax[0].set_ylabel('Probability')
        bar1 = ax[0].bar(range(len(data1)), data1, tick_label=labels, align='center', width=0.5)
        ax[1].set_title("Probability Distribution")
        ax[1].set_xlabel('Keys')
        ax[1].set_ylabel('Probability')
        bar2 = ax[1].bar(range(len(data2)), data2, tick_label=labels, align='center', width=0.5)
        if label == True:
            autolabel(bar1, ax[0])
            autolabel(bar2, ax[1])
        fig.tight_layout()
    if savepath is not None and os.path.exists(os.path.split(savepath)[0]):
        plt.savefig(savepath)
    else:
        plt.show()


def draw_histogram(pmf, label=False, width=12.0, height=6.0,savepath=None):
    plt.rcParams['figure.figsize'] = (width, height)
    _, ax = plt.subplots()
    ax.set_title("Probability Distribution")
    ax.set_xlabel('Keys')
    ax.set_ylabel('Probability')
    bar = ax.bar(range(len(pmf.probability)), pmf.probability, tick_label=pmf.keys, align='center', width=0.5)
    if label == True:
        autolabel(bar, ax)
    if savepath is not None and os.path.exists(os.path.split(savepath)[0]):
        plt.savefig(savepath)
    else:
        plt.show()
   
        
def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
    Stable version by Dougal J. Sutherland.
    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.
    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrt(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrt((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
			np.trace(sigma2) - 2 * tr_covmean)
			

def compute_kl_divergence(data1, data2, continuous = False, interval = 1):
    '''
    Kullback-Leibler divergence
    '''
    pmf1 = PMF(data1, continuous = continuous, interval = interval)
    pmf2 = PMF(data2, continuous = continuous, interval = interval)
    if pmf1.continuous == True and pmf2.continuous == True:
        minimum = min(pmf1.minimum, pmf2.minimum)
        maximum = max(pmf1.maximum, pmf2.maximum)
        pmf1.interval_align(minimum=minimum, maximum=maximum)
        pmf2.interval_align(minimum=minimum, maximum=maximum)
    elif pmf1.continuous == False and pmf2.continuous == False:
        pmf1.interval_align(keys=pmf2.keys)
        pmf2.interval_align(keys=pmf1.keys)
    else:
        print("ERROR")
        return
    pro1 = [i + 1e-4 for i in pmf1.probability]
    pro2 = [i + 1e-4 for i in pmf2.probability]
    return scipy.stats.entropy(pro1, pro2)

    
def compute_js_divergence(data1, data2, continuous = False, interval = 1):
    '''
    Jensen-Shannon divergence:
        Djs[P,Q]=1/2Dkl[P,M]+1/2Dkl[Q,M], where M represents 1/2(P+Q).
    '''
    pmf1 = PMF(data1, continuous = continuous, interval = interval)
    pmf2 = PMF(data2, continuous = continuous, interval = interval)
    if pmf1.continuous == True and pmf2.continuous == True:
        minimum = min(pmf1.minimum, pmf2.minimum)
        maximum = max(pmf1.maximum, pmf2.maximum)
        pmf1.interval_align(minimum=minimum, maximum=maximum)
        pmf2.interval_align(minimum=minimum, maximum=maximum)
    elif pmf1.continuous == False and pmf2.continuous == False:
        pmf1.interval_align(keys=pmf2.keys)
        pmf2.interval_align(keys=pmf1.keys)
    else:
        print("ERROR")
        return
    pro1 = [i + 1e-4 for i in pmf1.probability]
    pro2 = [i + 1e-4 for i in pmf2.probability]
    m = [(a + b) / 2 for a, b in zip(pro1, pro2)]
    kl1 = compute_kl_divergence(pro1, m)
    kl2 = compute_kl_divergence(pro2, m)
    return kl1 / 2 + kl2 / 2


def plot3d_histogram(x, y, z, color='b', width=0.5):
    """Make a bar plot.
    """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    zpos = np.zeros_like(x)
    dx = width * np.ones_like(zpos)
    dy = dx.copy()
    ax.bar3d(x, y, zpos, dx, dy, z, color=color)
    plt.show()


def addtodict(thedict, key_a, key_b, val):
    """Add a value to a 2-dimensional dictionary like {'a': {'a': 1, 'b': 3}, 'b': {'a': 6}}..
    """
    if key_a in thedict:
        if key_b in thedict[key_a]:
            thedict[key_a][key_b] = thedict[key_a].get(key_b, 0) + val
        else:
            thedict[key_a].update({key_b: val})
    else:
        thedict.update({key_a:{key_b: val}})


def calculate_joint_PMF(pmf1, pmf2):
    """To calculate the joint distribution of two given 'PMF' objects.
    """
    ypos, xpos = np.meshgrid(pmf2.keys, pmf1.keys)
    x = [m for n in xpos for m in n]
    y = [m for n in ypos for m in n]
    length = len(pmf1.data)
    
    # Use a dict to count and then convert it into a list.
    count_map = {}
    for i in range(length):
        key1 = pmf1.calculate_key(pmf1.data[i])
        key2 = pmf2.calculate_key(pmf2.data[i])
        addtodict(count_map, key1, key2, 1)
    count_map = dict(sorted(count_map.items(), key=lambda x:x[0]))
    for k in count_map:
        count_map[k] = dict(sorted(count_map[k].items(), key=lambda x:x[0]))
    
    count_list = []
    for i in pmf1.keys:
        for j in pmf2.keys:
            if i in count_map:
                if j in count_map[i]:
                    count_list.append(count_map[i][j])
                else:
                    count_list.append(0)
            else:
                count_list.append(0)

    # Taking into account non-continuous variables, we need to convert it into a continuous variable.
    if pmf1.continuous == False:
        x = list(range(len(pmf1.keys)))
        x = [val for val in x for i in range(len(pmf2.keys))]

    if pmf2.continuous == False:
        y = list(range(len(pmf2.keys)))
        y = y * len(pmf1.keys)

    # Calculating the probability distribution
    count_list = [round(val/length, 3) for val in count_list]
    return x, y, count_list


def draw_3d_histogram(pmf1, pmf2):
    """Make a bar plot for two given `PMF` objects to display their joint distribution.
    """
    x, y, count_list = calculate_joint_PMF(pmf1, pmf2)
    plot3d_histogram(x, y, count_list)