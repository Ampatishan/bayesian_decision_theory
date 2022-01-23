import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

number_of_male = 1000000
number_of_female = 1000000
male_height = torch.randn(number_of_male) * 10 + 160
female_height = torch.randn(number_of_female) * 10 + 140


def generate_pdf(input_data):
    x = np.histogram(input_data, bins=int(max(input_data) - min(input_data)))
    pdf = x[0] / len(input_data)
    bin_centers = (x[1][1:] + x[1][:-1]) / 2
    return pdf, bin_centers

# height = torch.cat((male_height, female_height))
# sns.distplot(p_male_height)
# plt.show()
# sns.distplot(male_height)
# sns.distplot(female_height)
# plt.show()
#
# sns.barplot(['m', 'f'], [number_of_male, number_of_female])
# plt.show()
