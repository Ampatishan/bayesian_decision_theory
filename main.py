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
    x = np.histogram(input_data, bins=range(100, 200))
    pdf = x[0] / len(input_data)
    bin_centers = (x[1][1:] + x[1][:-1]) / 2
    return pdf, bin_centers


# finding the likelihood P(X|H)
p_x_given_male, bin_center_male = generate_pdf(male_height)
p_x_given_female, bin_center_female = generate_pdf(female_height)

# plotting the calculated likelihoods
plt.bar(bin_center_male, p_x_given_male)
plt.bar(bin_center_female, p_x_given_female, alpha=0.5)
plt.show()
# finding the prior P(X)
p_male = number_of_male / (number_of_male + number_of_female)
p_female = number_of_female / (number_of_male + number_of_female)

# plotting determined prior
plt.figure()
plt.bar(['m'], [p_male])
plt.bar(['f'], [p_female])
plt.show()

# finding the posterior using bayes rule P(Y|X)
p_male_given_x = p_x_given_male * p_male / (p_x_given_male * p_male + p_x_given_female * p_female)
p_female_given_x = p_x_given_female * p_female / (p_x_given_male * p_male + p_x_given_female * p_female)

# plotting the calculated posterior
plt.figure()
plt.plot(bin_center_male, p_male_given_x, linewidth=4)
plt.plot(bin_center_female, p_female_given_x, linewidth=4)
plt.show()
