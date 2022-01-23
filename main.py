import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings('ignore')

number_of_male = 1000000
number_of_female = 4000000
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
plt.bar(bin_center_male, p_x_given_male, color='b')
plt.bar(bin_center_female, p_x_given_female, color='r', alpha=0.5)
plt.show()
# finding the prior P(X)
p_male = number_of_male / (number_of_male + number_of_female)
p_female = number_of_female / (number_of_male + number_of_female)

# plotting determined prior
plt.figure()
plt.bar(['m'], [p_male], color='b')
plt.bar(['f'], [p_female], color='r')
plt.show()

# finding the posterior using bayes rule P(Y|X)
p_male_given_x = p_x_given_male * p_male / (p_x_given_male * p_male + p_x_given_female * p_female)
p_female_given_x = p_x_given_female * p_female / (p_x_given_male * p_male + p_x_given_female * p_female)

# plotting the calculated posterior
plt.figure()
plt.plot(bin_center_male, p_male_given_x, linewidth=4, color='b')
plt.plot(bin_center_female, p_female_given_x, linewidth=4, color='r')
plt.show()

# loss weights
predicting_male_weight = 0.25
predicting_female_weight = 0.75

# plotting the loss weights
plt.figure()
plt.bar(['m'], [predicting_male_weight])
plt.bar(['f'], [predicting_female_weight])
plt.show()

# calculating per sample risk
'''
    An important point to note here that we have to use the probability of female given x when calculating
    the error for predicting male, cause when the prediction of male is false that means the real value Y is
    female
'''
per_sample_risk_male = p_female_given_x * predicting_male_weight
per_sample_risk_female = p_male_given_x * predicting_female_weight

plt.figure()
plt.plot(per_sample_risk_male, linewidth=4, color='b')
plt.plot(per_sample_risk_female, linewidth=4, color='r')
plt.show()

# calculating the total error given a threshold
total_risk_per_threshold = []

for threshold in bin_center_male:
    total_risk_per_threshold.append(
        (np.sum(per_sample_risk_male[bin_center_male >= threshold]) +
        np.sum(per_sample_risk_female[bin_center_female < threshold]))/len(bin_center_male)
    )

plt.figure()
plt.plot(bin_center_male, total_risk_per_threshold)
plt.show()
