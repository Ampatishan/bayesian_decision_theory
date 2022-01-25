import matplotlib.pyplot as plt
import main

number_of_male = 1000000
number_of_female = 1000000
male_height_mean = 160
female_height_mean = 140
male_height_deviation = 10
female_height_deviation = 10

# loss weights
error_weight_male = 0.5
error_weight_female = 0.5

# finding the prior P(X)
p_male, p_female = main.get_priors(number_of_male, number_of_female)

# plotting determined prior
plt.figure()
plt.bar(['m'], [p_male], color='b')
plt.bar(['f'], [p_female], color='r')
plt.show()

# finding the likelihood P(X|H)
p_x_given_male, bin_center_male = main.generate_pdf(number_of_male, male_height_mean, male_height_deviation)
p_x_given_female, bin_center_female = main.generate_pdf(number_of_female, female_height_mean, female_height_deviation)

# plotting the calculated likelihoods
plt.bar(bin_center_male, p_x_given_male, color='b')
plt.bar(bin_center_female, p_x_given_female, color='r', alpha=0.5)
plt.show()

# finding the posterior using bayes rule P(Y|X)
p_male_given_x, p_female_given_x = main.get_posterior(p_male, p_female, p_x_given_male, p_x_given_female)

# plotting the calculated posterior
plt.figure()
plt.plot(bin_center_male, p_male_given_x, linewidth=4, color='b')
plt.plot(bin_center_female, p_female_given_x, linewidth=4, color='r')
plt.show()

# plotting the loss weights
plt.figure()
plt.bar(['m'], [error_weight_male])
plt.bar(['f'], [error_weight_female])
plt.show()

# calculating per sample risk
'''
    An important point to note here that we have to use the probability of female given x when calculating
    the error for predicting male, cause when the prediction of male is false that means the real value Y is
    female
'''
per_sample_risk_male = main.get_per_sample_loss(p_female_given_x, error_weight_male)
per_sample_risk_female = main.get_per_sample_loss(p_male_given_x, error_weight_female)

plt.figure()
plt.plot(per_sample_risk_male, linewidth=4, color='b')
plt.plot(per_sample_risk_female, linewidth=4, color='r')
plt.show()

# calculating the total error given a threshold
total_risk_per_threshold = main.get_total_loss(per_sample_risk_male, per_sample_risk_female, bin_center_male,
                                               bin_center_female)

plt.figure()
plt.plot(bin_center_male, total_risk_per_threshold)
plt.show()
