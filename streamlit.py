import streamlit as st
import matplotlib.pyplot as plt
import warnings

import main

plt.style.use('dark_background')
warnings.filterwarnings('ignore')

st.title('Application of Bayesian decision theory')
st.write('A visualization of the factors affecting bayesian decision theory')

number_of_male = st.slider("Number of male", 1, 1000000, 100000, 1000)
number_of_female = st.slider('Number of female', 1, 1000000, 100000, 1000)

p_male, p_female = main.get_priors(number_of_male, number_of_female)

fig = plt.figure()
plt.bar(['m'], [p_male], color='c')
plt.bar(['f'], [p_female], color='m')
plt.ylabel('Percentage')
st.pyplot(fig)

mean_male = st.slider('Mean value of male height', 100, 200, 160, 1)
mean_female = st.slider('Mean value of female height', 100, 200, 140, 1)

deviation_male = st.slider('Deviation of male height', 0, 50, 10, 1)
deviation_female = st.slider('Deviation of female height', 0, 50, 10, 1)

p_x_given_male, bin_center_male = main.generate_pdf(number_of_male, mean_male, deviation_male)
p_x_given_female, bin_center_female = main.generate_pdf(number_of_female, mean_female, deviation_female)

fig = plt.figure()
plt.bar(bin_center_male, p_x_given_male, color='c', label='male')
plt.bar(bin_center_female, p_x_given_female, color='m', label='female', alpha=0.7)
plt.xlabel('height(cm)')
plt.ylabel('probability density')
plt.legend()
st.pyplot(fig)

p_male_given_x, p_female_given_x = main.get_posterior(p_male, p_female, p_x_given_male, p_x_given_female)

fig = plt.figure()
plt.plot(bin_center_male, p_male_given_x, linewidth=4, color='c', label='male')
plt.plot(bin_center_female, p_female_given_x, linewidth=4, color='m', label='female')
plt.xlabel('height(cm)')
plt.ylabel('probability')
plt.legend()
st.pyplot(fig)

error_weight_male = st.slider('Error weight for male', 0.0, 1.0, 0.5, 0.01)
error_weight_female = st.slider('Error weight for female', 0.0, 1.0, 0.5, 0.01)

fig = plt.figure()
plt.bar('m', error_weight_male, color='c', label='male')
plt.bar('f', error_weight_female, color='m', label='female', alpha=0.7)
plt.ylabel('loss_weight')
st.pyplot(fig)

per_sample_risk_male = main.get_per_sample_loss(p_female_given_x, error_weight_male)
per_sample_risk_female = main.get_per_sample_loss(p_male_given_x, error_weight_female)

fig = plt.figure()
plt.plot(per_sample_risk_male, linewidth=4, color='c', label='male')
plt.plot(per_sample_risk_female, linewidth=4, color='m', label='female')
plt.xlabel('height(cm)')
plt.ylabel('loss')
plt.legend()
st.pyplot(fig)

total_risk_per_threshold = main.get_total_loss(per_sample_risk_male, per_sample_risk_female, bin_center_male,
                                               bin_center_female)

fig = plt.figure()
plt.plot(bin_center_male, total_risk_per_threshold)
plt.xlabel('threshold(cm)')
plt.ylabel('total loss')
st.pyplot(fig)
