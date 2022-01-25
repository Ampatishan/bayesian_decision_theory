import torch
import numpy as np
import warnings

warnings.filterwarnings('ignore')


def generate_pdf(number, mean, deviation):
    input_data = torch.randn(number) * deviation + mean
    x = np.histogram(input_data, bins=range(100, 200))
    pdf = x[0] / len(input_data)
    bin_centers = (x[1][1:] + x[1][:-1]) / 2
    return pdf, bin_centers


def get_priors(x, y):
    p1 = x / (x + y)
    p2 = y / (x + y)
    return p1, p2


def get_posterior(prior1, prior2, likelihood1, likelihood2):
    pos1 = prior1 * likelihood1 / (prior1 * likelihood1 + prior2 * likelihood2)
    pos2 = prior2 * likelihood2 / (prior1 * likelihood1 + prior2 * likelihood2)
    return pos1, pos2


def get_per_sample_loss(posterior, weight):
    return posterior * weight


def get_total_loss(per_sample_loss_male, per_sample_loss_female, bin_center_m, bin_center_f):
    total_risk_per_threshol = []

    for threshold in bin_center_m:
        total_risk_per_threshol.append(
            (np.nansum(per_sample_loss_male[bin_center_m >= threshold]) +
             np.nansum(per_sample_loss_female[bin_center_f < threshold])) / len(bin_center_m)
        )
    return total_risk_per_threshol
