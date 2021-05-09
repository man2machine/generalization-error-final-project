import numpy as np
from scipy.stats import rv_continuous
from scipy.stats import chi2
from scipy.stats import uniform
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt


class random_distribution():
    def __init__(self, dim=1, random_sampling_variable=uniform, num_points=1, points=None, max_var=2):
        # dim: dimension of the output vectors
        # random_sampling_variable: distribution from which to draw random points
        # num_points: number of random points to generate
        # points: predefined points, only when predefined points should be used instead of randomly generated ones
        self.generate_points = points
        self.max_var = max_var
        self.random_sampling_variable = random_sampling_variable

        if self.generate_points is None:
            self.generate_points = []
            for index in range(num_points):
                self.generate_points.append(self.random_sampling_variable.rvs(size=dim))
        self.num_points = num_points
        if points is not None:
            self.num_points = len(points)
            self.dim = np.shape(points)[1]
        self.calculate_var()


    def calculate_var(self):
        self.variances = self.max_var * self.random_sampling_variable.rvs(size=(self.num_points, self.dim))

    def pdf(self, x):
        output = 0
        index = 0
        for point in self.generate_points:
            output = output + multivariate_normal(mean=point, cov=np.eye(self.dim)*self.variances[index]).pdf(x)
        return output/self.num_points



# pdf_var = random_distribution(points=[[0.5,0.5,0.5]])
# print(pdf_var.pdf([0.1,0.2,0.3]))
# print(pdf_var.pdf([0.55,0.5,0.5]))
# print(pdf_var.pdf([0.45,0.5,0.5]))
# print(pdf_var.pdf([0.5,0.5,0.5]))
# print(pdf_var.pdf([-3,-5,-7]))

def metropolis_hastings(
        p,
        dim=2,
        iterations=1000,
        burn=100,
        thin=1
):
    x = np.random.normal(size=dim)
    samples = []

    for i in range(iterations):
        x_star = x + np.random.normal(size=dim)
        if np.random.rand() < p(x_star) / p(x):
            x = x_star
        if i >= burn and (i - burn) % thin == 0:
            samples.append(x)

    return np.array(samples)


if __name__ == '__main__':
    chi_squared = lambda x: chi2.pdf(x, df=55)
    samples = metropolis_hastings(chi_squared, dim=1)
    plt.hist(samples)
    # plt.show()
