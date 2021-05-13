import numpy as np
from scipy.stats import rv_continuous
from scipy.stats import chi2
from scipy.stats import uniform
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

class random_distribution():
    def __init__(
        self, 
        dim=1, 
        random_sampling_variable=uniform, 
        num_points=1, 
        points=None, 
        max_var=0.1
    ):
        '''
        Params:
            dim: dimension of the output vectors
            random_sampling_variable: distribution from which to draw random points
            num_points: number of random points to generate
            points: predefined points, only when predefined points should be used instead of randomly generated ones
            max_var: maximum value of variance for the gaussian distributions
        '''

        self.generate_points = points
        self.max_var = max_var
        self.random_sampling_variable = random_sampling_variable

        if self.generate_points is None:
            self.generate_points = []
            for index in range(num_points):
                self.generate_points.append(self.random_sampling_variable.rvs(size=dim))
            self.generate_points = np.array(self.generate_points)
        self.num_points = num_points
        if points is not None:
            self.num_points = len(points)
            self.dim = np.shape(points)[1]
        else:
            self.dim = dim
        self.calculate_var()

    def calculate_var(self):
        self.variances = self.max_var * self.random_sampling_variable.rvs(size=(self.num_points, self.dim))

    def pdf(self, x):
        output = 0
        index = 0
        for point in self.generate_points:
            output = output + multivariate_normal(mean=point, cov=np.eye(self.dim) * self.variances[index]).pdf(x)
        return output / self.num_points

def generate_classification_problem(
    dim,
    num_peaks_pdf,
    num_classifications=2,
    iterations=10000,
    burn=100,
    thin=1,
    plot=True
):
    pdf_var = random_distribution(dim=dim, random_sampling_variable=uniform, num_points=num_peaks_pdf)
    samples = metropolis_hastings(pdf_var.pdf, dim, iterations, burn, thin)
    if plot:
        plt.scatter(samples[:, 0], samples[:, 1], color='b')
        plt.scatter(pdf_var.generate_points[:, 0], pdf_var.generate_points[:, 1], color='r')
        plt.show()
    return samples

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
    # samples = generate_classification_problem(2, 4)
    samples, labels = make_classification(
                    n_samples=10000, 
                    n_features=2, 
                    n_informative=2,
                    n_redundant=0, 
                    n_repeated=0, 
                    n_classes=2, 
                    n_clusters_per_class=1,
                    class_sep=2,
                    flip_y=0,
                    weights=[0.5,0.5], 
                    random_state=17)
