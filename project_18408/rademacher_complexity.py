import numpy as np

def random_sample_ball(radius, num_samples, d):
    output_samples = np.random.normal(size=(num_samples,d)) 
    output_samples = (output_samples.T / np.linalg.norm(output_samples, axis=1)).T
    random_radii = np.random.random(num_samples) ** (1/d)
    output_samples = np.multiply(output_samples, random_radii[:, np.newaxis])
    return output_samples * radius

def random_sample_binary(num_samples, d):
    return np.random.choice([1, -1], (num_samples, d))

def evaluate_network(network, sample):
    current_output = np.array(sample)
    for i, matrix in enumerate(network):
        current_output = matrix @ current_output
        if i < len(network) - 1:
            current_output[current_output < 0] = 0
    return current_output

def random_sample_parameter_matrices(samples, weights_bounds, num_samples_network, param_dims):
    d = len(weights_bounds)
    parameter_matrices = []
    for i in range(num_samples_network):
        parameter_matrices_sample = []
        for j in range(d):
            parameter = random_sample_ball(weights_bounds[j], 1, param_dims[j]*param_dims[j+1])[0].reshape((param_dims[j+1], param_dims[j]))
            parameter_matrices_sample.append(parameter)
        parameter_matrices.append(parameter_matrices_sample)
    outputs = []
    for network in parameter_matrices:
        outputs_i = []
        for sample in samples:
            outputs_i.append(evaluate_network(network, sample))
        outputs.append(np.array(outputs_i))
    
    return outputs

def compute_rademacher_complexity(weights_bounds, B, N, num_samples_bin, num_samples_network, param_dims, apply_relu=True):
    """
    weights_bounds is a list of spectral bounds on weights (length d)
    param_dims is a list of dimensions of outputs from each layer
    num_samples_bin 
    B is bound on sample norms
    N is number of inputs 
    """
    assert param_dims[-1] == 1
    xis = random_sample_binary(num_samples_bin, N) # +/- 1 coefficients
    sum_expected_values = 0
    samples = random_sample_ball(B, N, param_dims[0]) # input samples 
    outputs = random_sample_parameter_matrices(samples, weights_bounds, num_samples_network, param_dims)
    print(len(outputs))

    for xi in xis:
        max_dot = -float('inf')
        for output in outputs:
            if apply_relu:
                output = np.maximum(output, 0)
            dot = np.dot(xi, output)
            max_dot = max(max_dot, abs(dot[0]))
        sum_expected_values += max_dot

    return sum_expected_values/(N*num_samples_bin)

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # num_samples = 1000
    # out = np.zeros()
    # for i in range(num_samples):
    # output = compute_rademacher_complexity([1, 1, 1], 1.2, 20, 30, 28, [5, 4, 3, 2, 1])
    # print(output)

    num_samples_network = 1000
    num_samples_bin = 100
    norm_bound_on_ball = 5
    num_samples_ball = 5
    max_depth = 5
    for d in range(1, max_depth + 1):
        data = []
        for i in range(1):
            param_dims = [3]*d
            param_dims.append(1)
            weights_bounds = [1]*d
            X = compute_rademacher_complexity(weights_bounds, norm_bound_on_ball, num_samples_ball, num_samples_bin, num_samples_network, param_dims)
            data.append(X)
        plt.hist(data, bins=50, label="Depth {}".format(d))
    plt.title("Rademacher complexity on depth [1, {}]".format(max_depth))
    plt.legend()
    plt.show()
