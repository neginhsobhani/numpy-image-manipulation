from PIL import Image
import numpy as np
from PIL.ExifTags import TAGS

pixel = 2000


# Question 3
def extract_metadata(image):
    has_metadata = False
    exifdata = image.getexif()
    for tag_id in exifdata:
        tag = TAGS.get(tag_id, tag_id)
        data = exifdata.get(tag_id)
        if isinstance(data, bytes):
            has_metadata = True
            data = data.decode()
        print(f"{tag:25}: {data}")
    if not has_metadata:
        print("This image doesn't have metadata")
    return has_metadata


# Question 4 - Mean of Data
def mean_matrix(matrix):
    mean = np.zeros((1, 3))
    for row in range(matrix.shape[0]):
        mean = mean + matrix[row]
    mean /= matrix.shape[0]
    print("The mean of data is :")
    print_matrix(mean)
    return mean


# ًQuestion 5 - Covariance matrix
def covariance_matrix(matrix, mean):
    B_matrix = matrix
    for row in range(matrix.shape[0]):
        B_matrix[row] = B_matrix[row] - mean
    B_matrix = B_matrix.transpose()
    S_matrix = (B_matrix.dot(B_matrix.transpose())) / (matrix.shape[0] - 1)
    print("The covariance matrix is :")
    print_matrix(S_matrix)
    return S_matrix


# Question 6 - Covariance and Variance analysis
def calculate_variance_matrix(s_matrix):
    variance_matrix = np.zeros((1, 3))
    for r in range(3):
        variance_matrix[0][r] = s_matrix[r][r]
    return variance_matrix


def analysis(s_matrix):
    for row in range(s_matrix.shape[0]):
        for column in range(s_matrix.shape[1]):
            if row != column:
                print("COVARIANCE OF( X", row + 1, ", X", column + 1, ")=", round(s_matrix[row][column], 4), end=" \t")
                if round(s_matrix[row][column], 4) == 0:
                    print("\tX", row + 1, "and X", column + 1, "are UNCORRELATED.")
                else:
                    print("")
    print('VARIANCES ANALYSIS :')
    variance_matrix = calculate_variance_matrix(s_matrix)
    for v in range(variance_matrix.shape[1]):
        print("VARIANCE OF( X", v + 1, ")=", round(variance_matrix[0][v], 4))


def calculate_eigenvalues(matrix):
    eigenvalues, eigenvctors = np.linalg.eig(matrix)
    return eigenvalues


# diagonal matrix of eigenvalues
def catculate_d_matrix(eigenvalues_matrix):
    d_matrix = np.zeros((3, 3))
    for i in range(3):
        d_matrix[i][i] = eigenvalues_matrix[i]
    return d_matrix


def calculate_eigenvectors(matrix):
    eigenvalues, eigenvectors = np.linalg.eig(matrix)
    return eigenvectors


# Question 7
def calculate_total_variance(d_matrix):
    total_var = 0
    for d in range(d_matrix.shape[0]):
        total_var += d_matrix[d][d]
    return total_var


def dimension_reduction(s_matrix):
    eigenvalues = calculate_eigenvalues(s_matrix)
    d_matrix = catculate_d_matrix(eigenvalues)
    total_var = calculate_total_variance(d_matrix)
    print("EIGENVALUES Of Covariance Matrix :")
    for e in range(3):
        print("Eigenvalue", e + 1, " : ", eigenvalues[e], "Or in rounded form is :", round(eigenvalues[e], 4),"\n")
    print("The D-matrix is :")
    print_matrix(d_matrix)
    print("The total variance( tr(D) ) is :", total_var, " Or in rounded form is :", round(total_var, 4), "\n")

    for i in range(3):
        print('x', i + 1, ':', d_matrix[i][i], '/', total_var, '=',
              round(100 * d_matrix[i][i] / total_var, 2), '%')


# Question 8
def principal_component_analysis(image_matrix, matrix):
    p = calculate_eigenvectors(matrix)
    print("Eigenvector matrix :")
    print_matrix(p)
    print("\n")
    p_inverse = np.linalg.inv(p)
    print("The inverse of Eigenvector matrix :")
    print_matrix(p_inverse)
    print("\n")
    y = p_inverse.dot(image_matrix.transpose())
    print("The Y = P^(-1) * X matrix :")
    print(y)
    print("\n")
    y = y.transpose()
    data = np.zeros((pixel, pixel, 3), dtype=np.uint8)
    for p in range(pixel * pixel):
        data[int(p / pixel), p % pixel] = [y[p][0], y[p][1], y[p][2]]
    result = Image.fromarray(data, 'RGB')
    result_name = 'result.png'
    result.save(result_name)
    result.show()


def print_matrix(matrix):
    matrix = matrix.transpose()
    for row in range(matrix.shape[0]):
        if row == 0:
            print('  [[', end=' ')
        else:
            print('   [', end=' ')
        for col in range(matrix.shape[1]):
            m = matrix[row][col]
            print(np.round(m, 4), end='   \t')
        if row == matrix.shape[0] - 1:
            print(']]')
        else:
            print(']')


def image_processing():
    print("Question 1 and Question 2")
    print("First we get the image and resize it and convert it to a matrix")
    image = Image.open("coffee.jpg")
    original_image = image
    image = image.resize((pixel, pixel))
    image_matrix = np.array(image)
    image_matrix.resize(pixel * pixel, 3)
    print(image_matrix.transpose())
    print("><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
    print("Question 3 - Extracting the METADATA of image")
    extract_metadata(original_image)
    print("><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
    print("Question 4 - Finding the mean of data")
    mean_data = mean_matrix(image_matrix)
    print("><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
    print("Question 5 - Building the Covariance matrix")
    cov_matrix = covariance_matrix(image_matrix, mean_data)
    print("><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
    print("Question 6 - Covariance and Variance Analysis")
    analysis(cov_matrix)
    print("><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
    print("Question 7 - Dimension Reduction")
    # var_matrix = calculate_variance_matrix(cov_matrix)
    dimension_reduction(cov_matrix)
    print("><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><><>")
    print("Question 8 - Principal Component Analysis")
    principal_component_analysis(image_matrix, cov_matrix)
    print("The result image is :")


image_processing()
