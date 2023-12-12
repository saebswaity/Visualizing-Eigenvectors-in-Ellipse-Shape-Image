import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

def main():
    # Load the image
    image_path = 'ellipse_133.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Threshold the image to extract black points
    _, binary_image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY)

    # Find coordinates of black points
    black_points = np.column_stack(np.where(binary_image == 0))

    # Perform SVD
    mean = black_points.mean(0)
    _, S, Vh = np.linalg.svd(black_points - mean, full_matrices=True)


    # Eigenvalues and eigenvectors
    eigenvalues = S
    eigenvectors = Vh / np.linalg.det(Vh)  # normalization





    # Plot the results
    plot_results2(image,black_points, mean, eigenvectors, eigenvalues)


def plot_results2(image, black_points, mean, eigenvectors, eigenvalues):
    origin = np.flip(mean)
    # Rotate the original black points
    rotation_matrix = np.transpose(eigenvectors)
    # Calculate rotation angle
    rotation_angle = np.arctan2(eigenvectors[0, 0], eigenvectors[1, 0])
    rotated_points2 = np.dot(black_points - mean, rotation_matrix) + mean

    # Set up the figure
    fig, ax = plt.subplots(figsize=(8, 8))

    # Plot original image
    ax.imshow(image, cmap='gray')
    ax.set_title('Original Image with Results')

    # Plot original black points
    #ax.scatter(black_points[:, 1] ,black_points[:, 0], c='black', marker='.', label='Original Points',)

    # Plot eigenvectors
    
    scale_factor = 0.066  # Adjust this for better visualization
    A = scale_factor * np.diag(eigenvalues)  # lambdas
    Av = A @ eigenvectors

    Av1 = Av[0, :]
    Av2 = Av[1, :]

    ax.quiver(origin[0], origin[1], Av1[0], Av1[1], angles='xy', scale_units='xy', color='green',
              scale=1, label=f'Eigenvector {1}')

    ax.quiver(origin[0], origin[1], Av2[0], Av2[1], angles='xy', scale_units='xy', color='red',
              scale=1, label=f'Eigenvector {2}')

    # Plot rotated black points
    ax.scatter(rotated_points2[:, 1], rotated_points2[:, 0], c='blue', marker='o', label='Rotated Points', s = .5)

    # Plot rotated ellipse
    ellipse = Ellipse(xy=origin, width=2 * A[0, 0], height=2 * A[1, 1],
                      angle=np.degrees(rotation_angle), edgecolor='purple',  alpha=0.1, label='Ellipse')
    ax.add_patch(ellipse)

    # Set equal aspect ratio for the plot
    ax.set_aspect('equal', adjustable='box')

    # Set legend
    ax.legend()

    plt.show()






if __name__ == "__main__":
    main()
