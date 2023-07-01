from model import KMeans
from utils import get_image, show_image, save_image, error
import matplotlib.pyplot as plt

def get_clustered(image,num_clusters):
    # create model
    kmeans = KMeans(num_clusters)

    # fit model
    kmeans.fit(image)

     # replace each pixel with its closest cluster center
    image_clustered = kmeans.replace_with_cluster_centers(image)
    MSE = error(image, image_clustered)
    return MSE,image_clustered
def main():
    # get image
    image = get_image('image.jpg')
    img_shape = image.shape

    # reshape image
    image = image.reshape(image.shape[0] * image.shape[1], image.shape[2])

    clusters_sizes = [2,5,10,20,50]
    MSE_total = []
    for num_clusters in [2,5,10,20,50]:
        print("\n---------- Running for num_clusters = {}\n\n".format(num_clusters))
        MSE, image_clustered = get_clustered(image,num_clusters)
        MSE_total.append(MSE)
        # Print the error
        print('MSE with K = {} is {}:'.format(num_clusters,error(image, image_clustered)))
        # reshape image
        image_clustered = image_clustered.reshape(img_shape)

        # show/save image
        # show_image(image)
        save_image((image_clustered*255).astype('uint8'), f'image_clustered_{num_clusters}.jpg')

    plt.plot(clusters_sizes,MSE_total,'-o')
    plt.xlabel('Num_CLusters')
    plt.ylabel('MSE')
    plt.savefig('K_vs_MSE.jpg')

if __name__ == '__main__':
    main()
