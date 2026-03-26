import matplotlib.pyplot as plt
def plot_sample(xi_train, y_train):
    plt.figure(figsize=(10,2))
    for i in range(20):
        ax = plt.subplot(2,10,i+1)
        plt.imshow(xi_train[i] , cmap='gray_r')
        plt.title(f"label={y_train[i]}" )
        plt.axis('off')
    plt.show()