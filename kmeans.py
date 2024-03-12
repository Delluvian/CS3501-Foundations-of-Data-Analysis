import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np 

kmeans_df = pd.read_csv("Data-ps2/2D_data.txt" , header = None, delimiter=r"\s+")
kmeans_df = pd.DataFrame(kmeans_df)
kmeans_df.columns = ['X', 'Y']

print(len(kmeans_df.columns))
print(kmeans_df)

# prompts to enter a k-value to generate k-clusters
x_values = kmeans_df['X'] 
y_values = kmeans_df['Y']

k = 2 #number of clusters

def kmeans(k):
    numFeatures = 2 #for this dataset x and y 
    centroids = get_random_centroids(k) #generates k number of random centroids

    iterations = 0
    old_centroids = None 

    #loop exits when should_stop return True
    while not should_stop(old_centroids, centroids, iterations):
        
        iterations += 1
        old_centroids = centroids
        
        # Assign labels to each datapoint based on centroids
        labels = get_labels(centroids)
        
        # Assign centroids based on datapoint labels
        centroids = new_centroids(labels, k)
        
    # We can get the labels too by calling getLabels(dataSet, centroids)
    return centroids


# generates k number of random centroids from given data frame
def get_random_centroids(k):
    return np.array(kmeans_df.sample(n=k))


# returns true if the data is clustered or has exceeded the maximum num of iterations o/w returns false
def should_stop(old_centroids, centroids, iterations):
    if iterations > 10:
        return True
    else:
        return np.array_equal(old_centroids, centroids)

#helper method to calculate the euclidian distance between a data point and a centroid
def get_distance(data_point,centroid_point):
    return np.sqrt((np.sum(data_point-centroid_point)**2))

#creates a list

def get_labels(centroids):
    labels  = []
    
    for index, row in kmeans_df.iterrows():
        distance = [get_distance(np.array(row), centroid) for centroid in np.array(centroids)]
        labels.append(np.argmin(distance))

    return labels
    


#generates new ceontroids based on the assigned labels
def new_centroids(labels, k):
    df = kmeans_df.copy()
    df['labels'] = labels
    
    centroids = pd.DataFrame()
    for value in range(k):
        mean_x  = df.loc[df['labels']== value, 'X'].mean()
        mean_y  = df.loc[df['labels']== value, 'Y'].mean()
        centroids[value] = [mean_x,mean_y]
        
    print(np.array(centroids).T)
    