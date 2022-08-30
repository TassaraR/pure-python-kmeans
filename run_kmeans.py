import kmeans
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import utils

n_clusters = 5
n_processors = 8
nr_state = 1111

df, _ = make_blobs(n_samples=100000, centers=5, n_features=25, random_state=0)

km_numpy = kmeans.KMeans(k=n_clusters, kind='numpy', random_state=1111)
km_py = kmeans.KMeans(k=n_clusters, random_state=1111)
km_par_k = kmeans.KMeans(k=n_clusters, processors=n_processors, kind='python_k', random_state=nr_state)
km_par_chunks = kmeans.KMeans(k=n_clusters, processors=n_processors, kind='python_chunks', random_state=nr_state)
km_par_rows = kmeans.KMeans(k=n_clusters, processors=n_processors, kind='python_rows', random_state=nr_state)

if __name__ == '__main__':

    print(f'Dataset shape: {df.shape}\nn_clusters: {n_clusters}\nprocessors: {n_processors}', end='\n'*2)

    km_np, _ = utils.measure_runtime(df, km_numpy, name='Numpy', verbose=True)
    km_py, _ = utils.measure_runtime(df, km_py, name='Python', verbose=True)
    print('- Parallelized models:')
    km_pk, _ = utils.measure_runtime(df, km_par_k, name='Par-K', verbose=True)
    km_pc, _ = utils.measure_runtime(df, km_par_chunks, name='Par-Chunks', verbose=True)
    km_pr, _ = utils.measure_runtime(df, km_par_rows, name='Par-Rows', verbose=True)

    print('-' * 3)
    km_scikit = KMeans(n_clusters=5, init=km_np.get_init_k(), n_init=1)
    km_scikit, _ = utils.measure_runtime(df, km_scikit, name='KMeans-scikit', verbose=True)
    print('-' * 3)

    fitted_kmeans = {'Numpy': km_np,
                     'Python': km_py,
                     'Par-K': km_pk,
                     'Par-Chunks': km_pc,
                     'Par-Rows': km_pr,
                     'Scikit': km_scikit}

    print('\nTrained K-Means algorithms saved in variable: fitted_kmeans')
