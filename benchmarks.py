import kmeans
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
import utils
import pickle

n_clusters = 5
proc_list = range(2, 9)
benchmark = {
            'small': {
                 'n_proc': list(proc_list),
                 'Numpy': [],
                 'Scikit': [],
                 'Python': [],
                 'Par-K': [],
                 'Par-Chunks': [],
                 'Par-Rows': []
                },
            'large': {
                 'n_proc': list(proc_list),
                 'Numpy': [],
                 'Scikit': [],
                 'Python': [],
                 'Par-K': [],
                 'Par-Chunks': [],
                 'Par-Rows': []
                }
            }

small_df, _ = make_blobs(n_samples=1000, centers=5, n_features=20, random_state=0)
large_df, _ = make_blobs(n_samples=50000, centers=5, n_features=20, random_state=0)
data = [small_df, large_df]

if __name__ == '__main__':

    for df, name in zip(data, ['small', 'large']):
        print('Processing:', name)

        km_numpy = kmeans.KMeans(k=n_clusters, kind='numpy', random_state=1111)
        km_py = kmeans.KMeans(k=n_clusters, random_state=1111)

        km, t_np = utils.measure_runtime(df, km_numpy)
        _, t_py = utils.measure_runtime(df, km_py)

        km_scikit = KMeans(n_clusters=n_clusters, init=km.get_init_k(), n_init=1)
        _, t_scikit = utils.measure_runtime(df, km_scikit)

        benchmark[name]['Numpy'].extend([t_np] * len(proc_list))
        benchmark[name]['Scikit'].extend([t_scikit] * len(proc_list))
        benchmark[name]['Python'].extend([t_py] * len(proc_list))

        for proc in proc_list:
            print('- n_proc:', proc)
            km_par_k = kmeans.KMeans(k=n_clusters, processors=proc, kind='python_k', random_state=1111)
            km_par_chunks = kmeans.KMeans(k=n_clusters, processors=proc, kind='python_chunks', random_state=1111)
            km_par_rows = kmeans.KMeans(k=n_clusters, processors=proc, kind='python_rows', random_state=1111)

            _, t_par_k = utils.measure_runtime(df, km_par_k)
            _, t_par_chunks = utils.measure_runtime(df, km_par_chunks)
            _, t_par_rows = utils.measure_runtime(df, km_par_rows)

            benchmark[name]['Par-K'].append(t_par_k)
            benchmark[name]['Par-Chunks'].append(t_par_chunks)
            benchmark[name]['Par-Rows'].append(t_par_rows)

    print('Saving...')
    with open('benchmark.pickle', 'wb') as f:
        pickle.dump(benchmark, f, protocol=pickle.HIGHEST_PROTOCOL)
    print('Done.')
