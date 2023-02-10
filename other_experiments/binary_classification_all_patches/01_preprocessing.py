import argparse
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import IncrementalPCA
from tc_formation.binary_classifications.data.full_domain_tfrecords_data_loader import FullDomainTFRecordsDataLoader
import tensorflow as tf
from tqdm import tqdm


DATA_PATH = 'data/ncep_WP_EP_6h_all_Train.tfrecords'
N_COMPONENTS = 20


def parse_arguments(args=None):
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'inpath',
        default=DATA_PATH,
        help='Path to .tfrecords training dataset.')
    parser.add_argument(
        '--nb-pca',
        dest='nb_pca',
        default=N_COMPONENTS,
        type=int,
        help='Number of PCA components.')
    parser.add_argument(
        '--suffix',
        default='',
        help='Suffix to be added to output files.')

    return parser.parse_args(args)


def extract_reshape_data_to_2d_matrix(data, *args):
    """
    This function will extract the data part only,
    and reshape it from (W, H, C) into (W * H, C)
    so it can be processed easily downstream.
    """
    datashape = data.shape
    return tf.reshape(data, (-1, datashape[-1]))


def pickle_dump(obj, filename):
    with open(filename, 'wb') as obj_out:
        pickle.dump(obj, obj_out, protocol=pickle.HIGHEST_PROTOCOL)


def main(args=None):
    args = parse_arguments(args)

    ds = FullDomainTFRecordsDataLoader(
        datashape=(41, 161, 136)).load_dataset(args.inpath)
    ds = ds.map(extract_reshape_data_to_2d_matrix)

    # First, fit standard scaler.
    scaler = StandardScaler()
    for data in tqdm(iter(ds), desc='Standard Scaler'):
        scaler.partial_fit(data)

    # Next, scale data using standard scaler, and then fit PCA.
    pca = IncrementalPCA(args.nb_pca)
    for data in tqdm(iter(ds), desc='PCA'):
        data = scaler.transform(data)
        pca.partial_fit(data)

    # Finally, save these objects.
    suffix = args.suffix
    pickle_dump(scaler, f'scaler{suffix}.pkl')
    pickle_dump(pca, f'pca{suffix}.pkl')


if __name__ == '__main__':
    main()
