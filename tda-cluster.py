"""
    @author: River Reger
    @date: 20211126

    Purpose: Topological Data Analysis Clustering Script
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MiniBatchKMeans
from sklearn import datasets, decomposition, model_selection, neighbors, preprocessing, metrics
from sklearn.pipeline import Pipeline
import itertools
import os
import logging
import sys
import argparse
import persim
from ripser import Rips
from boundinnerclass import BoundInnerClass
import codecs

# location of this script
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

# logger
logging.basicConfig(level=logging.DEBUG, filename='tda.log', format='%(asctime)s %(name)s %(levelname)s:%(message)s')
logger = logging.getLogger(__name__)

# Print log to stdout
handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


class Timer:
    """
        Purpose:
            Timer class to record and log how long a process took.

        Inputs:
            name: title for the process

        Usage:
            with Timer():
                # do something
    """
    def __init__(self, name=None):
        self.name = name

    def __enter__(self):
        self.start = time.time()

    def __exit__(self, type, value, traceback):
        self.stop = time.time()
        if self.name:
            logger.info('[{}]: Elapsed: {} seconds.'.format(self.name, self.stop - self.start))
        else:
            logger.info('Elapsed: {} seconds.'.format(time.time() - self.start))


class _TDAParser(argparse.ArgumentParser):
    def __init__(self):
        # Initialization method for parsing command line inputs.
        # Define formatter class
        format = lambda prog: argparse.RawTextHelpFormatter(prog, max_help_position=52)

        # Initialize the subclass via init method
        super(_TDAParser, self).__init__(formatter_class=format,
                                            epilog='Topological Data Analysis Clustering Demo.')

        # Supported dataset: iris
        self.add_argument('--datasets', type=str)

        # Numerical arguments restricting operation
        self.add_argument("--threshold", type=float, choices=[self.__Range(0.0, 1.0)], metavar="THRESHOLD",
                          help="minimum threshold that must be exceeded.")

        # CLI mode boolean flags
        self.add_argument("--test", type=bool, choices={True, False}, metavar="TEST",
                          help=argparse.SUPPRESS)
        self.add_argument("--debug", type=bool, choices={True, False}, metavar="DEBUG",
                          help=argparse.SUPPRESS)

        # MODE to run in - right now either DEMO or MAIN:
        self.add_argument("--mode", type=str, choices={'DEMO','MAIN'}, metavar='MODE',
                          help='mode to run the script in.')

        # Control size of image chips
        self.add_argument("--chipsize", type=int, default=80, choices=range(1000), metavar="CHIPSIZE",
                          help=argparse.SUPPRESS)

        # Parse arguments and store them
        args = vars(self.parse_args())
        self.args = {k: v for k, v in args.items() if v is not None}

    class __Range(object):
        # Class for Ranges to define an interval [a, b].
        def __init__(self, start, end):
            # Initializes Range with a start and end.
            self.start = start
            self.end = end

        def __eq__(self, other):
            # Method to determine if a number is within a particular range.
            return self.start <= other <= self.end


class Dataset:
    """
        Purpose: General dataset class with functionlity like k-means, PCA, etc.
    """
    def __init__(self, dataset=None, x=None, y=None, target_names=None):
        """
            Purpose: Initializer method to construct a dataset class instance.

            Inputs:
                dataset: str
                    A string that gives the name of the dataset.
                x: array-like
                    Holds the input values.
                y: array-like
                    Holds the label (i.e., target) values.
                target_names: list, set
                    Holds the target names.

        """
        self.dataset = dataset  # dataset name - explicit support for iris
        self.set_targets(x)
        self.set_labels(y)
        self.set_target_names(target_names)
        self.X_dict = dict()
        self.set_x_dict()

        self.bPCA = False  # boolean to tell if PCA has been run
        self.X_r = None  # PCA reduced input data

        self.bKN = False  # boolean to tell if K-Nearest neighbors have been run on data
        self.bPD = False  # boolean to tell if Persistence Diagrams have been generated.
        self.bDist = False  # boolean to tell if the distance matrix has been generated.

    def set_targets(self, x=None):
        """
            Purpose: Set X target values

            Inputs:
                x: array-like
                ordered list of targets
        """
        if x is not None:
            self.X = np.array(x)
        else:
            logger.warning('WARNING: Dataset X is None.')
            self.X = x

    def set_labels(self, y=None):
        """
            Purpose: Set Y label values

            Inputs:
                y: array-like
                ordered list of labels

        """
        if y is not None:
            self.Y = np.array(y)
        else:
            logger.warning('WARNING: Labels Y is None.')
            self.Y = y

    def set_target_names(self, target_names=None):
        """
            Purpose: set target names

            Inputs:
                target_names: set like

        """
        if target_names is not None:
            self.target_names = set(target_names)
        else:
            logger.warning('WARNING: Target names not given. Using labels instead.')
            if self.Y is not None:
                self.target_names = set(self.Y)
            else:
                logger.warning('WARNING: Unable to set target names. Y labels were not set.')
                self.target_names = target_names

    def set_x_dict(self):
        """
            Purpose: set a dictionary for data

            Inputs:
                None, but X and Y should already be set

            Outputs:
                X_dict: dict
                 of the form: X[label] = matrix of examples with label

        """
        if self.Y is not None and self.X is not None:
            i = 0
            for name in self.target_names:
                mask = self.Y == i
                self.X_dict[name] = self.X[mask, :]
                i = i + 1
        elif self.Y is None:
            logger.warning('WARNING: Unable to generate X dictionary - labels were not set.')
            self.X_dict = dict()
        else:
            logger.warning('WARNING: Unable to generate X dictionary - X was not set.')
            self.X_dict = dict()

    def pca(self, n_components=2, batch_size=64):
        """
            Purpose: Perform PCA on dataset

            Inputs:
                n_components: int
                    Number of principal components to use.

                batch_size: int
                    Number of examples per batch (for scalability to larger datasets).

            Outputs:
                X_r: Data reduced by PCA
        """
        if self.X is not None and self.Y is not None:
            pca = decomposition.IncrementalPCA(n_components, batch_size)
            self.X_r = pca.fit_transform(self.X)
            self.bPCA = True
        elif self.Y is None:
            logger.warning('WARNING: Unable to generate X dictionary - labels were not set.')
        else:
            logger.warning('WARNING: Unable to generate X dictionary - X was not set.')

    def display_clusters_2d(self, lw=2, colors=None):
        """
            Purpose: Display a simple 2D projection of clusters of data via PCA

            Inputs:
                lw: int
                    Width of scatter points

                colors: array-like
                    Colors to be used in scatter plot. Default is None and chooses for the user.

        """
        if self.X_r is None:
            logger.warning('WARNING: X_r has not been set. PCA needs to be run before displaying 2D clusters.')
            return
        elif self.Y is None:
            logger.warning('WARNING: Y labels have not been set. Set data first!')
            return
        elif self.target_names is None:
            logger.warning('WARNING: Target names have not been set. Attempting to set with Y values.')
            self.set_target_names(target_names=set(self.Y))

        plt.figure()
        labels = set(self.Y)

        if colors is not None:
            for color, label, target_name in zip(colors, labels, self.target_names):
                # Show the PCA cluster results
                plt.scatter(
                    self.X_r[self.Y == label, 0], self.X_r[self.Y == label, 1], color=color, alpha=0.8, lw=lw,
                    label=target_name)
        else:
            for label, target_name in zip(labels, self.target_names):
                # Show the PCA cluster results
                plt.scatter(
                    self.X_r[self.Y == label, 0], self.X_r[self.Y == label, 1], alpha=0.8, lw=lw,
                    label=target_name)
        plt.legend(loc="best", shadow=False, scatterpoints=1)
        plt.title("PCA of {} dataset".format(self.dataset))
        plt.savefig('{}_PCA.png'.format(self.dataset))
        plt.close()

    @BoundInnerClass
    class KNeighbors:
        """
            Purpose: Inner class for K Neighbors algorithms and datasets.

        """
        def __init__(self, dset, test_size=0.7, shuffle=True):
            """
                Purpose: Holds the data for K Neighbors Classification. Inner class of Dataset.

                Inputs:
                    dset: Dataset instance

                    test_size: float
                        Percentage of data split.

                    shuffle: bool
                        Whether or not to shuffle data

            """
            self.Dataset = dset  # Dataset class instance

            if self.Dataset.X_r is not None:
                x = self.Dataset.X_r
            elif self.Dataset.X is not None:
                logger.warning('WARNING: PCA not run. Using raw data.')
                x = self.Dataset.X
            elif self.Dataset.X is None:
                logger.warning('WARNING: X was not set. Cannot perform K Neighbors Clustering.')
                return
            elif self.Dataset.Y is None:
                logger.warning('WARNING: Y was not set. Cannot perform K Neighbors Clustering.')
                return

            # Set train & test sets
            self.x_train, self.x_test, self.y_train, self.y_test = model_selection.train_test_split(x, self.Dataset.Y,
                                                                                                    test_size=test_size,
                                                                                                    random_state=1,
                                                                                                    shuffle=shuffle)

        def kn_classification(self):
            """"
                Purpose: Classify data using the K Neighbors algorithm; baseline algorithm.

                Outputs:
                    KNclassification_score: float
                        score on the test data as a decimal percent.
                    KNtraining_score: float
                        score on the training data as a decimal percent.

                    Plots confusion matrix as a PNG.
            """
            if not self.Dataset.bPCA:
                logger.warning('WARNING: PCA has not yet been run. Attempting to run...')
                self.Dataset.pca()
            model = neighbors.KNeighborsClassifier()
            pipeline = Pipeline([('scalar', preprocessing.MinMaxScaler()), ('classifier', model)])
            pipeline.fit(self.x_train, self.y_train)
            self.KNclassification_score = pipeline.score(self.x_test, self.y_test)
            self.KNtraining_score = pipeline.score(self.x_train, self.y_train)
            self.Dataset.bKN = True

            logger.info('K Neighbors Classification Score: {}'.format(self.KNclassification_score))
            logger.info('K Neighbors Training Score: {}'.format(self.KNtraining_score))

            preds = pipeline.predict(self.x_test)
            cm = metrics.confusion_matrix(self.y_test, preds)
            disp = metrics.ConfusionMatrixDisplay(cm, display_labels=self.Dataset.target_names)
            disp.plot()
            plt.savefig('{}_confusion_matrix.png'.format(self.Dataset.dataset))
            plt.close()

    @BoundInnerClass
    class TDA:
        """
            Purpose: Holds data associated with topological data analysis.

            Inputs:
                dset: Dataset outer class instance.

                rips: Instance of Rips()

                maxdim: int
                    Maximum dimension of the Vietoris-Rips (VR) homologies.
                    Only matters if rips is None.

                coeff: int
                    Number of coefficients to use for homology groups.
                    Only matters if rips is None.
        """
        def __init__(self, dset, rips=None, maxdim=1, coeff=2):
            self.Dataset = dset  # Instance of Dataset outer class

            if self.Dataset.X is None:
                logger.warning('WARNING: X was not set. Cannot perform TDA.')
                return
            elif self.Dataset.Y is None:
                logger.warning('WARNING: Y was not set. Cannot perform TDA.')
                return
            elif self.Dataset.target_names is None:
                self.Dataset.set_target_names()

            if rips is None:
                self.rips = Rips(maxdim=maxdim, coeff=coeff)
            else:
                self.rips = rips

            self.PD = dict()
            self.get_persistence_diagrams()
            self.get_distance_matrix()

        def get_persistence_diagrams(self):
            """
                Purpose: Compute the persistence diagram of each dataset class/label.

                Inputs:
                    None, dataset must be set however
            """
            if not self.Dataset.bPD:
                self.Dataset.bPD = True
                logger.info('Creating persistence diagrams for the first time for this dataset.')
            elif self.PD:
                print(self.PD)
                logger.info('PDs already exist. Overwriting...')
                self.PD = dict()

            for name in self.Dataset.target_names:
                # compute PD for each target class
                self.PD[name] = [self.rips.fit_transform(self.Dataset.X_dict[name])[1]]

                # plot diagram
                plt.figure()
                self.rips.plot(self.PD[name][0], show=False)
                plt.title('PD of {} of $H_0$'.format(name))
                plt.savefig('PD_H0_{}.png'.format(name))
                plt.close()

        def get_distance_matrix(self):
            """
                Purpose: Compute the bottleneck distance matrix for all pairs of PDs.

                Inputs:
                    None; PDs must have been computed already however.

            """
            if self.PD is None:
                logger.warning('WARNING: Persistence Diagrams have not been set or computed.')
                return
            elif not self.PD:
                logger.warning('WARNING: Persistence Diagrams dict is empty.')
                return

            # Get bottleneck distances
            self.Dist = np.zeros((len(self.PD), len(self.PD)))
            for pair in itertools.combinations_with_replacement(range(len(self.PD)),2):
                self.Dist[pair[0], pair[1]] = persim.bottleneck(self.PD[list(self.PD.keys())[pair[0]]][0],
                                                                self.PD[list(self.PD.keys())[pair[1]]][0])

            # Bottleneck distance is a metric, hence symmetric, so we can symmetrize the distance matrix
            self.Dist = self.Dist + self.Dist.T - np.diag(self.Dist.diagonal())

        def get_persistence_images(self,pixel_size=1):
            """
                Purpose: Write persistence images to PNG.

                Inputs:
                    pixel_size: float
                        This is the resolution percentage at which to write the PIs from a persistence surface

            """
            logger.info('Getting persistence images.')
            self.pimgr = persim.PersistenceImager(pixel_size=pixel_size)
            self.pimgs = dict()
            for target in self.Dataset.target_names:
                self.pimgr.fit(np.array(self.PD[target]))
                self.pimgs[target] = self.pimgr.transform(self.PD[target])[0]

                plt.figure()
                plt.title('Persistence Image of {}'.format(target))
                self.pimgr.plot_image(self.pimgs[target])
                plt.savefig('PI_H0_{}'.format(target))


def _test():
    pass

def _main():
    pass

def _demo(dataset=None):
    if not dataset:
        logger.warning("WARNING: Dataset not set. Defaulting to Iris dataset.")
        dataset = 'iris'
    elif dataset == 'iris':
        logger.info('Iris dataset selected.')
        iris = datasets.load_iris()
        X = iris.data
        Y = iris.target
        target_names = iris.target_names
    elif dataset == 'digits':
        logger.info('NIST Digits dataset selected.')
        nist = datasets.load_digits()
        X = nist.data
        Y = nist.target
        target_names = nist.target_names
    elif dataset == 'cancer':
        logger.info('Breast Cancer dataset selected.')
        cancer = datasets.load_breast_cancer()
        X = cancer.data
        Y = cancer.target
        target_names = cancer.target_names
    else:
        logger.warning("WARNING: Dataset not recognized. Defaulting to Iris dataset")
        dataset = 'iris'

    # Baseline PCA Clustering
    data = Dataset(dataset=dataset, x=X, y=Y, target_names=target_names)
    data.pca(n_components=3, batch_size=64)
    data.display_clusters_2d(lw=2)

    # Baseline K-Nearest Neighbors Classification
    k_neighbors = data.KNeighbors()
    k_neighbors.kn_classification()

    # TDA-Informed Clusters
    tda = data.TDA()
    logger.info('Distance Matrix for PDs: \n {}'.format(tda.Dist))
    tda.get_persistence_images()


if __name__ == '__main__':
    args = _TDAParser().args

    if 'datasets' in args.keys():
        if args['datasets'] == 'IRIS':
            ds = 'iris'
        elif args['datasets'] == 'DIGITS':
            ds = 'digits'
        elif args['datasets'] == 'CANCER':
            ds = 'cancer'
        else:
            logger.warning('WARNING: Dataset not recognized. Defaulting to the Iris dataset.')
            ds = 'iris'
    else:
        logger.warning('WARNING: No dataset selected. Defaulting to the Iris dataset.')
        ds = 'iris'

    if 'test' in args.keys():
        if args['test']:
            _test()

    if 'debug' in args.keys():
        if args['debug']:
            logging.basicConfig(level=logging.DEBUG)

    if 'mode' in args.keys():
        if args['mode'] == 'MAIN':
            _main()
        elif args['mode'] == 'DEMO':
            _demo(dataset=ds)
        else:
            logger.warning('WARNING: Mode not recognized. Defaulting to DEMO mode...')
            _demo()
    else:
        logger.warning('WARNING: Mode not selected. Exiting...')