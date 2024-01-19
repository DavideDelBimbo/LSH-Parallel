import numpy as np
import time
import params

class SequentialLSH:
    def __init__(self, num_hash_tables: int, num_hash_functions: int, bucket_width: float, threshold: int, top_n: int) -> None:
        '''
        Class constructor to initialize the sequential LSH algorithm.

        Parameters:
            - num_hash_tables (int): number of hash tables to create (M).
            - num_hash_functions (int): number of hash functions to use to reduce the dimensionality of the input vector (K).
            - bucket_width (float): width of the bucket used in LSH algorithm (w). Larger values of bucket width may lead to more approximate matches but require more time to retrieve the results (default 4).
            - threshold (int): number of times a sample must be hashed near the query to be considered a candidate (default 2).
            - top_n (int): number of nearest neighbors to retrieve as recommendations (default 10).
        '''
        self.num_hash_tables = num_hash_tables
        self.num_hash_functions = num_hash_functions
        self.top_n = top_n
        self.bucket_width = bucket_width
        self.threshold = threshold

        # Set seed for reproducibility.
        np.random.seed(params.SEED)

    def find_similar_samples(self, data: np.ndarray, query: np.ndarray) -> (np.ndarray, float, float, float):
        '''
        Locality Sensitive Hashing (LSH) algorithm for approximate nearest neighbor search.

        Parameters:
            - data (np.ndarray): array containing the data samples embeddings to be hashed (N x 768).
            - query (np.ndarray): query sample embeddings for which to find the approximate nearest neighbors (1 x 768).

        Returns:
            - recommended_samples_index (np.ndarray): array containing the indices of the top N similar samples as recommendation items.
            - creation_time (float): time to create the hash tables.
            - recommendation_time (float): time to find the approximate nearest neighbors.
            - execution_time (float): total time to execute the LSH algorithm.
        '''
        # Reset seed for reproducibility.
        np.random.seed(params.SEED)

        # Define occurrences tables to store number of times a sample is hashed near the query.
        occurrences: np.ndarray = np.zeros(shape=(data.shape[0]), dtype=np.int32)

        # Start creation time.
        start_creation_time: float = time.time()

        # Define a random vectors R (with size of 768*k) from normal distribution to reduce dimensions of input vector.
        random_vectors: np.ndarray = [np.random.normal(size=(data.shape[1], self.num_hash_functions)) for _ in range(self.num_hash_tables)]

        for i in range(self.num_hash_tables):
            # Calculate hash vector for data.
            data_hash: np.ndarray = np.dot(data, random_vectors[i])

            # Calculate hash vector for query.
            query_hash: np.ndarray = np.dot(query, random_vectors[i])

            # Calculate Eucldean distance between each data sample and query.
            distances: np.ndarray = np.linalg.norm(data_hash - query_hash, axis=1)
            
            # Increment occurrence number of each samples with Euclidian distance from query less then w/2.
            occurrences += (distances <= self.bucket_width/2)

        # Calculate creation time.
        end_creation_time: float = time.time()
        creation_time: float = end_creation_time - start_creation_time
        print(f'Sequential LSH creation time (s): {round(creation_time, 4)}') if params.VERBOSE > 1 else None

        # Start recommendation time.
        start_recommendation_time: float = time.time()

        # Find candidate samples with occurrence higher than threshold.
        candidates_index: np.ndarray = np.where(occurrences >= self.threshold)[0]

        # Calculate similarity between candidate samples and query.
        distances_similarity: np.ndarray = np.linalg.norm(data[candidates_index] - query, axis=1)

        # Get first N samples sorted according to Euclidean distance.
        ranked_candidates: np.ndarray = np.argsort(distances_similarity)
        recommended_samples_index: np.ndarray = candidates_index[ranked_candidates][:self.top_n]

        # Calculate recommendation time.
        end_recommendation_time: float = time.time()
        recommendation_time: float = end_recommendation_time - start_recommendation_time
        print(f'Sequential LSH recommendation time (s): {round(recommendation_time, 4)}') if params.VERBOSE > 1 else None

        # Calculate execution time.
        execution_time: float = creation_time + recommendation_time
        print(f'Sequential LSH execution time (s): {round(execution_time, 4)}') if params.VERBOSE > 1 else None

        # Return top N similar samples as recommendation items.
        return recommended_samples_index, creation_time, recommendation_time, execution_time