import numpy as np
import concurrent.futures
import time
import params

class ParallelLSH:
    def __init__(self, num_processes: int, num_hash_tables: int, num_hash_functions: int, bucket_width: float, threshold: int, top_n: int) -> None:
        '''
        Class constructor to initialize the parallel LSH algorithm.

        Parameters:
            - num_processes (int): number of processes to use.
            - num_hash_tables (int): number of hash tables to create (M).
            - num_hash_functions (int): number of hash functions to use to reduce the dimensionality of the input vector (K).
            - bucket_width (float): width of the bucket used in LSH algorithm (w). Larger values of bucket width may lead to more approximate matches but require more time to retrieve the results (default 4).
            - threshold (int): number of times a sample must be hashed near the query to be considered a candidate (default 2).
            - top_n (int): number of nearest neighbors to retrieve as recommendations (default 10).
        '''
        self.num_processes = num_processes
        self.num_hash_tables = num_hash_tables
        self.num_hash_functions = num_hash_functions
        self.top_n = top_n
        self.bucket_width = bucket_width
        self.threshold = threshold

        # Set seed for reproducibility.
        np.random.seed(params.SEED)

    def hash_table_worker(self, data_block: np.ndarray, query: np.ndarray, random_vectors: np.ndarray) -> np.ndarray:    
        '''
        Worker function to calculate M hash tables for a block of data samples.

        Parameters:
            - data_block (np.ndarray): array containing a block of data samples embeddings to be hashed (N x 768).
            - query (np.ndarray): query sample embeddings for which to find the approximate nearest neighbors (1 x 768).
            - random_vectors (np.ndarray): array containing random vectors R (with size of 768*k) from normal distribution to reduce dimensions of input vector (M x 768 x K).

        Returns:
            - occurrences_block (np.ndarray): array containing the number of times a sample is hashed near the query in the block (N x 1).
        '''
        # Define occurrences table to store number of times a sample is hashed near the query in the block.
        occurrences_block: np.ndarray = np.zeros(shape=(data_block.shape[0]), dtype=np.int32)

        for i in range(self.num_hash_tables):
            # Calculate hash vector for data.
            data_hash: np.ndarray = np.dot(data_block, random_vectors[i])

            # Calculate hash vector for query.
            query_hash: np.ndarray = np.dot(query, random_vectors[i])

            # Calculate Eucldean distance between each data sample and query.
            distances: np.ndarray = np.linalg.norm(data_hash - query_hash, axis=1)

            # Increment occurrence number of each samples with Euclidian distance from query less then w/2.
            occurrences_block += (distances <= self.bucket_width/2)

        return occurrences_block

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
        
        # Define occurrences table to store number of times a sample is hashed near the query.
        occurrences: np.ndarray = np.zeros(shape=(data.shape[0]), dtype=np.int32)

        # Start creation time.
        start_creation_time: float = time.time()

        # Define list of random vectors R (with size of 768*k) from normal distribution to reduce dimensions of input vector.
        random_vectors: list[np.ndarray] = [np.random.normal(size=(data.shape[1], self.num_hash_functions)) for _ in range(self.num_hash_tables)]
        
        with concurrent.futures.ProcessPoolExecutor() as executor:
            # Divide the matrix into blocks and distribute the work among the processes.
            block_size, remainder = divmod(data.shape[0], self.num_processes)
            block_starts = [i * block_size + min(i, remainder) for i in range(self.num_processes)]
            block_ends = [(i + 1) * block_size + min(i + 1, remainder) for i in range(self.num_processes)]
            block_indices = [(block_start, block_end) for block_start, block_end in zip(block_starts, block_ends)]
            
            futures = []
            for block_start, block_end in block_indices:
                # Create a new process and pass the arguments.
                futures.append(executor.submit(self.hash_table_worker, data[block_start:block_end, :], query, random_vectors))
            
            # Wait for the processes to finish and collect the results.
            for future in concurrent.futures.as_completed(futures):
                # Get index of the future that has completed its execution.
                future_index: int = futures.index(future)

                # Get the result of the future and store it in the occurrences table.
                block_start, block_end = block_indices[future_index]
                occurrences[block_start:block_end] = future.result()

        # Calculate creation time.
        end_creation_time: float = time.time()
        creation_time: float = end_creation_time - start_creation_time
        print(f'Parallel LSH creation time (s): {round(creation_time, 4)}') if params.VERBOSE > 1 else None

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
        print(f'Parallel LSH recommendation time (s): {round(recommendation_time, 4)}') if params.VERBOSE > 1 else None
        
        # Calculate xecution time.
        execution_time: float = creation_time + recommendation_time
        print(f'Parallel LSH total time (s): {round(execution_time, 4)}') if params.VERBOSE > 1 else None

        # Return top N similar samples as recommendation items.
        return recommended_samples_index, creation_time, recommendation_time, execution_time