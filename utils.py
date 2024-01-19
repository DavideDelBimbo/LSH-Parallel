import numpy as np
import os

def load_embeddings(code_examples_embeddings_path: str, nl_queries_embeddings_path: str) -> tuple[np.ndarray, np.ndarray]:
    '''
    Function to load embeddings for code examples and Natural Language queries from files.

    Parameters:
        - code_examples_embeddings_path (str): file path of the pickle file (.pkl) containing embeddings for code examples.
        - nl_queries_embeddings_path (str): file path of the pickle file (.pkl) containing embeddings for Natural Language queries.

    Returns:
        - tuple: a tuple containing two numpy arrays:
            - data_vectors (np.ndarray): numpy array of embeddings for code examples.
            - nl_query_vectors (np.ndarray): numpy array of embeddings for Natural Language queries.
    '''
    # Check if files exist.
    if not os.path.exists(code_examples_embeddings_path):
        raise FileNotFoundError(f'File {code_examples_embeddings_path} not found.')
    if not os.path.exists(nl_queries_embeddings_path):
        raise FileNotFoundError(f'File {nl_queries_embeddings_path} not found.')
    
    # Load embeddings for code examples.
    data_vectors = np.load(code_examples_embeddings_path, allow_pickle = True)
    data_vectors = np.array(data_vectors)

    # Load embeddings for Natural Language queries.
    nl_query_vectors = np.load(nl_queries_embeddings_path, allow_pickle = True)
    nl_query_vectors = np.array(nl_query_vectors)

    # Return a tuple containing numpy arrays of code examples embeddings and Natural Language queries embeddings.
    return (data_vectors, nl_query_vectors)

def save_results(output_path: str, num_processes: int, num_code_examples: int, num_hash_tables: int, num_hash_functions: int, creation_time: float, recommendation_time: float, execution_time: float, iterations: list) -> None:
    '''
    Function to save results to a file.

    Parameters:
        - output_path (str): file path of the file to save the results.
        - num_processes (int): number of processes used to run the LSH algorithm.
        - num_code_examples (int): number of code examples used to create the hash tables.
        - num_hash_tables (int): number of hash tables used to create the hash tables.
        - num_hash_functions (int): number of hash functions used to create the hash tables.
        - creation_time (float): time to create the hash tables.
        - recommendation_time (float): time to find the approximate nearest neighbors.
        - execution_time (float): total time to execute the LSH algorithm.
        - iterations (list): list of the number of iterations for each query.
    '''
    header = 'num_processes,num_code_examples,num_hash_tables,num_hash_functions,creation_time,recommendation_time,execution_time,iterations\n'
    
    if not os.path.exists(output_path):
        with open(output_path, 'w') as f:
            f.write(header)
    
    with open(output_path, 'a') as f:
        f.write(f'{num_processes},{num_code_examples},{num_hash_tables},{num_hash_functions},{creation_time},{recommendation_time},{execution_time},{iterations}\n')
