import numpy as np
from lsh_sequential import SequentialLSH
from lsh_parallel import ParallelLSH
import utils
import params
import argparse
import sys

def main(args):
    # Parse arguments.
    CODE_EXAMPLES_EMBEDDINGS_PATH = args.code_examples_embeddings_path
    QUERIES_EMBEDDINGS_PATH = args.queries_embeddings_path
    OUTPUT_PATH = args.output_path
    EXECUTION_TYPE = args.execution_type
    NUM_PROCESSES = args.num_processes
    NUM_HASH_TABLES = args.num_hash_tables
    NUM_HASH_FUNCTIONS = args.num_hash_functions
    BUCKET_WIDTH = args.bucket_width
    THRESHOLD = args.threshold
    TOP_N = args.top_n

    print("Loading embeddings...") if params.VERBOSE > 1 else None

    # Load data.
    data, queries = utils.load_embeddings(CODE_EXAMPLES_EMBEDDINGS_PATH, QUERIES_EMBEDDINGS_PATH)

    # Store times.
    creation_times: list = []
    recommendation_times: list = []
    execution_times: list = []

    if EXECUTION_TYPE == 'sequential':
        # Set number of processes to 1.
        NUM_PROCESSES = 1

        # Create LSH Sequential object.
        print("Running sequential LSH...") if params.VERBOSE > 1 else None
        lsh: SequentialLSH = SequentialLSH(NUM_HASH_TABLES, NUM_HASH_FUNCTIONS, BUCKET_WIDTH, THRESHOLD, TOP_N)

    elif EXECUTION_TYPE == 'parallel':
        # Create LSH Parallel object.
        print("Running parallel LSH on " + str(NUM_PROCESSES) + " processes...") if params.VERBOSE > 1 else None
        lsh: ParallelLSH = ParallelLSH(NUM_PROCESSES, NUM_HASH_TABLES, NUM_HASH_FUNCTIONS, BUCKET_WIDTH, THRESHOLD, TOP_N)

    else:
        print("Invalid execution type. Please use 'sequential' or 'parallel'.")
        sys.exit(1)

    # Execute LSH algorithm for each query.
    for i, query in enumerate(queries):
        # Find similar samples to query.
        recommended_samples_index, creation_time, recommendation_time, execution_time = lsh.find_similar_samples(data, query)

        # Print recommended samples.
        print(f'Recommended samples index for query {i}: {recommended_samples_index}') if params.VERBOSE > 1 else None

        # Save times.
        creation_times.append(creation_time)
        recommendation_times.append(recommendation_time)
        execution_times.append(execution_time)

    # Calculate average times.
    avg_creation_time = np.mean(creation_times)
    avg_recommendation_time = np.mean(recommendation_times)
    avg_execution_time = np.mean(execution_times)

    print(f'Average creation time: {avg_creation_time}') if params.VERBOSE > 1 else None
    print(f'Average recommendation time: {avg_recommendation_time}') if params.VERBOSE > 1 else None
    print(f'Average execution time: {avg_execution_time}') if params.VERBOSE > 1 else None

    # Save results.
    utils.save_results(OUTPUT_PATH, NUM_PROCESSES, data.shape[0], NUM_HASH_TABLES, NUM_HASH_FUNCTIONS, avg_creation_time, avg_recommendation_time, avg_execution_time, queries.shape[0]) if params.VERBOSE > 0 else None

if __name__ == '__main__':
    # Define arguments.
    parser = argparse.ArgumentParser(description='LSH Algorithm for code recommendation on queries.')
    parser.add_argument('-C', '--code_examples_embeddings_path', type=str, default=None, help='Path to the pickle file containing code examples embeddings.', required=True)
    parser.add_argument('-Q', '--queries_embeddings_path', type=str, default=None, help='Path to the pickle file containing query embeddings.', required=True)
    parser.add_argument('-O', '--output_path', type=str, default='./results/results.txt', help='Path to save the results.')
    parser.add_argument('-E', '--execution_type', type=str, choices=['sequential', 'parallel'], default=None, help='Execution type: "sequential" or "parallel".', required=True)
    parser.add_argument('-P', '--num_processes', type=int, default=None, help='Number of processes to run in parallel. Required only when execution_type is parallel.', required='--execution_type=parallel' in sys.argv or '-E=parallel' in sys.argv)
    parser.add_argument('-M', '--num_hash_tables', type=int, default=None, help='Number of hash tables.', required=True)
    parser.add_argument('-K', '--num_hash_functions', type=int, default=None, help='Number of hash functions.', required=True)
    parser.add_argument('-w', '--bucket_width', type=float, default=4, help='Width of the bucket used in LSH algorithm.')
    parser.add_argument('-l', '--threshold', type=int, default=2, help='Number of times a sample must be hashed near the query to be considered a candidate.')
    parser.add_argument('-N', '--top_n', type=int, default=10, help='Number of nearest neighbors to retrieve as recommendations.')

    # Analyze arguments from command line.
    args = parser.parse_args()

    # Execute main function.
    main(args)