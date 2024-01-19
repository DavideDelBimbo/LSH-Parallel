# LSH-Parallel
This repository contains a python implementation of a Local Sensitive Hashing (LSH) algorithm for approximate neighbor search in a set of java code examples for code recommendations (see more in [Code-Recommendations](https://github.com/DavideDelBimbo/Code-Recommendations) repository). The main objective is to provide an efficient, parallel implementation of the LSH algorithm to improve search performance in large datasets.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [License](#license)

## Prerequisites
Before you get started, make sure you have Python installed and PATH variable set.

## Installation
Follow these steps to set up and compile the code:
1. Clone this repository to your local machine using the following command:
<p  align="center"><code>git clone DavideDelBimbo/LSH-Parallel</code></p>

2. Navigate to the project directory:
<p  align="center"><code>cd LSH-Parallel</code></p>

3. Install dependencies from `requirements.txt` using the following command:
<p  align="center"><code>pip install requirements.txt</code></p>

4. Get embeddings of code examples and queries from some BERT model (see more in [Code-Recommendations](https://github.com/DavideDelBimbo/Code-Recommendations) repository).

## Usage
To execute the code, use the following command:
<p align="center"><code>python main.py --code_examples_embeddings_path --queries_embeddings_path [--output_path] --execution_type [--num_processes] --num_hash_tables --num_hash_functions [--bucket_width --threshold --top_n]</code></p>

Where:
- `--code_examples_embeddings_path`: Path to code examples embeddings.
- `--queries_embeddings_path`: Path to queries embeddings.
- `--output_path`: Path for the results.
- `--execution_type`: The execution type (use either 'parallel' or 'sequential').
- `--num_processes` (required only with `<execution_type> = 'parallel'`): The number of processes to use for parallel execution.
- `--num_hash_tables`: The number of hash tables (a higher value allows for greater accuracy in the results, but a higher cost).
- `--num_hash_functions`: The number of hash tables (a higher value allows fewer false positives, but greater complexity).
- `--bucket_width` (optional): Width of a bucket in a hash table.
- `--threshold` (optional): Number of times a sample must be hashed near the query to be considered a candidate.
- `--top_n` (optional): Number of nearest neighbors to retrieve as recommendations.

For example:
<p align="center"><code>python main.py -C='embeddings\code_examples_embeddings.pkl' -Q='embeddings\nl_queries_embeddings.pkl' -E=parallel -P=4 -M=10 -K=50 -w=100 -l=2 -N=20</code></p>

You may want to take a moment to configure the parameters in `params.h` to customize the behavior of the LSH algorithm according to your specific needs and datasets.

## License
This project is licensed under the <a href="https://github.com/DavideDelBimbo/K-Means-OpenMP/blob/main/LICENSE" target="_blank">MIT</a> License.