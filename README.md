# Welcome to our Wikipedia Search Engine
[<img src="./MidJourney_IR.png" width="350"/>](./MidJourney_IR.png)


Our search engine is designed to index and search all English Wikipedia pages. We use inverted indices to quickly retrieve data, and have 3 inverted indexes: one for the title, one for the anchor text, and one for the body text. The body inverted index ignores words with less than 50 occurrences. We tokenize words using Regex, and ignore English stop words (using the nltk module and corpus stopwords) while separating between the title, body, and anchor text of each page.

## Main Features

Our search engine has several features that users can access through an API:

* **search** - Our main search engine. You can send requests to http://our_domain/search, for example: http://our_domain/search?query=hello+world. This engine combines all other methods, uses a weighted sum to retrieve and rank results, and returns the top 100 best results.


* **search_body** - Search engine to retrieve matches from the body of documents. You can send requests to http://our_domain/search_body, for example: http://our_domain/search_body?query=hello+world. This engine returns up to 100 search results for the query, using TF-IDF and Cosine similarity of the body of articles only. Words with less than 50 appearances in the corpus are ignored.


* **search_title** - Search engine to retrieve matches from the titles of documents. You can send requests to http://our_domain/search_title, for example: http://our_domain/search_title?query=hello+world. This engine returns all search results for the query, ordered in descending order of the number of distinct query words that appear in the title.


* **search_anchor** - Search engine to retrieve matches from the anchor links of documents. You can send requests to http://our_domain/search_anchor, for example: http://our_domain/search_anchor?query=hello+world. This engine returns all search results for the query, ordered in descending order of the number of distinct query words that appear in the anchor text.


* **get_pagerank** - Returns PageRank values for a list of provided Wikipedia article IDs. You can use this by issuing a POST request to a URL like: http://our_domain/get_pagerank with a json payload of the list of article ids.


* **get_pageview** - Returns page view values for a list of provided Wikipedia article IDs. You can use this by issuing a POST request to a URL like: http://our_domain/get_pageview with a json payload of the list of article ids.


## Technologies and Concepts Used

Our search engine is built using several technologies and mathematical concepts, including: 

* Google Cloud Platform (GCP)
* Inverted indices
* TF-IDF
* Cosine similarity
* Binary search
* BM25
* PageRank
* Pandas
* Numpy
* Flask
* Spark
* Regex
* nltk

## Parameters and Customization

Users can tweak the following parameters to customize the behavior of the search engine:

* Weights in the main search engine
* Path to the corpus
* Number of minimum appearances in the body inverted index

## Use Cases and Suitability

Our search engine is particularly well-suited for searching Wikipedia files, as it was trained and tested only on this corpus.

## Example Queries and Benchmark Results

We provide examples of queries and their benchmark results for users to test the search engine.

## Performance Metrics

We are currently working on creating and presenting performance metrics for users to assess the quality of the search engine's results.

## Installation and Requirements

To install and run our search engine, you will need:

* Python 
* Numpy 
* Pandas
* sklearn 
* Flask
* pickle
* collections
* nltk

## Known Issues and Limitations

Currently, we are still trying to embed Doc2Vec neural network algorithm and it is not working properly yet. This feature is not currently available in our search engine.
