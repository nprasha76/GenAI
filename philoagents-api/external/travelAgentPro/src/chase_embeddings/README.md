# Chase Embeddings Project

This project is designed to download content from the Chase credit card website, convert it into markdown format, create vector embeddings using the Sentence Transformers library, and save those embeddings for further use.

## Project Structure

```
chase-embeddings
├── src
│   ├── createEmbeddings.py      # Main script to orchestrate the process
│   ├── crawler.py                # Contains function to download content
│   ├── converter.py              # Converts downloaded content to markdown
│   ├── embeddings.py             # Generates vector embeddings
│   └── utils.py                  # Utility functions for various tasks
├── tests
│   └── test_create_embeddings.py  # Unit tests for createEmbeddings.py
├── requirements.txt              # Project dependencies
├── pyproject.toml                # Project configuration
├── .gitignore                    # Files to ignore by Git
└── README.md                     # Project documentation
```

## Installation

To set up the project, clone the repository and install the required dependencies:

```bash
git clone <repository-url>
cd chase-embeddings
pip install -r requirements.txt
```

## Usage

To run the main functionality of the project, execute the following command:

```bash
python src/createEmbeddings.py
```

This will download the content from the Chase credit card website, convert it to markdown, create vector embeddings, and save the embeddings in the current directory.

## Dependencies

This project requires the following Python packages:

- `crawl4ai`: For downloading content from the web.
- `sentence-transformers`: For creating vector embeddings.

Make sure to install these packages using the `requirements.txt` file.

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue for any suggestions or improvements.

## License

This project is licensed under the MIT License. See the LICENSE file for more details.