import unittest
from src.createEmbeddings import main_function

class TestCreateEmbeddings(unittest.TestCase):

    def test_download_content(self):
        # Test the content downloading functionality
        content = main_function.download_content("https://creditcards.chase.com/?jp_ltg=chsecate_featured&CELL=6TKV")
        self.assertIsNotNone(content)
        self.assertGreater(len(content), 0)

    def test_convert_to_markdown(self):
        # Test the conversion to markdown functionality
        sample_content = "<html><body><h1>Sample Title</h1><p>Sample paragraph.</p></body></html>"
        markdown_content = main_function.convert_to_markdown(sample_content)
        self.assertIn("# Sample Title", markdown_content)
        self.assertIn("Sample paragraph.", markdown_content)

    def test_create_embeddings(self):
        # Test the embeddings creation functionality
        sample_markdown = "# Sample Title\n\nSample paragraph."
        embeddings = main_function.create_embeddings(sample_markdown)
        self.assertIsNotNone(embeddings)
        self.assertEqual(len(embeddings), 384)  # Check the embedding size for "all-MiniLM-L6-v2"

    def test_save_embeddings(self):
        # Test the saving of embeddings
        sample_embeddings = [0.1] * 384  # Example embedding
        file_path = "test_embeddings.npy"
        main_function.save_embeddings(sample_embeddings, file_path)
        self.assertTrue(os.path.exists(file_path))

if __name__ == '__main__':
    unittest.main()