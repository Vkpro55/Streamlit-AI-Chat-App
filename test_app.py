import unittest
from unittest.mock import patch, MagicMock
from chatapp import get_pdf_text, get_text_chunks, get_vector_store, get_conversational_chain, user_input

class TestApp(unittest.TestCase):
    
    @patch('chatapp.PdfReader')  # Ensure we patch the PdfReader used in chatapp.py
    def test_get_pdf_text(self, MockPdfReader):
        # Create a mock PdfReader instance
        mock_pdf_reader = MagicMock()
        # Mock the pages and extract_text method
        mock_pdf_reader.pages = [MagicMock(extract_text=MagicMock(return_value="Sample text from the PDF."))]
        
        # Return the mocked PdfReader instance when PdfReader is called
        MockPdfReader.return_value = mock_pdf_reader
        
        # The pdf_docs list can be anything, since it's mocked
        pdf_docs = ["sample_pdf.pdf"]
        
        # Call the function
        result = get_pdf_text(pdf_docs)
        
        # Assertions
        self.assertIsInstance(result, str)
        self.assertIn("Sample text from the PDF.", result)  # Check if mocked text is included


    def test_get_text_chunks(self):
        # Test input where text can be split into two chunks
        text = "A" * 52000  # Text length = 52000, chunk_size=50000, overlap=1000
        result = get_text_chunks(text)
        
        # Print the lengths of the chunks to debug
        print(f"Result chunks: {[len(chunk) for chunk in result]}")  # Debugging
        
        # Check if we get the expected number of chunks
        self.assertEqual(len(result), 2)
        
        # Test chunk size and overlap
        self.assertEqual(len(result[0]), 50000)
        self.assertEqual(len(result[1]), 3000)  # Remaining part of the text
        self.assertEqual(result[0][-1000:], result[1][:1000])  # Check overlap
        
        # Test input smaller than chunk_size (should not split)
        text_short = "A" * 4000  # Text length = 4000, chunk_size=50000
        result_short = get_text_chunks(text_short)
        self.assertEqual(len(result_short), 1)
        self.assertEqual(len(result_short[0]), 4000)
        
        # Test input with exactly one chunk (size equal to chunk_size)
        text_exact = "A" * 50000  # Text length = 50000, chunk_size=50000
        result_exact = get_text_chunks(text_exact)
        self.assertEqual(len(result_exact), 1)
        self.assertEqual(len(result_exact[0]), 50000)

        # Test input with text length less than chunk_size (edge case)
        text_edge = "A" * 100  # Text length = 100, chunk_size=50000
        result_edge = get_text_chunks(text_edge)
        self.assertEqual(len(result_edge), 1)
        self.assertEqual(len(result_edge[0]), 100)
    

    @patch("chatapp.GoogleGenerativeAIEmbeddings")  # Make sure the patch target is correct
    @patch("chatapp.FAISS")  # Mocking FAISS
    def test_get_vector_store(self, MockFAISS, MockEmbeddings):
        # Mock the embeddings and FAISS objects
        mock_embeddings = MagicMock()
        MockEmbeddings.return_value = mock_embeddings
        
        mock_vector_store = MagicMock()
        MockFAISS.from_texts.return_value = mock_vector_store

        # Simulate text chunks to pass into the function
        text_chunks = ["This is a sample chunk of text.", "Another chunk for testing."]
        
        # Call the function
        get_vector_store(text_chunks)
        
        # Assert that the embeddings were created with the correct model using assert_any_call
        MockEmbeddings.assert_any_call(model="models/embedding-001")
        
        # Assert that FAISS.from_texts was called with the correct arguments
        MockFAISS.from_texts.assert_called_with(text_chunks, embedding=mock_embeddings)
        
        # Assert that save_local was called to store the vector index
        mock_vector_store.save_local.assert_called_with("faiss_index")
    

    @patch("chatapp.load_qa_chain")  # Mock load_qa_chain
    @patch("chatapp.PromptTemplate")  # Mock PromptTemplate
    @patch("chatapp.ChatGoogleGenerativeAI")  # Mock ChatGoogleGenerativeAI
    def test_get_conversational_chain(self, MockChatGoogleGenerativeAI, MockPromptTemplate, MockLoadQAChain):
        # Mock the return values
        mock_model = MagicMock()
        MockChatGoogleGenerativeAI.return_value = mock_model
        
        mock_prompt_template = MagicMock()
        MockPromptTemplate.return_value = mock_prompt_template
        
        mock_chain = MagicMock()
        MockLoadQAChain.return_value = mock_chain

        # Call the function
        chain = get_conversational_chain()

        # Assertions
        # Check that the ChatGoogleGenerativeAI was initialized with the correct parameters
        MockChatGoogleGenerativeAI.assert_called_with(model="gemini-pro", temperature=0.3)
        
        # Adjust the expected template to match actual formatting (leading newlines and spaces)
        expected_template = """
    Answer the question as detailed as possible from the provided context, make sure to provide all the details. 
    If the answer is not in the provided context, just say, "answer is not available in the context", don't provide the wrong answer.

    Context:\n{context}?\n
    Question:\n{question}\n

    Answer:
    """
        
        # Check that the PromptTemplate was initialized with the correct parameters
        MockPromptTemplate.assert_called_with(template=expected_template, input_variables=["context", "question"])
        
        # Check that load_qa_chain was called with the correct parameters
        MockLoadQAChain.assert_called_with(mock_model, chain_type="stuff", prompt=mock_prompt_template)

        # Ensure the return value of the function is the mock chain
        self.assertEqual(chain, mock_chain)

    
    
    @patch("chatapp.FAISS.load_local")  # Mock FAISS load_local method
    @patch("chatapp.GoogleGenerativeAIEmbeddings")  # Mock GoogleGenerativeAIEmbeddings
    @patch("chatapp.get_conversational_chain")  # Mock get_conversational_chain
    @patch("chatapp.st.write")  # Mock st.write
    @patch("chatapp.st.error")  # Mock st.error
    def test_user_input(self, MockStError, MockStWrite, MockGetConversationalChain, MockEmbeddings, MockFAISS):
        # Mock the return value of FAISS.load_local
        mock_db = MagicMock()
        MockFAISS.return_value = mock_db
        
        # Mock the return value of similarity_search
        mock_docs = ["doc1", "doc2"]
        mock_db.similarity_search.return_value = mock_docs
        
        # Mock the chain
        mock_chain = MagicMock()
        MockGetConversationalChain.return_value = mock_chain
        
        # Mock the response from the chain
        mock_response = {"output_text": "This is a response"}
        mock_chain.return_value = mock_response
        
        # Mock the embeddings initialization
        MockEmbeddings.return_value = MagicMock()
        
        # Test the case where FAISS index exists and the process runs successfully
        with patch("os.path.exists", return_value=True):
            user_question = "What is AI?"
            user_input(user_question)
            
            # Check that the expected calls were made
            MockFAISS.assert_called_with("faiss_index", MockEmbeddings.return_value, allow_dangerous_deserialization=True)
            mock_db.similarity_search.assert_called_with(user_question)
            MockGetConversationalChain.assert_called_once()
            mock_chain.assert_called_with({"input_documents": mock_docs, "question": user_question}, return_only_outputs=True)
            MockStWrite.assert_called_with("Reply: ", mock_response["output_text"])
        
        # Test the case where FAISS index is not found
        with patch("os.path.exists", return_value=False):
            user_input(user_question)
            MockStError.assert_called_with("The FAISS index file was not found. Please process the PDF files first.")

        # Test the case where FAISS loading raises an exception
        MockFAISS.side_effect = Exception("FAISS loading error")
        with patch("os.path.exists", return_value=True):
            user_input(user_question)
            MockStError.assert_called_with("Error loading FAISS index: FAISS loading error")


if __name__ == "__main__":
    unittest.main()
