from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
)


class Chunking:

    def get_text(self):
        return """
        Chunking is the hidden reason 80% of RAG systems fail in production. In this tutorial you'll learn 4 chunking strategies with hands-on Python code - fixed size, paragraph-based, recursive character splitting, and markdown-aware splitting.

        Plus how to pick chunk size and overlap the right way. Episode 16 of the free 8-week AI Engineer Course designed to get you hired in 2026.

        Every senior RAG engineer will tell you the same thing - you spend more time tuning chunking than any other RAG component. Get chunking right and mediocre retrieval works fine. Get chunking wrong and no amount of advanced retrieval, reranking, or better embeddings will save you. Most tutorials skip this entirely or reduce it to "split every 500 characters" - which is exactly why so many production RAG systems return wrong answers.
        """

    def print_chunks(self, all_chunks: CharacterTextSplitter, type: str):
        text = self.get_text()
        print(f"============== {type.upper()} SIZE =============")
        for i, chunk in enumerate(all_chunks.split_text(text=text)):
            print(f"Chunk {i+1}")
            print(f"{chunk}\n---------------")

    def fixed_size_chunking(self):
        print("\n\n")
        fixed = CharacterTextSplitter(separator="", chunk_size=200, chunk_overlap=0)
        self.print_chunks(all_chunks=fixed, type="fixed")

    def paragraph_chunking(self):
        print("\n\n")
        paragraph = CharacterTextSplitter(
            separator="\n\n", chunk_size=200, chunk_overlap=0
        )
        self.print_chunks(all_chunks=paragraph, type="paragraph")

    def recursive_chunking(self):
        print("\n\n")
        recursive = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=10)
        self.print_chunks(all_chunks=recursive, type="recursive")


def main():
    chunking_obj = Chunking()
    chunking_obj.fixed_size_chunking()
    chunking_obj.paragraph_chunking()
    chunking_obj.recursive_chunking()


if __name__ == "__main__":
    main()
