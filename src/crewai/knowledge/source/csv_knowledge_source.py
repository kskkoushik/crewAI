import csv
from pathlib import Path
from typing import Dict, List

from crewai.knowledge.source.base_file_knowledge_source import BaseFileKnowledgeSource


class CSVKnowledgeSource(BaseFileKnowledgeSource):
    """Knowledge source class for handling CSV files using embeddings."""

    def load_content(self) -> Dict[Path, str]:
        """Reads all CSV files and converts them into a flat string content format."""
        file_content_map = {}
        for path in self.safe_file_paths:
            with open(path, mode="r", encoding="utf-8") as file:
                content = ""
                for line in csv.reader(file):
                    content += " ".join(line) + "\n"
                file_content_map[path] = content
        return file_content_map

    def add(self) -> None:
        """
        Chunks the loaded CSV content, creates embeddings,
        and persists those embeddings.
        """
        full_text = (
            self.content if isinstance(self.content, str) else str(self.content)
        )
        chunks = self._chunk_text(full_text)
        self.chunks.extend(chunks)
        self._save_documents()

    def _chunk_text(self, input_text: str) -> List[str]:
        """Splits a long text string into overlapping chunks."""
        return [
            input_text[i : i + self.chunk_size]
            for i in range(0, len(input_text), self.chunk_size - self.chunk_overlap)
        ]
