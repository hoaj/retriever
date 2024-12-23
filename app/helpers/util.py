from langchain_community.document_loaders import JSONLoader
import json
from typing import Any, Dict, List
from langchain_core.documents import Document


class GobalUtil:
    @staticmethod
    def save_data_to_json(data: Any, output_file: str):
        class CustomEncoder(json.JSONEncoder):
            def default(self, obj):
                if hasattr(obj, "__dict__"):
                    return obj.__dict__
                return super().default(obj)

        # Check if data is a string and convert it to a dictionary
        if isinstance(data, str):
            data = {"text": data}

        with open(output_file, "w") as file:
            json.dump(data, file, indent=2, ensure_ascii=False, cls=CustomEncoder)

    @staticmethod
    def metadata_func(json_obj: Dict, default_metadata: Dict) -> Dict:
        metadata = json_obj.get("metadata", {})
        # metadata = default_metadata.copy()
        # metadata.update(json_obj.get("metadata", {}))
        return metadata

    @staticmethod
    def load_docs(file_path: str) -> List[Document]:
        loader = JSONLoader(
            file_path=file_path,
            jq_schema=".[]",
            metadata_func=GobalUtil.metadata_func,
            content_key=".page_content",
            is_content_key_jq_parsable=True,
        )

        data = loader.load()
        return data
