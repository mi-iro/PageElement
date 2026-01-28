import requests
import json


class Qwen3VLReranker:
    def __init__(self, model_name_or_path):
        self.base_url = model_name_or_path.rstrip('/')
        self.models_url = f"{self.base_url}/v1/models"
        self.rerank_url = f"{self.base_url}/rerank"
        self.headers = {"accept": "application/json", "Content-Type": "application/json"}
        self.model_name = None

    def _get_model_name(self):
        """Lazy load the model name from the server."""
        if self.model_name is None:
            try:
                response = requests.get(self.models_url, headers=self.headers)
                response.raise_for_status()
                self.model_name = response.json()["data"][0]["id"]
            except Exception as e:
                raise RuntimeError(f"Failed to fetch model name from {self.models_url}: {e}")
        return self.model_name

    def _format_document(self, doc_item: dict) -> dict:
        """Converts user input document format to vLLM API compatible format."""
        if "text" in doc_item:
            return {
                "type": "text",
                "text": doc_item["text"]
            }
        elif "image" in doc_item:
            return {
                "type": "image_url",
                "image_url": {
                    "url": "file://" + doc_item["image"]
                }
            }
        else:
            raise ValueError(f"Unknown document format: {doc_item}")

    def process(self, inputs: dict):
        """
        Processes the reranking request.

        Args:
            inputs (dict): Dictionary containing 'query' and 'documents'.

        Returns:
            list: The 'results' list from the API response containing scores.
        """
        model = self._get_model_name()

        # Parse and format query
        query_text = inputs.get("query", {}).get("text", "")

        # Parse and format documents
        formatted_documents = []
        for doc in inputs.get("documents", []):
            formatted_documents.append(self._format_document(doc))
        formatted_documents = {"content": formatted_documents}

        # Construct payload
        data = {
            "model": model,
            "query": query_text,
            "documents": formatted_documents,
        }

        # Send request
        try:
            response = requests.post(self.rerank_url, headers=self.headers, json=data)
            response.raise_for_status()
            result_json = response.json()
            scores = result_json.get("results", result_json)
            scores = [ float(ele["relevance_score"]) for ele in scores]
            return scores
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response text: {e.response.text}")
            return None


# --- Usage Example ---
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--api_url", type=str, default="http://localhost:8004")
    args, _ = parser.parse_known_args()

    client_reranker = Qwen3VLReranker(model_name_or_path=args.api_url)

    test_inputs = {
        "query": {"text": "A woman playing with her dog on a beach at sunset."},
        "documents": [
            {"text": "AA woman shares a joyful moment with her golden retriever on a sun-drenched beach at sunset, as the dog offers its paw in a heartwarming display of companionship and trust."},
            {"image": "/mnt/shared-storage-user/mineru3-share/wangzhengren/JIT-RAG/assets/images/beijing.jpg"},
        ],
    }

    try:
        scores = client_reranker.process(test_inputs)
        print(scores)
    except Exception as e:
        print(f"An error occurred: {e}")