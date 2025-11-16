from datetime import datetime

from dotenv import load_dotenv
import os

from elasticsearch import Elasticsearch

load_dotenv()


class ElasticSearchClient:
    def __init__(self):
        self.es_model_id = 'intfloat__multilingual-e5-base'
        try:
            # Elasticsearch setup
            # es_endpoint = os.environ.get('ELASTIC_ENDPOINT')
            self.es_client = Elasticsearch(
                # es_endpoint,
                cloud_id=os.environ.get('ELASTIC_CLOUD_ID'),
                api_key=(os.environ.get('ELASTIC_API_ID'), os.environ.get("ELASTIC_API_KEY"))
            )
        except Exception as e:
            self.es_client = None

    # Define a function to check ES status
    def es_ping(self, _input):
        if self.es_client is None:
            return "ES client is not initialized."
        else:
            try:
                if self.es_client.ping():
                    return "ES is connected."
                else:
                    return "ES is not connected."
            except Exception as e:
                return f"Error pinging ES: {e}"

    def site_search(self, query):
        response = self.es_client.search(
                index="site",
                knn={
                    "field": "information_embedding.predicted_value",
                    "query_vector_builder": {
                        "text_embedding": {
                            "model_id": self.es_model_id,
                            "model_text": f"query: {query}"
                        }
                    },
                    "k": 5,
                    "num_candidates": 20,
                }
        )

        formatted_results = []
        for hit in response["hits"]["hits"]:
            result = {
                "score": hit["_score"],
                "관광지명": hit["_source"]["name"],
                "주소": hit["_source"]["address"],
                "관광지설명": hit["_source"]["information"]
            }
            formatted_results.append(result)

        return formatted_results

    def culture_search(self, query):
        response = self.es_client.search(
                index="culture",
                knn={
                    "field": "information_embedding.predicted_value",
                    "query_vector_builder": {
                        "text_embedding": {
                            "model_id": self.es_model_id,
                            "model_text": f"query: {query}"
                        }
                    },
                    "k": 5,
                    "num_candidates": 20,
                },
                query={
                    "bool": {
                        "filter": [
                            {
                                "range": {
                                    "startDate": {
                                        "lte": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                                    }
                                }
                            },
                            {
                                "range": {
                                    "endDate": {
                                        "gte": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ")
                                    }
                                }
                            }
                        ]
                    }
                }
        )

        formatted_results = []
        for hit in response["hits"]["hits"]:
            result = {
                "score": hit["_score"],
                "이름": hit["_source"]["name"],
                "지역": hit["_source"]["area"],
                "주소": hit["_source"]["address"],
                "문화정보": hit["_source"]["information"],
                "가격정보": hit["_source"]["price"],
                "시작일": hit["_source"]["startDate"],
                "종료일": hit["_source"]["endDate"],
                "위치정보 : 위도": hit["_source"]['location']["lat"],
                "위치정보 : 경도": hit["_source"]["location"]["lon"]
            }
            formatted_results.append(result)

        return formatted_results
