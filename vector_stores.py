from langchain_community.vectorstores import Chroma

import config_data


class VectorStoreService(object):
    def __init__(self,embedding):
        self.embedding = embedding

        self.vector_store = Chroma(
            collection_name=config_data.collection_name,
            embedding_function=self.embedding,
            persist_directory=config_data.persist_directory,
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": config_data.similarity_threshold})

if __name__ == "__main__":
    from langchain_community.embeddings import DashScopeEmbeddings
    retriever = VectorStoreService(DashScopeEmbeddings(dashscope_api_key=config_data.dashscope_api_key,
                                                       model="text-embedding-v4")).get_retriever()

    res = retriever.invoke("我体重180斤，尺码推荐")
    print(res)