package com.rag.service.retrieval;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import javax.annotation.PostConstruct;

import org.springframework.stereotype.Service;

import com.google.gson.JsonArray;
import com.google.gson.JsonObject;
import com.oracle.jrockit.jfr.DataType;

import io.milvus.v2.client.MilvusClientV2;
import io.milvus.v2.common.IndexParam;
import io.milvus.v2.service.collection.request.AddFieldReq;
import io.milvus.v2.service.collection.request.CreateCollectionReq;
import io.milvus.v2.service.vector.request.DeleteReq;
import io.milvus.v2.service.vector.request.InsertReq;
import io.milvus.v2.service.vector.request.QueryReq;
import io.milvus.v2.service.vector.request.SearchReq;
import io.milvus.v2.service.vector.response.QueryResp;
import io.milvus.v2.service.vector.response.SearchResp;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import main.java.com.rag.config.MilvusConfig;
import main.java.com.rag.model.DocumentChunk;
import main.java.com.rag.model.DocumentChunk.ChunkType;

/**
 * Milvus向量数据库服务
 * 负责文档块的向量存储和相似度检索
 * 支持父子索引：子块用于检索，父块通过parentChunkId关联
 */
@Slf4j
@Service
@RequiredArgsConstructor
public class MilvusVectorService {

    private final MilvusClientV2 milvusClient;
    private final MilvusConfig milvusConfig;

    private static final String FIELD_ID = "chunk_id";
    private static final String FIELD_PARENT_ID = "parent_chunk_id";
    private static final String FIELD_DOCUMENT_ID = "document_id";
    private static final String FIELD_CHUNK_TYPE = "chunk_type";
    private static final String FIELD_CONTENT = "content";
    private static final String FIELD_EMBEDDING = "embedding";

    /**
     * 应用启动时初始化Collection
     */
    @PostConstruct
    public void init() {
        try {
            ensureCollectionExists();
            log.info("Milvus Collection[{}]初始化完成", milvusConfig.getCollectionName());
        } catch (Exception e) {
            log.warn("Milvus初始化失败(服务可能未启动): {}", e.getMessage());
        }
    }

    /**
     * 确保Collection存在
     */
    private void ensureCollectionExists() {
        String collectionName = milvusConfig.getCollectionName();

        // 创建Collection Schema
        CreateCollectionReq.CollectionSchema schema = CreateCollectionReq.CollectionSchema.builder()
                .build();

        schema.addField(AddFieldReq.builder()
                .fieldName(FIELD_ID)
                .dataType(DataType.VarChar)
                .maxLength(256)
                .isPrimaryKey(true)
                .autoID(false)
                .build());

        schema.addField(AddFieldReq.builder()
                .fieldName(FIELD_PARENT_ID)
                .dataType(DataType.VarChar)
                .maxLength(256)
                .build());

        schema.addField(AddFieldReq.builder()
                .fieldName(FIELD_DOCUMENT_ID)
                .dataType(DataType.VarChar)
                .maxLength(256)
                .build());

        schema.addField(AddFieldReq.builder()
                .fieldName(FIELD_CHUNK_TYPE)
                .dataType(DataType.VarChar)
                .maxLength(20)
                .build());

        schema.addField(AddFieldReq.builder()
                .fieldName(FIELD_CONTENT)
                .dataType(DataType.VarChar)
                .maxLength(8192)
                .build());

        schema.addField(AddFieldReq.builder()
                .fieldName(FIELD_EMBEDDING)
                .dataType(DataType.FloatVector)
                .dimension(milvusConfig.getEmbeddingDimension())
                .build());

        // 创建向量索引
        IndexParam indexParam = IndexParam.builder()
                .fieldName(FIELD_EMBEDDING)
                .indexType(IndexParam.IndexType.valueOf(milvusConfig.getIndexType()))
                .metricType(IndexParam.MetricType.valueOf(milvusConfig.getMetricType()))
                .extraParams(Map.of("nlist", milvusConfig.getNlist()))
                .build();

        CreateCollectionReq request = CreateCollectionReq.builder()
                .collectionName(collectionName)
                .collectionSchema(schema)
                .indexParams(List.of(indexParam))
                .build();

        try {
            milvusClient.createCollection(request);
            log.info("创建Milvus Collection[{}]成功", collectionName);
        } catch (Exception e) {
            // Collection可能已存在
            log.debug("Collection[{}]可能已存在: {}", collectionName, e.getMessage());
        }
    }

    /**
     * 批量插入文档块（含向量）
     *
     * @param chunks 文档块列表（必须已包含embedding）
     */
    public void batchInsert(List<DocumentChunk> chunks) {
        if (chunks == null || chunks.isEmpty())
            return;

        String collectionName = milvusConfig.getCollectionName();
        List<JsonObject> data = new ArrayList<>();

        for (DocumentChunk chunk : chunks) {
            JsonObject row = new JsonObject();
            row.addProperty(FIELD_ID, chunk.getChunkId());
            row.addProperty(FIELD_PARENT_ID, chunk.getParentChunkId() != null ? chunk.getParentChunkId() : "");
            row.addProperty(FIELD_DOCUMENT_ID, chunk.getDocumentId());
            row.addProperty(FIELD_CHUNK_TYPE, chunk.getChunkType().name());
            row.addProperty(FIELD_CONTENT, chunk.getContent());

            JsonArray embeddingArray = new JsonArray();
            if (chunk.getEmbedding() != null) {
                for (float v : chunk.getEmbedding()) {
                    embeddingArray.add(v);
                }
            }
            row.add(FIELD_EMBEDDING, embeddingArray);
            data.add(row);
        }

        InsertReq insertReq = InsertReq.builder()
                .collectionName(collectionName)
                .data(data)
                .build();

        milvusClient.insert(insertReq);
        log.info("批量插入{}个文档块到Milvus", chunks.size());
    }

    /**
     * 向量相似度检索
     * 优先检索子块(CHILD)，通过parentChunkId关联父块
     *
     * @param queryEmbedding 查询向量
     * @param topK           返回数量
     * @param chunkType      检索的块类型
     * @return 检索结果列表
     */
    public List<VectorSearchResult> search(float[] queryEmbedding, int topK, ChunkType chunkType) {
        String collectionName = milvusConfig.getCollectionName();

        List<Float> queryVector = new ArrayList<>();
        for (float v : queryEmbedding) {
            queryVector.add(v);
        }

        SearchReq searchReq = SearchReq.builder()
                .collectionName(collectionName)
                .data(List.of(queryVector))
                .topK(topK)
                .filter(String.format("%s == \"%s\"", FIELD_CHUNK_TYPE, chunkType.name()))
                .outputFields(List.of(FIELD_ID, FIELD_PARENT_ID, FIELD_DOCUMENT_ID,
                        FIELD_CHUNK_TYPE, FIELD_CONTENT))
                .build();

        SearchResp searchResp = milvusClient.search(searchReq);

        List<VectorSearchResult> results = new ArrayList<>();
        List<List<SearchResp.SearchResult>> searchResults = searchResp.getSearchResults();
        for (List<SearchResp.SearchResult> resultList : searchResults) {
            for (SearchResp.SearchResult hit : resultList) {
                VectorSearchResult result = VectorSearchResult.builder()
                        .chunkId((String) hit.getEntity().get(FIELD_ID))
                        .parentChunkId((String) hit.getEntity().get(FIELD_PARENT_ID))
                        .documentId((String) hit.getEntity().get(FIELD_DOCUMENT_ID))
                        .content((String) hit.getEntity().get(FIELD_CONTENT))
                        .score(hit.getScore())
                        .build();
                results.add(result);
            }
        }

        log.debug("向量检索返回{}个结果", results.size());
        return results;
    }

    /**
     * 根据chunkId批量查询父块内容
     *
     * @param parentChunkIds 父块ID列表
     * @return 父块内容映射 chunkId -> content
     */
    public Map<String, String> batchGetParentContent(List<String> parentChunkIds) {
        if (parentChunkIds == null || parentChunkIds.isEmpty()) {
            return Collections.emptyMap();
        }

        String collectionName = milvusConfig.getCollectionName();

        // 构建查询过滤条件
        String filter = FIELD_ID + " in " + parentChunkIds;

        QueryReq queryReq = QueryReq.builder()
                .collectionName(collectionName)
                .filter(filter)
                .outputFields(List.of(FIELD_ID, FIELD_CONTENT))
                .build();

        QueryResp queryResp = milvusClient.query(queryReq);

        Map<String, String> parentContentMap = new HashMap<>();
        for (QueryResp.QueryResult queryResult : queryResp.getQueryResults()) {
            String chunkId = (String) queryResult.getEntity().get(FIELD_ID);
            String content = (String) queryResult.getEntity().get(FIELD_CONTENT);
            parentContentMap.put(chunkId, content);
        }

        return parentContentMap;
    }

    /**
     * 删除指定文档的所有块
     */
    public void deleteByDocumentId(String documentId) {
        String collectionName = milvusConfig.getCollectionName();
        String filter = String.format("%s == \"%s\"", FIELD_DOCUMENT_ID, documentId);

        DeleteReq deleteReq = DeleteReq.builder()
                .collectionName(collectionName)
                .filter(filter)
                .build();

        milvusClient.delete(deleteReq);
        log.info("删除文档[{}]的所有向量数据", documentId);
    }

    /**
     * 向量检索结果
     */
    @lombok.Data
    @lombok.Builder
    @lombok.NoArgsConstructor
    @lombok.AllArgsConstructor
    public static class VectorSearchResult {
        private String chunkId;
        private String parentChunkId;
        private String documentId;
        private String content;
        private double score;
    }
}