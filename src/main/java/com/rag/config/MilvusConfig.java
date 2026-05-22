package com.rag.config;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import io.milvus.v2.client.ConnectConfig;
import io.milvus.v2.client.MilvusClientV2;
import lombok.Data;

@Data
@Configuration
@ConfigurationProperties(prefix = "milvus")
public class MilvusConfig {

    private String host = "localhost";
    private int port = 19530;
    private String database = "default";
    private String collectionName = "rag_vectors";
    private int embeddingDimension = 1024;
    private String indexType = "IVF_FLAT";
    private String metricType = "COSINE";
    private int nlist = 1024;
    private int topK = 10;

    @Bean
    public MilvusClientV2 milvusClient() {
        ConnectConfig config = ConnectConfig.builder()
                .uri(String.format("http://%s:%d", host, port))
                .dbName(database)
                .build();
        return new MilvusClientV2(config);
    }
}