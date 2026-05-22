package com.rag.config;

import org.apache.http.HttpHost;
import org.elasticsearch.client.RestClient;
import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

import co.elastic.clients.elasticsearch.ElasticsearchClient;
import co.elastic.clients.json.jackson.JacksonJsonpMapper;
import co.elastic.clients.transport.rest_client.RestClientTransport;
import lombok.Data;

@Data
@Configuration
@ConfigurationProperties(prefix = "elasticsearch")
public class ElasticsearchConfig {

    private String host = "localhost";
    private int port = 9200;
    private String scheme = "http";
    private String username = "elastic";
    private String password = "changeme";
    private String indexName = "rag_bm25";
    private int topK = 10;

    @Bean
    public ElasticsearchClient elasticsearchClient() {
        RestClient restClient = RestClient.builder(
                new HttpHost(host, port, scheme)).build();

        RestClientTransport transport = new RestClientTransport(
                restClient, new JacksonJsonpMapper());
        return new ElasticsearchClient(transport);
    }
}