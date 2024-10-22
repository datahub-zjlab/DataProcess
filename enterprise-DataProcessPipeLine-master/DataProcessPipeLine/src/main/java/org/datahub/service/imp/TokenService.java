package org.datahub.service.imp;

import org.datahub.config.ZjcpConfig;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;
import lombok.extern.slf4j.Slf4j;

import javax.annotation.Resource;
import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.TimeUnit;

@Service
@Slf4j
public class TokenService {

    @Resource
    private ZjcpConfig zjcpConfig;

    @Autowired
    private StringRedisTemplate redisTemplate;

    private RestTemplate restTemplate = new RestTemplate();

    private static final String ACCESS_TOKEN_KEY = "zjcp:accessToken";

    public String getAccessToken() {
        String accessToken = redisTemplate.opsForValue().get(ACCESS_TOKEN_KEY);
        if (accessToken == null) {
            synchronized (this) {
                accessToken = redisTemplate.opsForValue().get(ACCESS_TOKEN_KEY);
                if (accessToken == null) {
                    accessToken = requestNewAccessToken();
                    redisTemplate.opsForValue().set(ACCESS_TOKEN_KEY, accessToken, 1, TimeUnit.HOURS);
                }
            }
        }
        log.info("get accessToken: {}", accessToken);
        return accessToken;
    }

    private String requestNewAccessToken() {
        // 设置请求头
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);

        // 设置请求体
        Map<String, String> requestBody = new HashMap<>();
        requestBody.put("appKey", zjcpConfig.getAppKey());
        requestBody.put("appSecret", zjcpConfig.getAppSecret());

        // 创建 HttpEntity
        HttpEntity<Map<String, String>> entity = new HttpEntity<>(requestBody, headers);
        String url = zjcpConfig.getServiceUrl() + zjcpConfig.getTokenUrl();
        // 发送 POST 请求
        ResponseEntity<Map> response = restTemplate.exchange(url, HttpMethod.POST, entity, Map.class);

        if (response.getStatusCode().is2xxSuccessful() && response.getBody() != null) {
            Map<String, Object> responseBody = response.getBody();
            if ((int) responseBody.get("code") == 0) {
                Map<String, String> data = (Map<String, String>) responseBody.get("data");
                return data.get("accessToken");
            } else {
                throw new RuntimeException("Failed to get access token: " + responseBody.get("msg"));
            }
        } else {
            throw new RuntimeException("Failed to get access token: HTTP error code " + response.getStatusCode());
        }
    }
}

