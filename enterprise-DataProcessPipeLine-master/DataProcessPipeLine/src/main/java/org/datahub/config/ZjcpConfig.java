package org.datahub.config;

import lombok.Getter;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;

@Configuration
@Getter
public class ZjcpConfig {
    @Value("${zjcpConfig.urls.service}")
    private String serviceUrl;
    @Value("${zjcpConfig.urls.token}")
    private String tokenUrl;
    @Value("${zjcpConfig.urls.createJob}")
    private String createJobUrl;
    @Value("${zjcpConfig.urls.deleteJob}")
    private String deleteJobUrl;
    @Value("${zjcpConfig.user.id}")
    private String userID;
    @Value("${zjcpConfig.job.subMissionId}")
    private String subMissionId;
    @Value("${zjcpConfig.job.image}")
    private String image;
    @Value("${zjcpConfig.job.volumeId}")
    private String volumeId;
    @Value("${zjcpConfig.credentials.appKey}")
    private String appKey;
    @Value("${zjcpConfig.credentials.appSecret}")
    private String appSecret;

}
