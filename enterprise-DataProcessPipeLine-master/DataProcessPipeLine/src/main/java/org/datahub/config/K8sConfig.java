package org.datahub.config;

import lombok.Getter;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.Configuration;

/**
 * @author maliming
 */
@Configuration
@Getter
public class K8sConfig {
    public static String NAMESPACE = "client-sdk-test";
    public static String IMAGE = "dhub.kubesre.xyz/library/busybox:latest";
    @Value("${k8sConfig.serviceUrl}")
    private String serviceUrl;
    @Value("${k8sConfig.file.path.relative}")
    private String k8sConfigRelativeFilePath;

    @Value("${k8sConfig.file.path.job}")
    private String k8sConfigJobFilePath;

}
