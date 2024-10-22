package org.datahub.config;

import com.aliyun.oss.OSS;
import com.aliyun.oss.OSSClientBuilder;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;

/**
 * @author maliming
 */
@Configuration
public class OssConfig {
    private static String endpoint = "http://oss-cn-hangzhou-zjy-d01-a.ops.cloud.zhejianglab.com/";
    private static String accessKeyId = "kJscloSzed09Lhy7";
    private static String accessKeySecret = "mQyqefxOLd7SPUPKiTam3JYsHhut12";

    @Bean
    public OSS getOss(){
        return new OSSClientBuilder().build(endpoint, accessKeyId, accessKeySecret);
    }
}
