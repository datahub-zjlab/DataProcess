package org.datahub.service.imp;

import org.datahub.config.K8sConfig;
import org.datahub.config.ZjcpConfig;
import org.datahub.domain.enums.JobTypeEnum;
import org.datahub.model.ProcessConfig;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpEntity;
import org.springframework.http.HttpHeaders;
import org.springframework.http.HttpMethod;
import org.springframework.http.MediaType;
import org.springframework.http.ResponseEntity;
import org.springframework.stereotype.Service;
import org.springframework.web.client.RestTemplate;

import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ThreadLocalRandom;

@Service
public class JobService {

    @Autowired
    private ZjcpConfig zjcpConfig;

    @Autowired
    private K8sConfig k8sConfig;

    private RestTemplate restTemplate = new RestTemplate();

    @Autowired
    private TokenService tokenService;

    private static final String BIZ_TYPE = "DROS";

    private static final Integer POD_NUMBER = 64;

    public String createJob(ProcessConfig processConfig) {
        String url = String.format("%s?userId=%s&bizType=%s",
                zjcpConfig.getServiceUrl() + zjcpConfig.getCreateJobUrl() ,
                zjcpConfig.getUserID(),
                BIZ_TYPE
        );
        String jobName = "datapipe_run" + getRandomNumber();
        // 设置请求头
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.APPLICATION_JSON);
        headers.set("accessToken", tokenService.getAccessToken());

        // 设置请求体
        Map<String, Object> requestBody = new HashMap<>();
        requestBody.put("userId", zjcpConfig.getUserID());

        Map<String, Object> jobMeta = new HashMap<>();
        jobMeta.put("jobName", jobName);
        jobMeta.put("describe", "一个推理作业");
        jobMeta.put("bizType", "DROS");
        jobMeta.put("jobType", "AI_INFERENCE");
        jobMeta.put("jobSpotType", "normal");
        jobMeta.put("subMissionId", zjcpConfig.getSubMissionId());
        jobMeta.put("subMissionName", "GeoCloud数据网络及计算引擎默认子任务");

        Map<String, Object> jobInfo = new HashMap<>();
        jobInfo.put("networkType", "default");
        jobInfo.put("image", zjcpConfig.getImage());

        Map<String, Object> volume = new HashMap<>();
        volume.put("volumeId", zjcpConfig.getVolumeId());
        volume.put("subPath", null);
        volume.put("mountPath", "/DATA/");
        volume.put("readOnly", false);
        jobInfo.put("volumes", new Map[]{volume});

        jobInfo.put("command",
                String.format("mkdir -p /root/.paddleocr/\n" +
                                "cp -rf /DATA/model/paddleocr/* /root/.paddleocr/\n" +
                                "mkdir -p /workspace/subject_textclf/\n" +
                                "cp /DATA/model/subject_fasttext/classifier_multi_subject.bin /workspace/subject_textclf/classifier_multi_subject.bin\n" +
                                "mkdir -p /root/.cache/torch/hub\n" +
                                "cp -rf /DATA/model/nougat/* /root/.cache/torch/hub/\n" +
                                "cd /workspace/\n" +
                                "bash start.sh \"%s\" \"%s\" \"0\" \"\" \"\" 5 \"\" \"%s\" \"zhijiang\"",
                        k8sConfig.getServiceUrl(),
                        processConfig.getJobID(),
                        Objects.requireNonNull(JobTypeEnum.parse(processConfig.getJobType())).getName()
                )
        );

        Map<String, Object> jobResource = new HashMap<>();
        jobResource.put("zoneType", "AI_GPU");
        jobResource.put("spec", "GPU_V100_32GB");
        jobResource.put("resourceType", "PUBLIC");
        jobResource.put("jobStartCount", POD_NUMBER);
        jobResource.put("gpu", 1);
        jobResource.put("cpu", 8);
        jobResource.put("memory", 32);

        requestBody.put("jobMeta", jobMeta);
        requestBody.put("jobInfo", jobInfo);
        requestBody.put("jobResource", jobResource);

        // 创建 HttpEntity
        HttpEntity<Map<String, Object>> entity = new HttpEntity<>(requestBody, headers);

        // 发送 POST 请求
        ResponseEntity<String> response = restTemplate.exchange(url, HttpMethod.POST, entity, String.class);

        if (response.getStatusCode().is2xxSuccessful()) {
            return response.getBody();
        } else {
            throw new RuntimeException("Failed to create job: HTTP error code " + response.getStatusCode());
        }
    }

    private int getRandomNumber() {
        int min = 200;
        int max = 10000;
        return getRandomIntInRange(min, max);
    }

    public static int getRandomIntInRange(int min, int max) {
        if (min > max) {
            throw new IllegalArgumentException("Max must be greater than or equal to min");
        }
        return ThreadLocalRandom.current().nextInt(min, max + 1);
    }

    public String deleteJob(String jobAiId) {
        String url = String.format("%s%s?userId=%s&bizType=%s&jobId=%s",
                zjcpConfig.getServiceUrl(),
                zjcpConfig.getDeleteJobUrl(),
                zjcpConfig.getUserID(),
                BIZ_TYPE,
                jobAiId);

        HttpHeaders headers = new HttpHeaders();
        headers.set("accessToken", tokenService.getAccessToken());
        HttpEntity<String> entity = new HttpEntity<>(headers);

        ResponseEntity<String> response = restTemplate.exchange(url, HttpMethod.DELETE, entity, String.class);
        return response.getBody();
    }
}

