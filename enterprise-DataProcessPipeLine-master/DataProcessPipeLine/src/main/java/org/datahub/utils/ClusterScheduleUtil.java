package org.datahub.utils;

import io.kubernetes.client.openapi.ApiException;
import io.kubernetes.client.openapi.models.*;
import io.kubernetes.client.util.Yaml;
import org.datahub.model.JobInfo;
import org.datahub.repository.JobInfoRepository;
import org.datahub.service.imp.JobService;
import org.datahub.service.imp.JobInfoService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Value;
import lombok.extern.slf4j.Slf4j;
import org.apache.pulsar.shade.org.apache.commons.lang3.StringUtils;
import org.datahub.config.K8sConfig;
import org.datahub.domain.enums.JobTypeEnum;
import org.datahub.model.ProcessConfig;
import org.springframework.scheduling.annotation.Async;
import org.springframework.stereotype.Service;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import javax.annotation.Resource;
import java.io.IOException;
import java.util.Collections;
import java.util.HashMap;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.TimeUnit;
import java.util.List;
import java.util.ArrayList;

@Service
@Slf4j
public class ClusterScheduleUtil {
    @Autowired
    private JobService jobService;

    @Autowired
    private JobInfoService jobInfoService;

    @Autowired
    private JobInfoRepository jobInfoRepository;
    @Resource
    private K8sConfig k8sConfig;

    @Value("${environment}")
    private String environment;

    public static final String ZHIJIANG = "zhijiang";
    public static final String YUNQI = "yunqi";

    public static final Integer CREATE_POD_NUMBER = 200;

    public static final int SLEEP_INTERVAL_MINUTES = 1; // 以分钟为单位

    public int startPods(int podNum, ProcessConfig processConfig) {
        if (ZHIJIANG.equals(environment)) {
            startPodsByZj(podNum, processConfig);
        } else if (YUNQI.equals(environment)) {
            return startPodsByYq(podNum, processConfig);
        } else {
            throw new IllegalArgumentException("Unknown environment: " + environment);
        }
        return 1; // 返回一个默认值
    }

    public int startPodsByYq(int podNum, ProcessConfig processConfig) {
        try {
            Map<String, Object> placeholders = new HashMap<>();
            placeholders.put("JobID", processConfig.getJobID());
            placeholders.put("PodNumber", podNum);
            placeholders.put("ServiceURL", k8sConfig.getServiceUrl());
            placeholders.put("ParseType", processConfig.getParseMethod() != null ? processConfig.getParseMethod() : "");
            placeholders.put("TagType", processConfig.getMarkMethod() != null ? processConfig.getMarkMethod() : "");
            placeholders.put("CleanType", processConfig.getCleanMethod() != null ? processConfig.getCleanMethod() : "");
            placeholders.put("ThreadNum", 8);
            placeholders.put("OutputPath", processConfig.getOutputPath() != null ? processConfig.getOutputPath() : "");
            placeholders.put("Environment", environment);
            placeholders.put("JobTypeName", Objects.requireNonNull(JobTypeEnum.parse(processConfig.getJobType())).getName());
            placeholders = Collections.unmodifiableMap(placeholders);

            String basePath = System.getProperty("user.dir") + k8sConfig.getK8sConfigRelativeFilePath();
            String templatePath = basePath + k8sConfig.getK8sConfigJobFilePath();

            String generatedFile = KubernetesYamlClientUtil.replacePlaceholders(templatePath, placeholders);
            KubernetesYamlClientUtil.applyYaml(generatedFile, "batch/v1", "Job", V1Job.class);

        } catch (IOException | ApiException e) {
            e.printStackTrace();
            return 0;
        }
        return 1;
    }

    @Async
    public void startPodsByZj(int podNum, ProcessConfig processConfig) {
        // 针对之江智算的话 要记得sleep，而且podNum要除以64，但优先根据常量来好了，比如常量就是200，因为单次启动可以64个最大容器启动
        int totalPods = Math.min(podNum, CREATE_POD_NUMBER);
        int loops = (int) Math.ceil(totalPods / 64.0);
        int sleepIntervalMillis = SLEEP_INTERVAL_MINUTES * 60 * 1000;
        log.info("jobID: {}, loop: {}", processConfig.getJobID(), loops);
        ObjectMapper objectMapper = new ObjectMapper();
        List<String> jobIds = new ArrayList<>();
        for (int i = 0; i < loops; i++) {
            String responseBody = jobService.createJob(processConfig);
            log.info("jobID: {}, responseBody: {}", processConfig.getJobID(), responseBody);
            try {
                JsonNode jsonNode = objectMapper.readTree(responseBody);
                String jobId = jsonNode.path("data").path("jobId").asText();
                jobIds.add(jobId);
            } catch (Exception e) {
                log.error("Error parsing responseBody", e);
            }
            // 仅在需要时才进行睡眠，检查是否还有后续循环
            if (i < loops - 1) {
                try {
                    TimeUnit.MILLISECONDS.sleep(sleepIntervalMillis);
                } catch (InterruptedException e) {
                    Thread.currentThread().interrupt();
                    throw new RuntimeException("Thread was interrupted", e);
                }
            }
        }
        // 将 jobIds 存储到数据库中
        String jobResults = String.join(",", jobIds);
        log.info("jobID: {}, jobResults: {}", processConfig.getJobID(), jobResults);
        // 使用适当的方式将 jobResults 存储到 job_info 表中，使用JPA
        jobInfoService.updateJobResult(processConfig.getJobID(), jobResults);
    }

    public V1JobList getNamespaceJobs(String namespace) throws ApiException {
        return KubernetesYamlClientUtil.getNamespaceJobs(namespace);
    }

    public boolean destroyPods(Long jobId) {
        if (ZHIJIANG.equals(environment)) {
            return deleteJobByZj(jobId);
        } else if (YUNQI.equals(environment)) {
            return deleteJobAndPods(jobId);
        } else {
            throw new IllegalArgumentException("Unknown environment: " + environment);
        }
    }

    public boolean deleteJobAndPods(long jobId) {
        try {
            String basePath = System.getProperty("user.dir") + k8sConfig.getK8sConfigRelativeFilePath();
            String templatePath = basePath + k8sConfig.getK8sConfigJobFilePath();
            String generatedFile = getGeneratedFile(jobId, 0, templatePath);
            if (!StringUtils.isEmpty(generatedFile)) {
                V1Job job = Yaml.loadAs(generatedFile, V1Job.class);
                V1Status v1Status = KubernetesYamlClientUtil.deleteJob(job.getMetadata().getName(), job.getMetadata().getNamespace());
                if ("Faliure".equals(v1Status.getStatus())) {
                    log.error("deleteJob error,reason:\n" + v1Status.getReason());
                    return false;
                }
                //如果job都没删除成功，pod也不管了，让运维删吧
                V1PodList podsForJob = KubernetesYamlClientUtil.getPodsForJob(job.getMetadata().getName(), job.getMetadata().getNamespace());
                for (V1Pod v1Pod : podsForJob.getItems()) {
                    //todo 目前删除失败会抛异常被捕获，暂不判断code（没这个字段）
                    V1Pod pod = KubernetesYamlClientUtil.deletePod(v1Pod.getMetadata().getName(), v1Pod.getMetadata().getNamespace());
                }
                return true;
            }
            return false;
        } catch (ApiException | IOException e) {
            log.error("deleteJob error", e);
            return false;
        }
    }


    public String getGeneratedFile(Long jobId, Integer podNum, String templatePath) throws IOException {

        Map<String, Object> placeholders = new HashMap<>();
        placeholders.put("JobID", jobId);
        placeholders.put("PodNumber", podNum);
        placeholders = Collections.unmodifiableMap(placeholders);

        return KubernetesYamlClientUtil.replacePlaceholders(templatePath, placeholders);
    }

    public boolean deleteJobByZj(long jobId) {
        // 现根据jobId搜索jobInfo表拿到job_result字段 判断如果为空的话就抛异常，如果不为空的话就去调用jobService.deleteJob方法
        JobInfo jobInfo = jobInfoRepository.findById(jobId).orElseThrow(() -> new RuntimeException("Job not found with id: " + jobId));

        // 判断job_result字段是否为空
        if (jobInfo.getJobResult() == null || jobInfo.getJobResult().isEmpty()) {
            throw new RuntimeException("Job result is empty for job id: " + jobId);
        }

        // 调用jobService.deleteJob方法
        String response = jobService.deleteJob(jobInfo.getJobResult());
        return response.contains("\"code\":200");
    }
}