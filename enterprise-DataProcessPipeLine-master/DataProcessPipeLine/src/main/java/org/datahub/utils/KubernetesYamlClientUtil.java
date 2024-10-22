package org.datahub.utils;

import io.kubernetes.client.openapi.ApiClient;
import io.kubernetes.client.openapi.ApiException;
import io.kubernetes.client.openapi.Configuration;
import io.kubernetes.client.openapi.apis.AppsV1Api;
import io.kubernetes.client.openapi.apis.BatchV1Api;
import io.kubernetes.client.openapi.apis.CoreV1Api;
import io.kubernetes.client.openapi.models.*;
import io.kubernetes.client.util.ClientBuilder;
import io.kubernetes.client.util.KubeConfig;
import io.kubernetes.client.util.Yaml;
import lombok.extern.slf4j.Slf4j;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.List;
import java.util.Map;

import static java.io.File.separator;

@Slf4j
public class KubernetesYamlClientUtil {
    private static ApiClient client;
    private static CoreV1Api coreV1Api;
    private static AppsV1Api appsV1Api;
    private static BatchV1Api batchV1Api;

    static {
        KubeConfig kubeconfig = null;
        try {

            kubeconfig = KubeConfig.loadKubeConfig(new FileReader(System.getProperty("user.dir") + "/target/classes" + separator + "k8s-kubeconfig.yml"));
            client = ClientBuilder
                    .kubeconfig(kubeconfig)
                    .build();
            Configuration.setDefaultApiClient(client);
            coreV1Api = new CoreV1Api();
            appsV1Api = new AppsV1Api();
            batchV1Api = new BatchV1Api();
        } catch (Exception e) {
            log.error("获取kubeconfig配置文件内容错误！", e);
        }
    }

    public static void applyYaml(String content, String apiVersion, String kind, Class<?> clazz) throws ApiException, IOException {
        Yaml.addModelMap(apiVersion, kind, clazz);
        // 加载生成的 YAML 文件
        List<Object> resources = Yaml.loadAll(content);

        for (Object resource : resources) {
            if (resource instanceof V1Namespace) {
                applyNamespace((V1Namespace) resource);
            } else if (resource instanceof V1Pod) {
                applyPod((V1Pod) resource);
            } else if (resource instanceof V1Service) {
                applyService((V1Service) resource);
            } else if (resource.getClass().equals(clazz)) {
                applyJob((V1Job) resource);
            } else {
                log.error("Unsupported resource type: " + resource.getClass().getName());
            }
        }
    }


//    public static List<Object> getConfig(String Content) {
//        try {
//            return Yaml.loadAll(Content);
//        } catch (Exception e) {
//            e.printStackTrace();
//            throw new RuntimeException("Failed to apply YAML file: " + e.getMessage(), e);
//        }
//    }

//    public static String getGeneratedYamlFile(String yamlFilePath) {
//        try {
//            return new String(Files.readAllBytes(Paths.get(yamlFilePath)));
//        } catch (IOException e) {
//            throw new RuntimeException(e);
//        }
//    }

    public static void applyNamespace(V1Namespace namespace) throws ApiException {
        coreV1Api.createNamespace(namespace, null, null, null, null);
    }

    public static void deleteNamespace(String namespace) throws ApiException {
        coreV1Api.deleteNamespace(namespace, null, null, null, null, null, null);
    }

    public static void getAllNamespaces(V1Namespace namespace) throws ApiException {
        coreV1Api.listNamespace(null, null, null, null, null, null, null, null, null, null);
    }

    public static void applyPod(V1Pod pod) throws ApiException {
        coreV1Api.createNamespacedPod(pod.getMetadata().getNamespace(), pod, null, null, null, null);
    }

    public static V1PodList getPods(String namespace) throws ApiException {
        return coreV1Api.listNamespacedPod(namespace, null, null, null, null, null, null, null, null, null, null);
    }
    public static V1PodList getPodsForJob(String  jobName, String namespace) {
        String labelSelector = String.format("job-name=%s",jobName);
        try {
            return coreV1Api.listNamespacedPod(namespace, null, null, null, null, labelSelector, null, null, null, null, null);
        } catch (Exception e) {
            throw new RuntimeException("Error listing pods for job", e);
        }
    }
    public static V1Pod deletePod(String podName, String namespace) throws ApiException {
        return coreV1Api.deleteNamespacedPod(podName, namespace, null, null, null, null, null, null);
    }

    public static void applyService(V1Service service) throws ApiException {
        coreV1Api.createNamespacedService(service.getMetadata().getNamespace(), service, null, null, null, null);
    }

    public static void applyJob(V1Job job) throws ApiException {
        batchV1Api.createNamespacedJob(job.getMetadata().getNamespace(), job, null, null, null, null);
    }

    public static V1JobList getNamespaceJobs(String namespace) throws ApiException {
        return batchV1Api.listNamespacedJob(namespace, null, null, null, null, null, null, null, null, null, null);
    }

    public static V1Status deleteJob(String jobName, String namespace) throws ApiException {
        return batchV1Api.deleteNamespacedJob(jobName, namespace, null, null, null, null, null, null);
    }

    public static String replacePlaceholders(String templatePath, Map<String, Object> placeholders) throws IOException {
        String content = new String(Files.readAllBytes(Paths.get(templatePath)));

        for (Map.Entry<String, Object> entry : placeholders.entrySet()) {
            content = content.replace("{{." + entry.getKey() + "}}", entry.getValue().toString());
        }
        return content;
    }

    public static void writeFile(String content, String outputPath) throws IOException {
        try (FileWriter writer = new FileWriter(outputPath)) {
            writer.write(content);
        }
    }
}
