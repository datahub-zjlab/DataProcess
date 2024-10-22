package org.datahub.service.imp;

import com.fly.framework.web.domain.DataResponse;
import lombok.extern.slf4j.Slf4j;
import org.datahub.domain.enums.TaskDispatchEnums;
import org.datahub.domain.vo.TaskBatchVO;
import org.datahub.exception.ConcurrentProcessingException;
import org.datahub.exception.JobNotInProgressException;
import org.datahub.exception.NoDataAvailableException;
import org.datahub.model.JobInfoDTO;
import org.datahub.service.Interface.TaskDispatchService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.jdbc.core.BeanPropertyRowMapper;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.type.TypeReference;
import com.fasterxml.jackson.core.JsonProcessingException;
import com.fasterxml.jackson.databind.JsonNode;
import com.fasterxml.jackson.databind.ObjectMapper;

import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.concurrent.TimeUnit;

@Service
@Slf4j
public class TaskDispatchServiceImpl implements TaskDispatchService {
    @Autowired
    private StringRedisTemplate redisTemplate;

    @Autowired
    private JdbcTemplate clickhouseJdbcTemplate;

    @Autowired
    private JdbcTemplate mysqlJdbcTemplate;

    public DataResponse handleGetBatchTask(String jobId) throws JobNotInProgressException, NoDataAvailableException {
        // 加锁防止并发
        boolean lock = setIfAbsent("lock_" + jobId);
        if (!lock) {
            throw new ConcurrentProcessingException("Task is being processed by another request");
        }

        try {
            // 查询 MySQL 获取任务信息
            JobInfoDTO jobInfo = getJobInfo(jobId);
            validateJobStatus(jobInfo);

            // 查询 ClickHouse 获取数据条数
            String datasetId = jobInfo.getDataset_id();
            int count = getCountFromCacheOrClickhouse(datasetId);
            checkDataAvailability(count);

            // 获取 offset 并增加 TAKE_ONCE
            String takenKey = "taken_" + jobId;
            String offsetStr = getValueFromRedis(takenKey);
            int offset = (offsetStr == null) ? 0 : Integer.parseInt(offsetStr);
            int newOffset = offset + TaskDispatchEnums.TAKE_ONCE.getValue();

            // 构造返回结果
            Map<String, Object> result = prepareResult(offset, newOffset, count, datasetId, jobId);
            // 更新 offset 到 Redis
            if (offset < count) {
                setValueInRedis(takenKey, String.valueOf(newOffset));
            }
            log.info("Taken JobID: {}, offset: {}, Limit: {}", jobId, offset, TaskDispatchEnums.TAKE_ONCE.getValue());
            return DataResponse.success("Task data retrieved successfully", result);
        } finally {
            // 释放锁
            unLockFromRedis("lock_" + jobId);
        }
    }

    private void validateJobStatus(JobInfoDTO jobInfo) throws JobNotInProgressException {
        if (jobInfo == null || jobInfo.getJob_status() != 10) {
            throw new JobNotInProgressException("Job is not in progress or does not exist");
        }
    }

    private void checkDataAvailability(int count) throws NoDataAvailableException {
        if (count == 0) {
            throw new NoDataAvailableException("No data available for processing");
        }
    }

    private Map<String, Object> prepareResult(int offset, int newOffset, int count, String datasetId, String jobId) {
        Map<String, Object> result = new LinkedHashMap<>();
        result.put("takenStart", offset);
        result.put("takenEnd", newOffset);
        if (offset >= count) {
            result.put("existsTask", false);
            return result;
        }

        List<TaskBatchVO> dataList;
        String batchData = redisTemplate.opsForValue().get("data_batch_" + jobId + ":" + offset);
        if (batchData != null) {
            log.info("JobID {}, Batch data found in Redis, offset: {}, Limit: {}", jobId, offset, TaskDispatchEnums.TAKE_ONCE.getValue());
            dataList = deserializeBatchData(batchData);
        } else {
            // 兜底
            log.info("JobID {}, Batch data not found in Redis, fetching from ClickHouse, offset: {}, Limit: {}", jobId, offset, TaskDispatchEnums.TAKE_ONCE.getValue());
            dataList = getDataFromClickhouse(datasetId, offset, TaskDispatchEnums.TAKE_ONCE.getValue());
        }

        result.put("existsTask", !dataList.isEmpty());
        result.put("list", dataList);
        return result;
    }

    private List<TaskBatchVO> deserializeBatchData(String batchData) {
        ObjectMapper mapper = new ObjectMapper();
        try {
            return mapper.readValue(batchData, new TypeReference<List<TaskBatchVO>>() {
            });
        } catch (JsonProcessingException e) {
            throw new RuntimeException("Failed to deserialize batch data from Redis", e);
        }
    }

    private String getValueFromRedis(String key) {
        return redisTemplate.opsForValue().get(key);
    }


    private void unLockFromRedis(String key) {
        redisTemplate.delete(key);
    }


    private boolean setIfAbsent(String key) {
        return Boolean.TRUE.equals(redisTemplate.opsForValue().setIfAbsent(key, "locked", 10, TimeUnit.SECONDS));
    }


    private void setValueInRedis(String key, String value) {
        redisTemplate.opsForValue().set(key, value);
    }


    public List<TaskBatchVO> getDataFromClickhouseOld(String version, int offset, int limit) {
        // 根据你的 ClickHouse 数据库和表结构，编写相应的 SQL 语句
        String sql = "SELECT * FROM process.data_process_base WHERE Version = ? ORDER BY ID LIMIT ? OFFSET ?";
        return clickhouseJdbcTemplate.query(sql, new Object[]{version, limit, offset}, (rs, rowNum) ->
                new TaskBatchVO(
                        rs.getString("ID"),
                        rs.getString("Version"),
                        rs.getString("DOI"),
                        rs.getString("Path"),
                        rs.getString("PdfPath"),
                        rs.getString("Attribute2"),
                        rs.getString("Attribute2"),
                        rs.getString("Attribute2")
                ));
    }

    public List<TaskBatchVO> getDataFromClickhouse(String version, int offset, int limit) {
        // 定义 SQL 查询语句
        String sql = "SELECT * FROM process.data_process_base WHERE Version = ? ORDER BY ID LIMIT ? OFFSET ?";

        // 使用 ObjectMapper 处理 JSON 解析
        ObjectMapper objectMapper = new ObjectMapper();

        return clickhouseJdbcTemplate.query(sql, new Object[]{version, limit, offset}, (rs, rowNum) -> {
            // 从结果集中提取字段
            String id = rs.getString("ID");
            String versionValue = rs.getString("Version");
            String doi = rs.getString("DOI");
            String path = rs.getString("Path");
            String pdfPath = rs.getString("PdfPath");
            String attribute2 = rs.getString("Attribute2");  // 保持为原始 JSON 格式

            // 如果 attribute2 是 JSON 字符串，尝试解析出 extension
            String fileExtension = null;
            String language = null;
            if (attribute2 != null && !attribute2.isEmpty()) {
                try {
                    JsonNode jsonNode = objectMapper.readTree(attribute2);
                    // 提取 extension 属性
                    fileExtension = jsonNode.has("extension") ? jsonNode.get("extension").asText() : null;
                    language = jsonNode.has("language") ? jsonNode.get("language").asText() : null;
                } catch (Exception e) {
                    e.printStackTrace();
                    // 处理解析错误
                }
            }

            // 返回 TaskBatchVO 对象，将原始的 attribute2 JSON 和解析出的 extension 一同传入
            return new TaskBatchVO(id, versionValue, doi, path, pdfPath, attribute2, fileExtension, language);
        });
    }


    public int getCountFromClickhouse(String version) {
        String sql = "SELECT COUNT(*) FROM process.data_process_base FINAL WHERE Version = ?";
        return clickhouseJdbcTemplate.queryForObject(sql, new Object[]{version}, Integer.class);
    }


    public JobInfoDTO getJobInfo(String jobId) {
        String sql = "SELECT * FROM job_info WHERE id = ? LIMIT 1";
        return mysqlJdbcTemplate.queryForObject(sql, new Object[]{jobId}, new BeanPropertyRowMapper<>(JobInfoDTO.class));
    }

    private int getCountFromCacheOrClickhouse(String datasetId) {
        String cacheKey = "count_" + datasetId;
        String countStr = redisTemplate.opsForValue().get(cacheKey);
        if (countStr != null) {
            return Integer.parseInt(countStr);
        } else {
            int count = getCountFromClickhouse(datasetId);
            redisTemplate.opsForValue().set(cacheKey, String.valueOf(count));
            return count;
        }
    }
}
