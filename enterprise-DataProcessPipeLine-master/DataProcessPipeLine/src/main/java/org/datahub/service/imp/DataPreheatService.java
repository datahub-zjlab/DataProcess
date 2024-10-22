package org.datahub.service.imp;

import lombok.extern.slf4j.Slf4j;
import org.datahub.domain.enums.TaskDispatchEnums;
import org.datahub.domain.vo.TaskBatchVO;
import org.datahub.model.JobInfoDTO;
import org.datahub.service.Interface.TaskDispatchService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.scheduling.annotation.Scheduled;
import org.springframework.stereotype.Service;
import org.springframework.data.redis.core.StringRedisTemplate;
import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.core.JsonProcessingException;

import java.util.HashMap;
import java.util.Map;
import java.util.List;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.ConcurrentHashMap;

@Service
@Slf4j
public class DataPreheatService {

    @Autowired
    private StringRedisTemplate redisTemplate;
    private static final int PREHEAT_THRESHOLD = 300;
    private static final int PREHEAT_LIMIT = TaskDispatchEnums.TAKE_ONCE.getValue() * 500;

    @Autowired
    private TaskDispatchService taskDispatchService;

    // 定义每个 jobId 的预热间隔时间
    private static final Map<String, Integer> jobPreheatIntervals = new HashMap<>();

    static {
//        jobPreheatIntervals.put("110", 60000); // 每分钟执行一次
        jobPreheatIntervals.put("112", 300000); // 每5分钟执行一次
        // 继续添加其他 jobId 和对应的预热间隔时间
    }

    private Map<String, Long> lastRunTimes = new ConcurrentHashMap<>();

    @Scheduled(fixedRate = 10000) // 每10秒检查一次所有的job
    public void checkAndPreheatJobs() {
        for (Map.Entry<String, Integer> entry : jobPreheatIntervals.entrySet()) {
            String jobId = entry.getKey();
            int interval = entry.getValue();

            long currentTime = System.currentTimeMillis();
            lastRunTimes.putIfAbsent(jobId, 0L);
            long lastRunTime = lastRunTimes.get(jobId);
            log.info("Checking job {} duration {}, interval {}", jobId, currentTime - lastRunTime, interval);
            if (currentTime - lastRunTime >= interval) {
                preheatData(jobId);
                lastRunTimes.put(jobId, currentTime);
            }
        }
    }

    public void preheatData(String jobId) {
        log.info("Preheat data is running...");
//        String jobId = "110";
        String preheatOffsetKey = "preheat_offset_" + jobId;
        // 获取 offset 并增加 TAKE_ONCE
        String takenKey = "taken_" + jobId;

        String preheatOffsetStr = redisTemplate.opsForValue().get(preheatOffsetKey);
        String currentOffsetStr = redisTemplate.opsForValue().get(takenKey);

        int preheatOffset = (preheatOffsetStr == null) ? 0 : Integer.parseInt(preheatOffsetStr);
        int currentOffset = (currentOffsetStr == null) ? 0 : Integer.parseInt(currentOffsetStr);
        // 如果预热 offset 值不存在，则进行首次预热
        if (preheatOffsetStr == null) {
            // 6655200
            preheatOffset = currentOffset;
            redisTemplate.opsForValue().set(preheatOffsetKey, String.valueOf(preheatOffset));
        }
        int remainingBatches = (preheatOffset - currentOffset) / TaskDispatchEnums.TAKE_ONCE.getValue();
        log.info("JobID {}, preheatOffset {}, currentOffset {}, remainingBatches {}", jobId, preheatOffset, currentOffset, remainingBatches);
        if (remainingBatches < PREHEAT_THRESHOLD) {
            int newPreheatOffset = preheatOffset + PREHEAT_LIMIT;
            JobInfoDTO jobInfo = taskDispatchService.getJobInfo(jobId);
            List<TaskBatchVO> dataList = taskDispatchService.getDataFromClickhouse(jobInfo.getDataset_id(), preheatOffset, PREHEAT_LIMIT);
            for (int i = 0; i < dataList.size(); i += TaskDispatchEnums.TAKE_ONCE.getValue()) {
                int end = Math.min(i + TaskDispatchEnums.TAKE_ONCE.getValue(), dataList.size());
                List<TaskBatchVO> batch = dataList.subList(i, end);
                String serializedData = serializeBatchData(batch);
                redisTemplate.opsForValue().set("data_batch_" + jobId + ":" + (preheatOffset + i), serializedData, 10, TimeUnit.HOURS);
            }

            redisTemplate.opsForValue().set(preheatOffsetKey, String.valueOf(newPreheatOffset));
            log.info("JobID {}, Preheated {} batches of data into Redis, remainingBatches {}", jobId, PREHEAT_LIMIT / TaskDispatchEnums.TAKE_ONCE.getValue(), remainingBatches + 500);
        } else {
            log.info("JobID {}, remainingBatches {}, Sufficient preheated data available, no need for additional preheat.", jobId, remainingBatches);
        }
    }

    public void deleteKeysByPattern(String basePattern, long start, long end, long increment) {
//        deleteKeysByPattern("data_batch_110:", 10019250, 12000000, TaskDispatchEnums.TAKE_ONCE.getValue());
        for (long i = start; i <= end; i += increment) {
            String key = basePattern + i;
            Boolean result = redisTemplate.delete(key);
            if (result != null && result) {
                log.info("Deleted key: {}", key);
            } else {
                log.info("Key not found or deletion failed: {}", key);
            }
        }
    }

    private String serializeBatchData(List<TaskBatchVO> dataList) {
        ObjectMapper mapper = new ObjectMapper();
        try {
            return mapper.writeValueAsString(dataList);
        } catch (JsonProcessingException e) {
            throw new RuntimeException("Failed to serialize batch data to JSON", e);
        }
    }

//    @Scheduled(fixedRate = 60000)
    public void test() {
        extendExpirationByPattern("data_batch_110:", 40415750, 40666000, TaskDispatchEnums.TAKE_ONCE.getValue());
    }
    public void extendExpirationByPattern(String basePattern, long start, long end, long increment) {
        // Iterate through the keys matching the base pattern with the specified range and increment
        for (long i = start; i <= end; i += increment) {
            String key = basePattern + i;
            Boolean hasKey = redisTemplate.hasKey(key);
            if (hasKey != null && hasKey) {
                Boolean result = redisTemplate.expire(key, 24, TimeUnit.HOURS);
                if (result != null && result) {
                    log.info("Extended expiration for key: {}", key);
                } else {
                    log.info("Failed to extend expiration for key: {}", key);
                }
            } else {
                log.info("Key not found: {}", key);
            }
        }
    }
}

