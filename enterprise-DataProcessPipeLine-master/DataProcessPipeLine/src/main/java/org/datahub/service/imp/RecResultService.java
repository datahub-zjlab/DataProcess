package org.datahub.service.imp;
/*
 * 1.接到pod处理结果消息，计入数据库*/

import lombok.extern.slf4j.Slf4j;
import org.datahub.model.TaskResultDTO;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.data.redis.core.StringRedisTemplate;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

import javax.annotation.Resource;
import java.time.LocalDate;
import java.time.format.DateTimeFormatter;
@Service
@Slf4j
public class RecResultService {
    @Resource
    private JdbcTemplate mysqlJdbcTemplate;

    @Autowired
    private StringRedisTemplate redisTemplate;

    public void run(TaskResultDTO taskResult){
        String sql = String.format("update data_process.job_info set success_task = success_task + %s ," +
                        "fail_task = fail_task + %s where id = %s", taskResult.getSuccessTask(),
                taskResult.getFailTask(), taskResult.getJobID());
        try {
            mysqlJdbcTemplate.update(sql);
            log.info("ReceiveTaskResult Update Mysql Successful! JobID: {}, SuccessTask: {}, FailTask: {}",
                    taskResult.getJobID(), taskResult.getSuccessTask(), taskResult.getFailTask());
        }
        catch (Exception e){
            log.error("ReceiveTaskResult Update Mysql Error! JobID:" + taskResult.getJobID() + "Error:" +e.toString());
        }
        // 获取当前日期并格式化为字符串
        String currentDate = LocalDate.now().format(DateTimeFormatter.ofPattern("yyyyMMdd"));

        // Redis keys
        String successKey = "successTaskCount:" + currentDate;
        String failKey = "failTaskCount:" + currentDate;

        // 更新 Redis 统计数据
        redisTemplate.opsForValue().increment(successKey, taskResult.getSuccessTask());
        redisTemplate.opsForValue().increment(failKey, taskResult.getFailTask());
    }
}
