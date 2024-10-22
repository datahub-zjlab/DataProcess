package org.datahub.service.imp;

import com.aliyun.odps.account.Account;
import lombok.extern.slf4j.Slf4j;
import org.datahub.model.JobInfoDTO;
import org.springframework.jdbc.core.BeanPropertyRowMapper;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.stereotype.Service;

import javax.annotation.Resource;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Service
@Slf4j
public class JobProgressService {
    @Resource
    private JdbcTemplate mysqlJdbcTemplate;
    public Object run(int userId){
        String querySql = String.format("select * from data_process.job_info where task_user = %s",userId);
        RowMapper<JobInfoDTO> rowMapper = new BeanPropertyRowMapper<>(JobInfoDTO.class);
        List<Map<String,Long>> result  = new ArrayList<>();
        try{
            List<JobInfoDTO> jobList = mysqlJdbcTemplate.query(querySql,rowMapper);
            for (JobInfoDTO job:jobList){
                Map<String,Long> jobProgress = new HashMap<>();
                jobProgress.put("FailTask", (long) job.getFail_task());
                jobProgress.put("SuccessTask", (long) job.getSuccess_task());
                jobProgress.put("TotalTask", (long) job.getTotal_task());
                jobProgress.put("jobID", job.getId());
                jobProgress.put("jobStatus", (long) job.getJob_status());
                result.add(jobProgress);
            }
        }
        catch (Exception e){
            log.error("JobProgressService Exception!" + e.toString());
        }
        return result;
    };
}
