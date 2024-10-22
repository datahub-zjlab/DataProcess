package org.datahub.service.imp;

import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.datahub.model.Enum.JobStatus;
import org.datahub.model.JobInfoDTO;
import org.datahub.utils.ClusterScheduleUtil;
import org.datahub.utils.Imp.MySqlService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.BeanPropertyRowMapper;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowMapper;
import org.springframework.stereotype.Component;

import javax.annotation.Resource;
import java.sql.Timestamp;
import java.text.SimpleDateFormat;
import java.util.*;

/*
 * 1.定时去数据库读取所有正在运行的任务的开始时间和预估时间，若超时则杀死job对应的pod
 * */
@Component
@Slf4j
public class TaskScheduleService {
    @Resource(name="mysqlJdbcTemplate")
    private JdbcTemplate mysqlJdbcTemplate;

    @Resource
    private ClusterScheduleUtil clusterScheduleUtil;

    @Autowired
    private MySqlService mySqlService;
    private final int period = 60000;

    private final int statusCode = 10;

    @Setter
    @Getter
    private boolean isRunning =false;
    public void StartWatch() {
        try{
            new Timer().schedule(new TimerTask() {
                @Override
                public void run() {
                    try {
                        log.info("TaskScheduleService is running.");
                        //查询所有的运行中任务
                        String querySql = String.format("select * from data_process.job_info where job_status = %s",statusCode);
                        RowMapper<JobInfoDTO> rowMapper = new BeanPropertyRowMapper<>(JobInfoDTO.class);
                        List<JobInfoDTO> jobList = mysqlJdbcTemplate.query(querySql,rowMapper);
                        for (JobInfoDTO job:jobList){
                            if(job.getStart_time()!=null) {
                                long curTime = System.currentTimeMillis() / 1000;//s unix时间
                                long duration = curTime - job.getStart_time().getTime() / 1000;//s
                                if (duration > job.getTimeout()) {
                                    boolean isSuccess = clusterScheduleUtil.destroyPods(job.getId());
                                    if (isSuccess) {
                                        log.info(String.format("JobID %s TimeOut!Stop cluster %s all pods Successful!Timeout:%s,duration:%s,start_time:%s",
                                                job.getId(), job.getCluster_id(),job.getTimeout(),duration,job.getStart_time()));
                                    } else {
                                        log.error(String.format("JobID %s TimeOut!Stop cluster %s all pods Fail!Timeout:%s,duration:%s,start_time:%s",
                                                job.getId(), job.getCluster_id(),job.getTimeout(),duration,job.getStart_time()));
                                    }
                                    //更新数据库
                                    java.util.Date date= new Date();
                                    SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
                                    Timestamp timestamp=new Timestamp(date.getTime());
                                    String updateSql = String.format("UPDATE data_process.job_info SET job_status=%s,end_time='%s',exit_pod_num=total_pod_num" +
                                                    " where id =%s;", JobStatus.STOP.getIndex(), formatter.format(timestamp), job.getId());
                                    mysqlJdbcTemplate.update(updateSql);
                                }
                            } else {
                                log.warn(String.format("JobID %s lack of start_time in mysql table.",job.getId()));
                            }
                        }
                    } catch (Exception e) {
                        log.error(e.toString());
                    }
                }
            },0,period);
            log.info("TaskScheduleService start successful.");
            isRunning=true;
        }
        catch (Exception e){
            log.info("TaskScheduleService start fail.Exception:"+ e);
        }
    }
}
