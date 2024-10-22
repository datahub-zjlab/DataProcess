package org.datahub.service.imp;

import com.google.common.collect.Maps;
import lombok.extern.slf4j.Slf4j;
import org.datahub.model.Const.ConstValue;
import org.datahub.model.Enum.JobStatus;
import org.datahub.model.ProcessConfig;
import org.datahub.service.Interface.TaskDispatchService;
import org.datahub.utils.ClusterScheduleUtil;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.jdbc.core.RowCallbackHandler;
import org.springframework.stereotype.Service;

import javax.annotation.Resource;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.Timestamp;
import java.text.SimpleDateFormat;
import java.util.*;
/*
 * 1.依据数据集名字查询数据库，获取待处理总量，计算预估处理用时timeout和pod数量
 * 2.通过v100集群api调用pod,具体接口待定，指定images、挂载、启动命令
 * */

@Service
@Slf4j
public class DataPipeLineService {
    @Resource(name = "clickhouseJdbcTemplate")
    protected JdbcTemplate clickhouseJdbcTemplate;

    @Resource(name = "mysqlJdbcTemplate")
    protected JdbcTemplate mysqlJdbcTemplate;

    private final Integer FETCH_SIZE = 100;
    @Autowired
    private TaskScheduleService taskScheduleService;

    @Autowired
    private TaskDispatchService taskDispatchService;

    @Autowired
    private ClusterScheduleUtil clusterScheduleUtil;

    public int run(ProcessConfig processConfig) {
        long total_num = taskDispatchService.getCountFromClickhouse(processConfig.getDataSetID());
        long total_time = total_num * ConstValue.timePerItem;
        int total_pods = (int) Math.ceil(total_num / ConstValue.numPerPod.doubleValue());
        try {
            int clusterID = clusterScheduleUtil.startPods(total_pods, processConfig);
            if (clusterID != 1) {
                log.error(String.format("Start %s pod Fail! Total task: %s ,time out: %s.", total_pods, total_num, total_time));
                return 0;
            }
            log.info(String.format("Start %s pod successful! Total task: %s ,time out: %s.", total_pods, total_num, total_time));
            java.util.Date date = new Date();
            SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
            Timestamp timestamp = new Timestamp(date.getTime());
            String sql = String.format("update data_process.job_info set total_task = %s ,total_pod_num = %s, timeout=%s, cluster_id = %s," +
                    "start_time = '%s' where id = %s;", total_num, total_pods, total_time, clusterID, formatter.format(timestamp), processConfig.getJobID());
            mysqlJdbcTemplate.update(sql);
            if (!taskScheduleService.isRunning()) {
                taskScheduleService.StartWatch();
            }
            return clusterID;
        } catch (Exception e) {
            log.error(String.format("Start %s pod Exception! Total task: %s ,time out: %s. Exception: %s", total_pods, total_num, total_time, e.toString()));
            return 0;
        }
    }

    public long queryCountByDataName(ProcessConfig processConfig) {
        Map<String, String> queriesInner = Optional.ofNullable(processConfig.getFileterRule()).orElseGet(Maps::newHashMap);
        StringBuilder sql = new StringBuilder(String.format("select count(*) as cnt from %s FINAL where 1 = 1 and Version = %s", "data_process_base", processConfig.getDataSetID()));
        queriesInner.forEach((k, v) -> sql.append(String.format(" and (%s='%s')", k, v)));

        List<Long> result = new ArrayList<>();
        batchStreamQuery(rs -> result.add(rs.getLong("cnt")), sql.toString());
        return result.get(0);
    }

    private void batchStreamQuery(RowCallbackHandler rch, String sql) {
        // 流io
        clickhouseJdbcTemplate.query(psc -> {
            PreparedStatement preparedStatement = psc.prepareStatement(sql, ResultSet.TYPE_FORWARD_ONLY, ResultSet.CONCUR_READ_ONLY);
            preparedStatement.setFetchSize(FETCH_SIZE);
            preparedStatement.setFetchDirection(ResultSet.FETCH_FORWARD);
            return preparedStatement;
        }, rch);
    }

    public boolean stop(Long jobID) {
        // 根据ID查询podID,停止所有pod
        try {
            String querySql = String.format("select cluster_id, job_status from data_process.job_info where id = %s", jobID);
            Map<String, Object> result = mysqlJdbcTemplate.queryForMap(querySql);
            int clusterId = (int) result.get("cluster_id");
            int jobStatus = (int) result.get("job_status");

            if (jobStatus == JobStatus.STOP.getIndex()) {
                log.warn("Job is already stopped! JobID:" + jobID);
                return false;
            }

            boolean isSuccess = false;
            if (clusterId > 0) {
                isSuccess = clusterScheduleUtil.destroyPods(jobID);
            }

            if (isSuccess) {
                log.info("Stop successful! JobID:" + jobID);
                java.util.Date date = new Date();
                SimpleDateFormat formatter = new SimpleDateFormat("yyyy-MM-dd HH:mm:ss");
                Timestamp timestamp = new Timestamp(date.getTime());
                String sql = String.format("update data_process.job_info set job_status = %d, exit_pod_num = total_pod_num, end_time = '%s' " +
                        "where id = %s;", JobStatus.STOP.getIndex(), formatter.format(timestamp), jobID);
                mysqlJdbcTemplate.update(sql);
                return true;
            } else {
                log.error("Stop Fail! JobID" + jobID);
            }
        } catch (Exception e) {
            log.error(String.format("Destroy pod Exception! JobID: %s ,Exception: %s", jobID, e));
        }
        return false;
    }
}
