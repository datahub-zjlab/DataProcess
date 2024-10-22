package org.datahub.utils.Imp;

import org.datahub.model.JobInfoDTO;
import org.datahub.utils.Interface.DatabaseService;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

import javax.annotation.Resource;
import java.lang.reflect.Field;

@Service
public class MySqlService implements DatabaseService {
    @Resource
    private JdbcTemplate mysqlJdbcTemplate;

    @Override
    public void save(JobInfoDTO jobInfo){
        String sql = String.format("Insert Into data_process.job_info (job_name,job_params,output_path," +
                        "dataset_id,job_status,job_type,start_time,end_time,task_user,deleted) Values ('%s','%s','%s',%s,%s,%s,%s," +
                        "%s,%s,%s);", jobInfo.getJob_name(), jobInfo.getJob_params(), jobInfo.getOutput_path(), jobInfo.getDataset_id(),
                jobInfo.getJob_status(), jobInfo.getJob_type(), jobInfo.getStart_time(), jobInfo.getEnd_time(), jobInfo.getTask_user(), 0);
        mysqlJdbcTemplate.update(sql);
    }

    @Override
    public int update(JobInfoDTO jobInfo) throws IllegalAccessException {
        StringBuilder sql = new StringBuilder("UPDATE data_process.job_info SET ");
        Field[] fields = jobInfo.getClass().getDeclaredFields();

        for (Field field:fields){
            field.setAccessible(true);
            Object value = field.get(jobInfo);
            String name = field.getName();
            if( value != null){
                sql.append(String.format("%s = %s,", name, value));
            }
        }

        sql = new StringBuilder(sql.substring(0, sql.length() - 1));
        sql.append("where id = ").append(jobInfo.getId()).append(";");
        return mysqlJdbcTemplate.update(sql.toString());
    }

    public int queryJobID(){
        return mysqlJdbcTemplate.queryForObject(String.format("SELECT id FROM data_process.job_info ORDER BY create_time DESC LIMIT 1;"), Integer.class);
    }
    public JobInfoDTO queryJobByID(Long jobID){
        return mysqlJdbcTemplate.queryForObject(String.format("SELECT id FROM data_process.job_info WHERE id=s%;",jobID), JobInfoDTO.class);
    }

    public int updateSQL(String sql){
        return mysqlJdbcTemplate.update(sql);
    }
}
