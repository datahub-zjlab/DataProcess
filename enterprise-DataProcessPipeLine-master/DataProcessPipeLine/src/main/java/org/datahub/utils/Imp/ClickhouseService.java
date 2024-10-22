package org.datahub.utils.Imp;

import org.datahub.model.JobInfoDTO;
import org.datahub.utils.Interface.DatabaseService;
import org.springframework.jdbc.core.JdbcTemplate;
import org.springframework.stereotype.Service;

import javax.annotation.Resource;

@Service
public class ClickhouseService implements DatabaseService {
    @Resource
    private JdbcTemplate clickhouseJdbcTemplate;
    @Override
    public void save(JobInfoDTO jobInfo){
    }

    @Override
    public int update(JobInfoDTO jobInfo) {
        return 0;
    }

}

