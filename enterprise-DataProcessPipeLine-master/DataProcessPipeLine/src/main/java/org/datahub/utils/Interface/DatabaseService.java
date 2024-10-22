package org.datahub.utils.Interface;

import org.datahub.model.JobInfoDTO;

public interface DatabaseService {
     void save(JobInfoDTO jobInfo);
     int update(JobInfoDTO jobInfo) throws IllegalAccessException;
}
