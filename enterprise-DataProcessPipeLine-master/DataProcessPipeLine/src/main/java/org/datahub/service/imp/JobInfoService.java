package org.datahub.service.imp;

import org.datahub.repository.JobInfoRepository;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.stereotype.Service;

@Service
public class JobInfoService {
    @Autowired
    private JobInfoRepository jobInfoRepository;

    public void updateJobResult(long jobId, String jobResult) {
        jobInfoRepository.updateJobResult(jobId, jobResult);
    }
}