package org.datahub.exception;

public class JobNotInProgressException extends RuntimeException {
    public JobNotInProgressException(String message) {
        super(message);
    }
}
