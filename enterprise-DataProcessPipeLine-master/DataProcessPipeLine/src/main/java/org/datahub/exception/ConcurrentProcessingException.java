package org.datahub.exception;

public class ConcurrentProcessingException extends RuntimeException {
    public ConcurrentProcessingException(String message) {
        super(message);
    }
}

