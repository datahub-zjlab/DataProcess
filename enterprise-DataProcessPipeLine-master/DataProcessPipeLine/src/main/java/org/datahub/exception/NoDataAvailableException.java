package org.datahub.exception;

public class NoDataAvailableException extends RuntimeException {
    public NoDataAvailableException(String message) {
        super(message);
    }
}
